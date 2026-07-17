#!/bin/bash
# Richer real-cluster sweep (paper pivot, Phase 1) — statistical robustness.
#
# All 16 LLM configs (4 models x 4 variants) + 2 real baselines (HPA, KEDA),
# each run N times, on TWO load traces against the same CPU-bound app:
#   - cpu_bursty  : traces/trace_cpu.npy   (synthetic, sharp bursts)
#   - wiki_diurnal: traces/trace_wiki.npy  (real Wikipedia hourly, diurnal+weekly)
#
# Each run is real-time: STEPS x INTERVAL seconds. Resume-friendly: complete runs
# are skipped, so the horizon/repeats can be extended later with larger STEPS/REPS.
#
# Can run as one sequential sweep (STREAM=A, default) against workload-cpu, or as
# two concurrent streams against two isolated deployments — see STREAM/PARITY and
# APP_DEPLOYMENT/DEPLOY_OVERRIDE_ARG below. Stream B needs workload-cpu-2 already
# applied and pinned to its own node (see k8s/workload-cpu-2.yaml).
#
# Usage:
#   nohup setsid bash run_richer_real.sh >> logs/richer_real_main.log 2>&1 &
#   STREAM=B APP_DEPLOYMENT=workload-cpu-2 DEPLOY_OVERRIDE_ARG="--deployment-override workload-cpu-2" \
#     nohup setsid bash run_richer_real.sh >> logs/richer_real_main_B.log 2>&1 &
set -u
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
AUTOSCALER="$SCRIPT_DIR/k8s_autoscaler.py"
LOADGEN="$SCRIPT_DIR/load_generator.py"
OUTBASE="$SCRIPT_DIR/results_richer"
LOGDIR="$SCRIPT_DIR/logs"
KEYS_FILE="$SCRIPT_DIR/api_keys.conf"
NTFY_TOPIC="llm-autoscaler-faraz-2026"

# ---- 2-stream parallel sweep support ----
# STREAM=A (default) drives workload-cpu on worker1+worker2.
# STREAM=B drives a 2nd isolated deployment (workload-cpu-2, pinned to node0) via
# --deployment-override, so both streams can run concurrently without colliding.
# Combos are split by parity of their position in the (identical, deterministic)
# scenario/rep/config enumeration order below, so each incomplete combo is claimed
# by exactly one stream — no double-writes to the same output CSV.
STREAM="${STREAM:-A}"
case "$STREAM" in
    A) PARITY=0 ;;
    B) PARITY=1 ;;
    *) echo "STREAM must be A or B"; exit 1 ;;
esac
APP_DEPLOYMENT="${APP_DEPLOYMENT:-workload-cpu}"
DEPLOY_OVERRIDE_ARG="${DEPLOY_OVERRIDE_ARG:-}"

# ---- Phase-1 budget (chosen: N=3, 120-step, cap-14, sequential ~9 days) ----
STEPS=120
INTERVAL=60
SCALE_FACTOR=1.0
REPS="1 2 3"
# cap 14 (was 20) — shared-cluster ceiling. Both traces peak at ~2199 rps, which
# needs ~12 replicas at the SLA boundary, so cap-14 clears peak demand (no SLA
# floor) while its max footprint (14 x 250m = 3.5 vCPU) fits the ~4 vCPU freed by
# limiting the other tenants (blackbox/hv-phase0) to ~1 core. No dedicated nodes
# available; isolation is provided externally by that tenant CPU cap, not taints.
MAX_REPLICAS=14
WORKLOAD_ARG="cpu"              # k8s_autoscaler --workload (deployment + prompt context)
# scenario_name:trace_file  (both drive the SAME app, cap 20)
SCENARIOS=(
    "cpu_bursty:traces/trace_cpu.npy"
    "wiki_diurnal:traces/trace_wiki.npy"
)
VARIANTS="zero_shot domain history_5 cot"

mkdir -p "$OUTBASE" "$LOGDIR"

notify() { curl -s -d "[$STREAM] $1" "ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; echo "$(date -u +%H:%M:%S) [NOTIFY] [$STREAM] $1"; }

# ---- provider keys ----
declare -A PROVIDER_KEYS
while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# || -z "$key" ]] && continue
    PROVIDER_KEYS[$provider]="$key"
done < "$KEYS_FILE"
# Stream B runs concurrently with stream A — give it its own nvidia quota bucket
# (nvidia2, otherwise only used as a failover key) instead of sharing stream A's.
if [ "$STREAM" = "B" ] && [ -n "${PROVIDER_KEYS[nvidia2]:-}" ]; then
    echo "$(date -u): STREAM B — using nvidia2 as primary nvidia key"
    PROVIDER_KEYS[nvidia]="${PROVIDER_KEYS[nvidia2]}"
fi
echo "$(date -u): [$STREAM] providers: ${!PROVIDER_KEYS[*]}"

RUNS=()
assign() {
    local model=$1; shift
    for p in "$@"; do
        if [ -n "${PROVIDER_KEYS[$p]:-}" ]; then RUNS+=("$model:$p"); echo "  $model -> $p"; return 0; fi
    done
    echo "  WARNING: no provider for $model (tried $*)"; return 1
}
echo "$(date -u): assigning LLMs to providers..."
assign llama-8b       nvidia groq
assign llama-70b      nvidia groq sambanova
assign mistral-small4 nvidia
# gpt-oss-120b REPLACES qwen3-80b in this sweep (Jul 15): same open weights on
# nvidia/cerebras/groq, nvidia primary (healthy, no deprecation banner).
assign gpt-oss-120b   nvidia cerebras groq
# qwen3-80b REMOVED Jul 15: nvidia retires this API 07/27/2026 and its backend
# already hangs to 504 on every call (llama on the same key answers in <1s).
# No point burning ~2h of stream time per config on a dead endpoint.
# assign qwen3-80b      nvidia

env_var_for() {
    case "$1" in
        groq) echo GROQ_API_KEY;; nvidia) echo NVIDIA_API_KEY;;
        cerebras) echo CEREBRAS_API_KEY;; sambanova) echo SAMBANOVA_API_KEY;;
        google) echo GOOGLE_API_KEY;; novita) echo NOVITA_API_KEY;;
        dashscope) echo DASHSCOPE_API_KEY;; siliconflow) echo SILICONFLOW_API_KEY;;
        openrouter) echo OPENROUTER_API_KEY;;
    esac
}

is_complete() {  # $1=outfile
    [ -f "$1" ] && [ -s "$1" ] && [ "$(wc -l < "$1")" -ge $((STEPS + 1)) ]
}

# ---- preflight: cluster + app reachable ----
if ! kubectl get deployment "$APP_DEPLOYMENT" >/dev/null 2>&1; then
    notify "ABORT: deployment $APP_DEPLOYMENT not found — provision the testbed app first"
    echo "ERROR: deployment $APP_DEPLOYMENT not found. Is the testbed up?"; exit 1
fi
SVC_IP=$(kubectl get svc "$APP_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
[ -z "$SVC_IP" ] && { echo "ERROR: no clusterIP for svc $APP_DEPLOYMENT"; exit 1; }
echo "$(date -u): app $APP_DEPLOYMENT @ $SVC_IP, max-replicas $MAX_REPLICAS (shared-cluster ceiling)"

# ---- run one (scenario, rep, model, variant, provider) ----
run_one() {
    local scen=$1 trace=$2 rep=$3 model=$4 variant=$5 provider=$6
    local outdir="$OUTBASE/$scen/rep$rep"
    local tag="${scen}/rep${rep}/${model}_${variant}"
    local outfile="$outdir/k8s_${WORKLOAD_ARG}_${model}_${variant}.csv"
    mkdir -p "$outdir"

    if is_complete "$outfile"; then echo "$(date -u +%H:%M:%S) SKIP $tag (complete)"; return 0; fi

    echo "$(date -u +%H:%M:%S) START $tag ($provider) [$COMPLETED/$TOTAL]"
    notify "Start: $tag ($provider) [$COMPLETED/$TOTAL]"

    kubectl scale deployment "$APP_DEPLOYMENT" --replicas=3 >/dev/null 2>&1
    sleep 15

    local llog="$LOGDIR/richer_load_${scen}_rep${rep}_${model}_${variant}.log"
    local alog="$LOGDIR/richer_auto_${scen}_rep${rep}_${model}_${variant}.log"

    $PYTHON "$LOADGEN" --service-ip "$SVC_IP" --trace "$SCRIPT_DIR/$trace" \
        --steps $STEPS --interval $INTERVAL --scale-factor $SCALE_FACTOR \
        --output "$outdir/load_${WORKLOAD_ARG}_${model}_${variant}.csv" > "$llog" 2>&1 &
    local LOAD_PID=$!

    local extra=""
    case "$model" in hpa|keda|rl-dqn|rl-ppo) ;; *) extra="--provider $provider";; esac
    local ev; ev=$(env_var_for "$provider")
    # pass a 2nd nvidia key (nvidia2 in api_keys.conf) for daily-quota failover —
    # only meaningful for stream A, since stream B's "nvidia" key IS nvidia2.
    local ev2=()
    if [ "$provider" = "nvidia" ] && [ "$STREAM" = "A" ] && [ -n "${PROVIDER_KEYS[nvidia2]:-}" ]; then
        ev2=("NVIDIA_API_KEY_2=${PROVIDER_KEYS[nvidia2]}")
    fi
    # pass keys of providers that host the exact same model, so the
    # autoscaler can fall back to them when every primary key is limited.
    local fallbacks=""
    case "$model" in
        qwen3-80b)    fallbacks="siliconflow openrouter novita dashscope";;
        gpt-oss-120b) fallbacks="cerebras groq";;
    esac
    for fb in $fallbacks; do
        [ "$fb" = "$provider" ] && continue
        [ -n "${PROVIDER_KEYS[$fb]:-}" ] && ev2+=("$(env_var_for "$fb")=${PROVIDER_KEYS[$fb]}")
    done

    env "${ev:-DUMMY_KEY}=${PROVIDER_KEYS[$provider]:-}" "${ev2[@]}" \
        $PYTHON "$AUTOSCALER" --workload "$WORKLOAD_ARG" --model "$model" --variant "$variant" \
        --trace "$SCRIPT_DIR/$trace" --steps $STEPS --interval $INTERVAL \
        --max-replicas $MAX_REPLICAS \
        --output-dir "$outdir" --resume $extra $DEPLOY_OVERRIDE_ARG > "$alog" 2>&1 &
    local AUTO_PID=$!

    wait $LOAD_PID $AUTO_PID 2>/dev/null
    if is_complete "$outfile"; then
        COMPLETED=$((COMPLETED + 1)); notify "Done: $tag [$COMPLETED/$TOTAL]"
    else
        notify "FAILED: $tag — see $(basename "$alog")"
    fi
}

# ---- count total ----
# GRAND_TOTAL = both streams combined (whole sweep). TOTAL = this stream's own
# share, computed by walking the exact same IDX/parity enumeration as the main
# loop below — this is what COMPLETED is compared against in notifications, so
# each lane reports progress against its own workload, not the combined one.
GRAND_TOTAL=0
TOTAL=0
COMPLETED=0
_count_idx=0
for _scen_spec in "${SCENARIOS[@]}"; do
    _scen="${_scen_spec%%:*}"
    for _rep in $REPS; do
        _outdir="$OUTBASE/$_scen/rep$_rep"
        for _b in hpa keda; do
            _count_idx=$((_count_idx + 1))
            GRAND_TOTAL=$((GRAND_TOTAL + 1))
            if [ $((_count_idx % 2)) -eq $PARITY ]; then
                TOTAL=$((TOTAL + 1))
                is_complete "$_outdir/k8s_${WORKLOAD_ARG}_${_b}_baseline.csv" && COMPLETED=$((COMPLETED + 1))
            fi
        done
        for _spec in "${RUNS[@]}"; do
            _model="${_spec%%:*}"
            for _v in $VARIANTS; do
                _count_idx=$((_count_idx + 1))
                # wiki llama-8b belongs to the groq offload lane; not ours
                if [ "$_scen" = "wiki_diurnal" ] && [ "$_model" = "llama-8b" ]; then continue; fi
                GRAND_TOTAL=$((GRAND_TOTAL + 1))
                if [ $((_count_idx % 2)) -eq $PARITY ]; then
                    TOTAL=$((TOTAL + 1))
                    is_complete "$_outdir/k8s_${WORKLOAD_ARG}_${_model}_${_v}.csv" && COMPLETED=$((COMPLETED + 1))
                fi
            done
        done
    done
done
unset _count_idx _scen_spec _scen _rep _outdir _b _spec _model _v

echo "$(date -u): [$STREAM] resuming at $COMPLETED/$TOTAL already complete this lane ($GRAND_TOTAL grand total, $STEPS steps x ${INTERVAL}s = $((STEPS*INTERVAL/60))min each, sequential)"
echo "$(date -u): rough wall-clock ~$((TOTAL * STEPS * INTERVAL / 3600)) h for this lane"
notify "Richer real-cluster sweep resumed: $COMPLETED/$TOTAL already done this lane ($GRAND_TOTAL grand total, N=3, ${STEPS}-step, max-rep $MAX_REPLICAS, cpu+wiki)"

# ---- priority pass: mistral-small4 x wiki_diurnal FIRST ----
# nvidia retires the mistral-small4 API on 07/27/2026 (same date as qwen3-80b,
# whose backend is already dead). These 12 runs are the only remaining
# mistral-small4 configs, so grab them before the regular sweep order reaches
# them. Walks the exact same enumeration and parity as the main loop below, so
# every config keeps its usual stream owner and the two streams stay disjoint;
# the main loop skips whatever this pass completed via is_complete.
echo "$(date -u): ===== PRIORITY PASS: mistral-small4 x wiki_diurnal ====="
PIDX=0
for scen_spec in "${SCENARIOS[@]}"; do
    scen="${scen_spec%%:*}"; trace="${scen_spec#*:}"
    for rep in $REPS; do
        for b in hpa keda; do
            PIDX=$((PIDX + 1))
        done
        for spec in "${RUNS[@]}"; do
            model="${spec%%:*}"; provider="${spec##*:}"
            for v in $VARIANTS; do
                PIDX=$((PIDX + 1))
                if [ "$scen" = "wiki_diurnal" ] && [ "$model" = "mistral-small4" ]; then
                    [ $((PIDX % 2)) -ne $PARITY ] && continue
                    run_one "$scen" "$trace" "$rep" "$model" "$v" "$provider" || true
                fi
            done
        done
    done
done

# ---- main: scenarios x reps x configs ----
# STREAM A and STREAM B iterate this identical deterministic order and each only
# acts on combos whose position (IDX) matches their parity — this is what makes
# the 2-stream split disjoint (see header comment) without any shared locking.
IDX=0
for scen_spec in "${SCENARIOS[@]}"; do
    scen="${scen_spec%%:*}"; trace="${scen_spec#*:}"
    echo "$(date -u): ===== SCENARIO $scen ($trace) ====="
    for rep in $REPS; do
        echo "$(date -u): --- $scen rep $rep ---"
        # baselines first (no API) — HPA + KEDA only: real rule-based controllers,
        # ZERO simulator dependency. Sim-trained RL (DQN/PPO) intentionally excluded.
        # The "perfect" oracle baseline is computed post-hoc from real capacity
        # (compute_oracle_baseline.py), not run live.
        for b in hpa keda; do
            IDX=$((IDX + 1))
            [ $((IDX % 2)) -ne $PARITY ] && continue
            run_one "$scen" "$trace" "$rep" "$b" "baseline" "none" || true
        done
        # LLMs
        for spec in "${RUNS[@]}"; do
            model="${spec%%:*}"; provider="${spec##*:}"
            for v in $VARIANTS; do
                IDX=$((IDX + 1))
                # wiki llama-8b belongs to the groq offload lane
                # (run_altprovider_lane.sh, still live) — never touch it here,
                # or the two could write the same CSV (lane D incident)
                if [ "$scen" = "wiki_diurnal" ] && [ "$model" = "llama-8b" ]; then continue; fi
                [ $((IDX % 2)) -ne $PARITY ] && continue
                run_one "$scen" "$trace" "$rep" "$model" "$v" "$provider" || true
            done
        done
    done
done

notify "Richer real-cluster sweep COMPLETE: $COMPLETED/$TOTAL runs. Results in results_richer/"
echo "$(date -u): ALL DONE — $COMPLETED/$TOTAL. Results in $OUTBASE/"
