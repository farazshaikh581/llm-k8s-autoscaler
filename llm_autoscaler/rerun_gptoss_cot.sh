#!/bin/bash
# Rerun gpt-oss-120b/cot only, with the max_tokens/parse-fallback fix
# (k8s_autoscaler.py: max_tokens 512->1536, unparseable response now holds
# current replicas instead of clamping to REPLICA_MIN). The old max_tokens=512
# runs truncated gpt-oss-120b's reasoning mid-response on overloaded steps
# (finish_reason="length", content=None), and the parser silently fell back
# to REPLICA_MIN=1 -- driving 5s latencies and ~50-63% success right when
# scale-up was needed most. Fixed data archived the old runs to
# results_richer/_buggy_gptoss120b_cot_maxtokens512/ (not deleted).
#
# Waits for the live run_richer_real.sh (PID given via WAIT_PID) to release
# the shared workload-cpu deployment before touching it -- running concurrently
# would double-write the same CSVs (see feedback_lane_scheduling_collision).
#
# Usage: WAIT_PID=<pid> nohup setsid bash rerun_gptoss_cot.sh >> logs/rerun_gptoss_cot_main.log 2>&1 &
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
APP_DEPLOYMENT="workload-cpu"

STEPS=120
INTERVAL=60
SCALE_FACTOR=1.0
MAX_REPLICAS=14
WORKLOAD_ARG="cpu"

mkdir -p "$OUTBASE" "$LOGDIR"

notify() { curl -s -d "[cot-fix] $1" "ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true; echo "$(date -u +%H:%M:%S) [NOTIFY] $1"; }

declare -A PROVIDER_KEYS
while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# || -z "$key" ]] && continue
    PROVIDER_KEYS[$provider]="$key"
done < "$KEYS_FILE"

is_complete() { [ -f "$1" ] && [ -s "$1" ] && [ "$(wc -l < "$1")" -ge $((STEPS + 1)) ]; }

# (scenario:trace:rep) -- the 5 configs archived above. wiki_diurnal/rep3 is
# NOT here: it hasn't run yet in the live sweep, so it gets the fix for free.
CONFIGS=(
    "cpu_bursty:traces/trace_cpu.npy:1"
    "cpu_bursty:traces/trace_cpu.npy:2"
    "cpu_bursty:traces/trace_cpu.npy:3"
    "wiki_diurnal:traces/trace_wiki.npy:1"
    "wiki_diurnal:traces/trace_wiki.npy:2"
)

if [ -n "${WAIT_PID:-}" ]; then
    echo "$(date -u): waiting for PID $WAIT_PID (live richer-real sweep) to release $APP_DEPLOYMENT..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "$(date -u): PID $WAIT_PID exited, proceeding"
fi

SVC_IP=$(kubectl get svc "$APP_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
[ -z "$SVC_IP" ] && { echo "ERROR: no clusterIP for svc $APP_DEPLOYMENT"; exit 1; }
echo "$(date -u): app $APP_DEPLOYMENT @ $SVC_IP -- rerunning gpt-oss-120b/cot x${#CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
    scen="${cfg%%:*}"; rest="${cfg#*:}"; trace="${rest%%:*}"; rep="${rest##*:}"
    outdir="$OUTBASE/$scen/rep$rep"
    outfile="$outdir/k8s_${WORKLOAD_ARG}_gpt-oss-120b_cot.csv"
    tag="${scen}/rep${rep}/gpt-oss-120b_cot"
    mkdir -p "$outdir"

    if is_complete "$outfile"; then echo "$(date -u +%H:%M:%S) SKIP $tag (already re-done)"; continue; fi

    echo "$(date -u +%H:%M:%S) START $tag (rerun, fixed max_tokens+fallback)"
    notify "Start rerun: $tag"

    kubectl scale deployment "$APP_DEPLOYMENT" --replicas=3 >/dev/null 2>&1
    sleep 15

    llog="$LOGDIR/rerun_cot_load_${scen}_rep${rep}.log"
    alog="$LOGDIR/rerun_cot_auto_${scen}_rep${rep}.log"

    $PYTHON "$LOADGEN" --service-ip "$SVC_IP" --trace "$SCRIPT_DIR/$trace" \
        --steps $STEPS --interval $INTERVAL --scale-factor $SCALE_FACTOR \
        --output "$outdir/load_${WORKLOAD_ARG}_gpt-oss-120b_cot.csv" > "$llog" 2>&1 &
    LOAD_PID=$!

    env NVIDIA_API_KEY="${PROVIDER_KEYS[nvidia]:-}" \
        NVIDIA_API_KEY_2="${PROVIDER_KEYS[nvidia2]:-}" \
        CEREBRAS_API_KEY="${PROVIDER_KEYS[cerebras]:-}" \
        GROQ_API_KEY="${PROVIDER_KEYS[groq]:-}" \
        $PYTHON "$AUTOSCALER" --workload "$WORKLOAD_ARG" --model gpt-oss-120b --variant cot \
        --trace "$SCRIPT_DIR/$trace" --steps $STEPS --interval $INTERVAL \
        --max-replicas $MAX_REPLICAS --output-dir "$outdir" --resume --provider nvidia > "$alog" 2>&1 &
    AUTO_PID=$!

    wait $LOAD_PID $AUTO_PID 2>/dev/null
    if is_complete "$outfile"; then
        notify "Done rerun: $tag"
    else
        notify "FAILED rerun: $tag -- see $(basename "$alog")"
    fi
done

notify "gpt-oss-120b/cot rerun COMPLETE (${#CONFIGS[@]} configs)"
echo "$(date -u): ALL RERUNS DONE"
