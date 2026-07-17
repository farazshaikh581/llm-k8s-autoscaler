#!/bin/bash
# New-model-family screening sweep (Jul 2026): a "light real run" -- 1 rep,
# 2 prompt variants (zero_shot, domain), cpu_bursty trace only -- for 5
# candidates identified as zero-new-credential (already-held Groq/Cerebras/
# NVIDIA keys), to see if any beat/match the RL-PPO/RL-DQN/HPA/KEDA baselines
# before committing to the full N=3 x 4-variant x 2-trace treatment.
#
# Two isolated lanes so a slow/stuck model in one family doesn't block the
# other. Lane 1 = known-good providers (Groq, Cerebras). Lane 2 = the two new
# NVIDIA NIM models, including DeepSeek-V4-Pro -- its sibling (V4-Flash)
# already failed this exact way once (5 steps in 28h from NVIDIA rate
# limits), so it's isolated in case it repeats that failure mode.
#
# Usage:
#   LANE=1 nohup setsid bash run_screening_lane.sh >> logs/screening_lane1.log 2>&1 &
#   LANE=2 nohup setsid bash run_screening_lane.sh >> logs/screening_lane2.log 2>&1 &
set -u
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
AUTOSCALER="$SCRIPT_DIR/k8s_autoscaler.py"
LOADGEN="$SCRIPT_DIR/load_generator.py"
OUTDIR="$SCRIPT_DIR/results_screening/cpu_bursty/rep1"
LOGDIR="$SCRIPT_DIR/logs"
KEYS_FILE="$SCRIPT_DIR/api_keys.conf"

STEPS=120
INTERVAL=60
MAX_REPLICAS=14
TRACE="traces/trace_cpu.npy"
VARIANTS="zero_shot domain"

LANE="${LANE:?set LANE=1 or LANE=2}"
case "$LANE" in
    1) APP_DEPLOYMENT="workload-cpu-7"
       RUNS="llama4-scout:groq qwen3.6-27b:groq glm-4.7:cerebras" ;;
    2) APP_DEPLOYMENT="workload-cpu-8"
       RUNS="nemotron3-super:nvidia deepseek-v4-pro:nvidia" ;;
    *) echo "LANE must be 1 or 2"; exit 1 ;;
esac

mkdir -p "$OUTDIR" "$LOGDIR"

declare -A PROVIDER_KEYS
while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# || -z "$key" ]] && continue
    PROVIDER_KEYS[$provider]="$key"
done < "$KEYS_FILE"

env_var_for() {
    case "$1" in
        groq) echo GROQ_API_KEY;; nvidia) echo NVIDIA_API_KEY;;
        cerebras) echo CEREBRAS_API_KEY;; sambanova) echo SAMBANOVA_API_KEY;;
    esac
}

is_complete() {
    [ -f "$1" ] && [ -s "$1" ] && [ "$(wc -l < "$1")" -ge $((STEPS + 1)) ]
}

SVC_IP=$(kubectl get svc "$APP_DEPLOYMENT" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
[ -z "$SVC_IP" ] && { echo "ERROR: no clusterIP for svc $APP_DEPLOYMENT"; exit 1; }
echo "$(date -u): [LANE $LANE] app $APP_DEPLOYMENT @ $SVC_IP"

TOTAL=0; COMPLETED=0
for spec in $RUNS; do for v in $VARIANTS; do TOTAL=$((TOTAL+1)); done; done

for spec in $RUNS; do
    model="${spec%%:*}"; provider="${spec##*:}"
    for variant in $VARIANTS; do
        outfile="$OUTDIR/k8s_cpu_${model}_${variant}.csv"
        tag="${model}_${variant}"
        if is_complete "$outfile"; then
            echo "$(date -u +%H:%M:%S) [LANE $LANE] SKIP $tag (complete)"
            COMPLETED=$((COMPLETED+1)); continue
        fi

        echo "$(date -u +%H:%M:%S) [LANE $LANE] START $tag ($provider) [$COMPLETED/$TOTAL]"
        kubectl scale deployment "$APP_DEPLOYMENT" --replicas=3 >/dev/null 2>&1
        sleep 15

        llog="$LOGDIR/screening_load_${model}_${variant}.log"
        alog="$LOGDIR/screening_auto_${model}_${variant}.log"

        $PYTHON "$LOADGEN" --service-ip "$SVC_IP" --trace "$SCRIPT_DIR/$TRACE" \
            --steps $STEPS --interval $INTERVAL --scale-factor 1.0 \
            --output "$OUTDIR/load_cpu_${model}_${variant}.csv" > "$llog" 2>&1 &
        LOAD_PID=$!

        ev=$(env_var_for "$provider")
        env "${ev}=${PROVIDER_KEYS[$provider]:-}" \
            $PYTHON "$AUTOSCALER" --workload cpu --model "$model" --variant "$variant" \
            --trace "$SCRIPT_DIR/$TRACE" --steps $STEPS --interval $INTERVAL \
            --max-replicas $MAX_REPLICAS --output-dir "$OUTDIR" --resume \
            --provider "$provider" --deployment-override "$APP_DEPLOYMENT" > "$alog" 2>&1 &
        AUTO_PID=$!

        wait $LOAD_PID $AUTO_PID 2>/dev/null
        if is_complete "$outfile"; then
            COMPLETED=$((COMPLETED+1)); echo "$(date -u +%H:%M:%S) [LANE $LANE] DONE $tag [$COMPLETED/$TOTAL]"
        else
            echo "$(date -u +%H:%M:%S) [LANE $LANE] FAILED $tag -- see $(basename "$alog")"
        fi
    done
done

echo "$(date -u): [LANE $LANE] SCREENING COMPLETE: $COMPLETED/$TOTAL"
