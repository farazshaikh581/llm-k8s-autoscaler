#!/bin/bash
# Real K8s autoscaling experiment — CPU + I/O workloads
# Runs load generator + autoscaler in parallel for each run
# Sends push notifications to Android via ntfy.sh

set -e
export PYTHONUNBUFFERED=1
# GROQ_API_KEY must be set in the environment before running this script

PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
DIR=/users/ffarazug/gym-sfu/llm_autoscaler
OUTDIR=$DIR/results_k8s
LOG=$DIR/logs/k8s_experiment.log
NTFY_TOPIC="llm-autoscaler-faraz-2026"

STEPS=60
INTERVAL=60
SCALE_FACTOR=1.0

MODELS="llama4-scout llama-8b"
VARIANTS="zero_shot domain"

mkdir -p "$OUTDIR" "$(dirname $LOG)"

notify() {
    curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1
    echo "$(date): [NOTIFY] $1" >> "$LOG"
}

run_one() {
    local workload=$1 model=$2 variant=$3
    local deployment="workload-${workload}"
    local trace="$DIR/traces/trace_${workload}.npy"
    local outfile="$OUTDIR/k8s_${workload}_${model}_${variant}.csv"
    local loadfile="$OUTDIR/load_${workload}_${model}_${variant}.csv"

    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        echo "$(date): SKIP $workload/$model/$variant (exists)" >> "$LOG"
        notify "⏭️ Skipped $workload/$model/$variant"
        return 0
    fi

    local SVC_IP=$(kubectl get svc "$deployment" -o jsonpath='{.spec.clusterIP}')

    notify "▶️ Starting $workload / $model / $variant"
    kubectl scale deployment "$deployment" --replicas=3
    sleep 15

    # Load generator in background
    $PYTHON $DIR/load_generator.py \
        --service-ip "$SVC_IP" \
        --trace "$trace" \
        --steps $STEPS --interval $INTERVAL \
        --scale-factor $SCALE_FACTOR \
        --output "$loadfile" >> "$LOG" 2>&1 &
    local LOAD_PID=$!

    # Autoscaler in background
    $PYTHON $DIR/k8s_autoscaler.py \
        --workload "$workload" --model "$model" --variant "$variant" \
        --trace "$trace" --steps $STEPS --interval $INTERVAL \
        --output-dir "$OUTDIR" >> "$LOG" 2>&1 &
    local AUTO_PID=$!

    wait $LOAD_PID $AUTO_PID 2>/dev/null
    local RET=$?

    if [ $RET -eq 0 ]; then
        notify "✅ $workload / $model / $variant DONE"
    else
        notify "❌ $workload / $model / $variant FAILED (exit=$RET)"
    fi
    return $RET
}

# ======================================================================
notify "🚀 K8s experiments STARTED — CPU + I/O workloads, $STEPS steps each"

total=0
done_count=0

# Count total runs
for workload in cpu io; do
    for baseline in hpa keda; do total=$((total+1)); done
    for model in $MODELS; do
        for variant in $VARIANTS; do total=$((total+1)); done
    done
done

for workload in cpu io; do
    notify "📦 === Starting $workload workload phase ==="

    # Baselines
    for baseline in hpa keda; do
        done_count=$((done_count+1))
        echo "[$done_count/$total] $workload / $baseline" >> "$LOG"
        run_one "$workload" "$baseline" "baseline"
    done

    # LLM models
    for model in $MODELS; do
        for variant in $VARIANTS; do
            done_count=$((done_count+1))
            echo "[$done_count/$total] $workload / $model / $variant" >> "$LOG"
            run_one "$workload" "$model" "$variant"
        done
    done
done

notify "🏁 ALL K8s EXPERIMENTS COMPLETE ($done_count/$total runs). Results in $OUTDIR/"
echo "$(date): ALL DONE" >> "$LOG"
ls -lh "$OUTDIR"/k8s_*.csv >> "$LOG" 2>&1
