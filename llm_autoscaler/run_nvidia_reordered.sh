#!/bin/bash
# Reordered NVIDIA worker for the long-sim v2 re-run.
# Reason: mistral-small4 and qwen3-80b are NVIDIA-exclusive (no other provider
# in api_keys.conf serves those exact builds), yet the original queue put them
# BEHIND the remaining llama-70b variants. llama-70b has Groq/SambaNova
# fallbacks, so it is the one safe to deprioritize. This worker runs the
# NVIDIA-exclusive core models FIRST, then finishes llama-70b last.
#
# Replaces worker 0 of run_long_v2.sh (killed). Groq/Cerebras workers untouched.
set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
SCRIPT=/users/ffarazug/gym-sfu/llm_autoscaler.py
TRACE="$SCRIPT_DIR/trace_alibaba_v2.npy"
STEPS=1440
COOLDOWN=2.5
OUTDIR="$SCRIPT_DIR/results_long_v2"
LOGDIR="$SCRIPT_DIR/logs"
NTFY_TOPIC="llm-autoscaler-faraz-2026"
KEYS_FILE="$SCRIPT_DIR/api_keys.conf"
LOG="$LOGDIR/v2_worker_0_nvidia_reordered.log"

# Reordered: NVIDIA-exclusive models first, llama-70b (has fallbacks) last.
# llama-70b:zero_shot already complete (1441 rows); omitted.
# llama-70b:domain is partial (~1240 rows) and resumes to completion.
RUN_LIST="mistral-small4:zero_shot mistral-small4:domain mistral-small4:history_5 mistral-small4:cot \
qwen3-80b:zero_shot qwen3-80b:domain qwen3-80b:history_5 qwen3-80b:cot \
llama-70b:domain llama-70b:history_5 llama-70b:cot"

notify() {
    curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
    echo "$(date): [NOTIFY] $1" >> "$LOG"
}

# Pull the nvidia key from api_keys.conf
NVIDIA_KEY=""
while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# ]] && continue
    [[ "$provider" == "nvidia" && -n "$key" ]] && NVIDIA_KEY="$key"
done < "$KEYS_FILE"

if [ -z "$NVIDIA_KEY" ]; then
    echo "ERROR: no nvidia key in $KEYS_FILE" >> "$LOG"
    exit 1
fi
export NVIDIA_API_KEY="$NVIDIA_KEY"

echo "$(date): Reordered NVIDIA worker starting — $RUN_LIST" >> "$LOG"
notify "v2 W0(nvidia): REORDERED queue — mistral, qwen first, llama-70b last"

for run in $RUN_LIST; do
    model="${run%%:*}"
    variant="${run##*:}"
    outfile="$OUTDIR/results_${model}_${variant}.csv"

    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        lines=$(wc -l < "$outfile")
        if [ "$lines" -ge 1400 ]; then
            echo "$(date): SKIP $model/$variant ($lines lines)" >> "$LOG"
            continue
        fi
    fi

    while true; do
        echo "$(date): Running $model/$variant" >> "$LOG"
        notify "v2 W0(nvidia): $model/$variant"

        $PYTHON "$SCRIPT" \
            --model "$model" --variant "$variant" --provider nvidia \
            --trace "$TRACE" --steps $STEPS --cooldown $COOLDOWN \
            --output-dir "$OUTDIR" --resume >> "$LOG" 2>&1 || true

        if [ -f "$outfile" ] && [ -s "$outfile" ]; then
            lines=$(wc -l < "$outfile")
            if [ "$lines" -ge 1400 ]; then
                notify "✅ v2 W0: $model/$variant DONE"
                break
            fi
            echo "$(date): $model/$variant at $lines rows, retry in 5min" >> "$LOG"
        else
            echo "$(date): $model/$variant no output, retry in 5min" >> "$LOG"
        fi
        sleep 300
    done
done

echo "$(date): Reordered NVIDIA worker FINISHED" >> "$LOG"
notify "🏁 v2 Worker 0 (nvidia, reordered) done"
