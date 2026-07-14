#!/bin/bash
# Long simulation v2: 1440 steps on the HARDENED simulator (sim-fidelity branch).
# Differences from run_long.sh:
#   - SCRIPT points at the reworked root llm_autoscaler.py (per-pod M/M/1/K,
#     requests/limits, node scheduling caps)
#   - Fresh OUTDIR results_long_v2/ (pre-rework results_long/ preserved)
#   - Paper lineup only: 6 models x 4 variants + HPA/KEDA baselines.
#     No deepseek-v3 (dead on SambaNova), no gemini-flash (too slow).
#   - llama-70b routed to NVIDIA (SambaNova route was unreliable)

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

mkdir -p "$OUTDIR" "$LOGDIR"

notify() {
    curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
    echo "$(date): [NOTIFY] $1"
}

# ---------------------------------------------------------------------------
# Parse api_keys.conf
# ---------------------------------------------------------------------------
GROQ_KEYS=()
CEREBRAS_KEYS=()
NVIDIA_KEYS=()

if [ ! -f "$KEYS_FILE" ]; then
    echo "ERROR: $KEYS_FILE not found"
    exit 1
fi

while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# ]] && continue
    [[ -z "$key" ]] && continue
    case "$provider" in
        groq)      GROQ_KEYS+=("$key") ;;
        cerebras)  CEREBRAS_KEYS+=("$key") ;;
        nvidia)    NVIDIA_KEYS+=("$key") ;;
    esac
done < "$KEYS_FILE"

NUM_GROQ=${#GROQ_KEYS[@]}
NUM_CEREBRAS=${#CEREBRAS_KEYS[@]}
NUM_NVIDIA=${#NVIDIA_KEYS[@]}

echo "$(date): Keys — Groq:$NUM_GROQ  Cerebras:$NUM_CEREBRAS  NVIDIA:$NUM_NVIDIA"

if [ $NUM_NVIDIA -eq 0 ] || [ $NUM_GROQ -eq 0 ] || [ $NUM_CEREBRAS -eq 0 ]; then
    echo "ERROR: need at least one key each for nvidia, groq, cerebras"
    exit 1
fi

# ---------------------------------------------------------------------------
# Baselines (instant, no API)
# ---------------------------------------------------------------------------
$PYTHON "$SCRIPT" --baselines-only --trace "$TRACE" --steps $STEPS --output-dir "$OUTDIR" 2>&1
notify "v2: Baselines done (HPA + KEDA)"

# ---------------------------------------------------------------------------
# Run lists (paper lineup)
# ---------------------------------------------------------------------------
VARIANTS="zero_shot domain history_5 cot"

NVIDIA_RUNS=""
for model in llama-8b llama-70b mistral-small4 qwen3-80b; do
    for v in $VARIANTS; do NVIDIA_RUNS="$NVIDIA_RUNS $model:$v"; done
done

GROQ_RUNS=""
for v in $VARIANTS; do GROQ_RUNS="$GROQ_RUNS llama4-scout:$v"; done

CEREBRAS_RUNS=""
for v in $VARIANTS; do CEREBRAS_RUNS="$CEREBRAS_RUNS gpt-oss-120b:$v"; done

NVIDIA_RUNS=$(echo $NVIDIA_RUNS | xargs)
GROQ_RUNS=$(echo $GROQ_RUNS | xargs)
CEREBRAS_RUNS=$(echo $CEREBRAS_RUNS | xargs)

echo "NVIDIA runs:   $NVIDIA_RUNS"
echo "Groq runs:     $GROQ_RUNS"
echo "Cerebras runs: $CEREBRAS_RUNS"

# ---------------------------------------------------------------------------
# Worker: processes "model:variant" pairs with retry+resume
# ---------------------------------------------------------------------------
run_worker() {
    local wid=$1
    local provider=$2
    local env_var=$3
    local api_key=$4
    local run_list=$5
    local log="$LOGDIR/v2_worker_${wid}_${provider}.log"

    echo "$(date): Worker $wid ($provider) starting — $run_list" > "$log"
    export "$env_var=$api_key"

    for run in $run_list; do
        local model="${run%%:*}"
        local variant="${run##*:}"
        local outfile="$OUTDIR/results_${model}_${variant}.csv"

        if [ -f "$outfile" ] && [ -s "$outfile" ]; then
            local lines
            lines=$(wc -l < "$outfile")
            if [ "$lines" -ge 1400 ]; then
                echo "$(date): SKIP $model/$variant ($lines lines)" >> "$log"
                continue
            fi
        fi

        while true; do
            echo "$(date): Running $model/$variant" >> "$log"
            notify "v2 W$wid($provider): $model/$variant"

            $PYTHON "$SCRIPT" \
                --model "$model" --variant "$variant" --provider "$provider" \
                --trace "$TRACE" --steps $STEPS --cooldown $COOLDOWN \
                --output-dir "$OUTDIR" --resume >> "$log" 2>&1 || true

            if [ -f "$outfile" ] && [ -s "$outfile" ]; then
                local lines
                lines=$(wc -l < "$outfile")
                if [ "$lines" -ge 1400 ]; then
                    notify "✅ v2 W$wid: $model/$variant DONE"
                    break
                fi
                echo "$(date): $model/$variant at $lines rows, retry in 5min" >> "$log"
            else
                echo "$(date): $model/$variant no output, retry in 5min" >> "$log"
            fi
            sleep 300
        done
    done

    echo "$(date): Worker $wid ($provider) FINISHED" >> "$log"
    notify "🏁 v2 Worker $wid ($provider) done"
}

# ---------------------------------------------------------------------------
# Launch one worker per provider
# ---------------------------------------------------------------------------
PIDS=()

echo "Worker 0 (nvidia): $NVIDIA_RUNS"
run_worker 0 nvidia NVIDIA_API_KEY "${NVIDIA_KEYS[0]}" "$NVIDIA_RUNS" &
PIDS+=($!)

echo "Worker 1 (groq): $GROQ_RUNS"
run_worker 1 groq GROQ_API_KEY "${GROQ_KEYS[0]}" "$GROQ_RUNS" &
PIDS+=($!)

echo "Worker 2 (cerebras): $CEREBRAS_RUNS"
run_worker 2 cerebras CEREBRAS_API_KEY "${CEREBRAS_KEYS[0]}" "$CEREBRAS_RUNS" &
PIDS+=($!)

notify "🔬 v2 long-sim launched: 3 workers, 24 LLM runs + baselines (hardened sim)"
echo "$(date): Waiting for 3 workers..."

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

notify "🏁 v2 long-sim ALL COMPLETE"
echo "$(date): ALL DONE"
ls -lh "$OUTDIR"/results_*.csv
