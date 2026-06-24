#!/bin/bash
# Long simulation: 1440 steps — multi-provider parallel execution
# Config: api_keys.conf (provider:key, one per line)
# Models auto-assigned to providers; bonus models get reduced variants.

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
SCRIPT="$SCRIPT_DIR/llm_autoscaler.py"
TRACE="$SCRIPT_DIR/trace_alibaba_v2.npy"
STEPS=1440
COOLDOWN=2.5
OUTDIR="$SCRIPT_DIR/results_long"
LOGDIR="$SCRIPT_DIR/logs"
NTFY_TOPIC="YOUR_NTFY_TOPIC"
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
SAMBANOVA_KEYS=()
CEREBRAS_KEYS=()
NVIDIA_KEYS=()
GOOGLE_KEYS=()

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
        sambanova) SAMBANOVA_KEYS+=("$key") ;;
        cerebras)  CEREBRAS_KEYS+=("$key") ;;
        nvidia)    NVIDIA_KEYS+=("$key") ;;
        google)    GOOGLE_KEYS+=("$key") ;;
        *) echo "Unknown provider: $provider" ;;
    esac
done < "$KEYS_FILE"

NUM_GROQ=${#GROQ_KEYS[@]}
NUM_SAMBA=${#SAMBANOVA_KEYS[@]}
NUM_CEREBRAS=${#CEREBRAS_KEYS[@]}
NUM_NVIDIA=${#NVIDIA_KEYS[@]}
NUM_GOOGLE=${#GOOGLE_KEYS[@]}
TOTAL_KEYS=$((NUM_GROQ + NUM_SAMBA + NUM_CEREBRAS + NUM_NVIDIA + NUM_GOOGLE))

echo "$(date): Keys — Groq:$NUM_GROQ  SambaNova:$NUM_SAMBA  Cerebras:$NUM_CEREBRAS  NVIDIA:$NUM_NVIDIA  Google:$NUM_GOOGLE"

if [ $TOTAL_KEYS -eq 0 ]; then
    echo "ERROR: No API keys in $KEYS_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Baselines (instant, no API)
# ---------------------------------------------------------------------------
$PYTHON "$SCRIPT" --baselines-only --trace "$TRACE" --steps $STEPS --output-dir "$OUTDIR" 2>&1
notify "Baselines done (HPA + KEDA)"

# ---------------------------------------------------------------------------
# Build run lists: "model:variant" pairs per provider
# ---------------------------------------------------------------------------
VARIANTS="zero_shot domain history_5 cot"

GROQ_RUNS=""
SAMBA_RUNS=""
CEREBRAS_RUNS=""
NVIDIA_RUNS=""
GOOGLE_RUNS=""

# Groq-only models (all 4 variants)
for model in llama4-scout; do
    for v in $VARIANTS; do
        GROQ_RUNS="$GROQ_RUNS $model:$v"
    done
done

# llama-8b + gemma4-31b: NVIDIA if available, else Groq
if [ $NUM_NVIDIA -gt 0 ]; then
    for v in $VARIANTS; do NVIDIA_RUNS="$NVIDIA_RUNS llama-8b:$v"; done
    for v in $VARIANTS; do NVIDIA_RUNS="$NVIDIA_RUNS mistral-small4:$v"; done
else
    for v in $VARIANTS; do GROQ_RUNS="$GROQ_RUNS llama-8b:$v"; done
fi

# llama-70b: SambaNova if available, else Groq
if [ $NUM_SAMBA -gt 0 ]; then
    for v in $VARIANTS; do SAMBA_RUNS="$SAMBA_RUNS llama-70b:$v"; done
else
    for v in $VARIANTS; do GROQ_RUNS="$GROQ_RUNS llama-70b:$v"; done
fi

# gpt-oss-120b: Cerebras > SambaNova > Groq
if [ $NUM_CEREBRAS -gt 0 ]; then
    for v in $VARIANTS; do CEREBRAS_RUNS="$CEREBRAS_RUNS gpt-oss-120b:$v"; done
elif [ $NUM_SAMBA -gt 0 ]; then
    for v in $VARIANTS; do SAMBA_RUNS="$SAMBA_RUNS gpt-oss-120b:$v"; done
else
    for v in $VARIANTS; do GROQ_RUNS="$GROQ_RUNS gpt-oss-120b:$v"; done
fi

# qwen3-80b: NVIDIA (all 4 variants)
if [ $NUM_NVIDIA -gt 0 ]; then
    for v in $VARIANTS; do NVIDIA_RUNS="$NVIDIA_RUNS qwen3-80b:$v"; done
fi

# Bonus: DeepSeek-V3 on SambaNova (only zero_shot + domain)
if [ $NUM_SAMBA -gt 0 ]; then
    SAMBA_RUNS="$SAMBA_RUNS deepseek-v3:zero_shot deepseek-v3:domain"
fi

# Bonus: Gemini 2.5 Flash on Google (only zero_shot + domain)
if [ $NUM_GOOGLE -gt 0 ]; then
    GOOGLE_RUNS="gemini-flash:zero_shot gemini-flash:domain"
fi

# Trim leading spaces
GROQ_RUNS=$(echo $GROQ_RUNS | xargs)
SAMBA_RUNS=$(echo $SAMBA_RUNS | xargs)
CEREBRAS_RUNS=$(echo $CEREBRAS_RUNS | xargs)
NVIDIA_RUNS=$(echo $NVIDIA_RUNS | xargs)
GOOGLE_RUNS=$(echo $GOOGLE_RUNS | xargs)

echo "Groq runs:      $GROQ_RUNS"
echo "SambaNova runs:  $SAMBA_RUNS"
echo "Cerebras runs:   $CEREBRAS_RUNS"
echo "NVIDIA runs:     $NVIDIA_RUNS"
echo "Google runs:     $GOOGLE_RUNS"

# ---------------------------------------------------------------------------
# Worker: processes "model:variant" pairs with retry+resume
# ---------------------------------------------------------------------------
# Usage: run_worker WID PROVIDER ENV_VAR API_KEY "run1 run2 ..."
run_worker() {
    local wid=$1
    local provider=$2
    local env_var=$3
    local api_key=$4
    local run_list=$5
    local log="$LOGDIR/worker_${wid}_${provider}.log"

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
            notify "W$wid($provider): $model/$variant"

            $PYTHON "$SCRIPT" \
                --model "$model" --variant "$variant" --provider "$provider" \
                --trace "$TRACE" --steps $STEPS --cooldown $COOLDOWN \
                --output-dir "$OUTDIR" --resume >> "$log" 2>&1 || true

            if [ -f "$outfile" ] && [ -s "$outfile" ]; then
                local lines
                lines=$(wc -l < "$outfile")
                if [ "$lines" -ge 1400 ]; then
                    notify "✅ W$wid: $model/$variant DONE"
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
    notify "🏁 Worker $wid ($provider) done"
}

# ---------------------------------------------------------------------------
# Distribute runs across keys and launch parallel workers
# ---------------------------------------------------------------------------
PIDS=()
WID=0

# Groq workers: split GROQ_RUNS round-robin across Groq keys
if [ $NUM_GROQ -gt 0 ] && [ -n "$GROQ_RUNS" ]; then
    GROQ_RUN_ARR=($GROQ_RUNS)
    for i in $(seq 0 $((NUM_GROQ - 1))); do
        WORKER_RUNS=""
        for j in "${!GROQ_RUN_ARR[@]}"; do
            if [ $((j % NUM_GROQ)) -eq $i ]; then
                WORKER_RUNS="$WORKER_RUNS ${GROQ_RUN_ARR[$j]}"
            fi
        done
        WORKER_RUNS=$(echo $WORKER_RUNS | xargs)
        if [ -n "$WORKER_RUNS" ]; then
            echo "Worker $WID (groq#$i): $WORKER_RUNS"
            run_worker $WID groq GROQ_API_KEY "${GROQ_KEYS[$i]}" "$WORKER_RUNS" &
            PIDS+=($!)
            WID=$((WID + 1))
        fi
    done
fi

# SambaNova workers
if [ $NUM_SAMBA -gt 0 ] && [ -n "$SAMBA_RUNS" ]; then
    SAMBA_RUN_ARR=($SAMBA_RUNS)
    for i in $(seq 0 $((NUM_SAMBA - 1))); do
        WORKER_RUNS=""
        for j in "${!SAMBA_RUN_ARR[@]}"; do
            if [ $((j % NUM_SAMBA)) -eq $i ]; then
                WORKER_RUNS="$WORKER_RUNS ${SAMBA_RUN_ARR[$j]}"
            fi
        done
        WORKER_RUNS=$(echo $WORKER_RUNS | xargs)
        if [ -n "$WORKER_RUNS" ]; then
            echo "Worker $WID (sambanova#$i): $WORKER_RUNS"
            run_worker $WID sambanova SAMBANOVA_API_KEY "${SAMBANOVA_KEYS[$i]}" "$WORKER_RUNS" &
            PIDS+=($!)
            WID=$((WID + 1))
        fi
    done
fi

# Cerebras workers
if [ $NUM_CEREBRAS -gt 0 ] && [ -n "$CEREBRAS_RUNS" ]; then
    CEREBRAS_RUN_ARR=($CEREBRAS_RUNS)
    for i in $(seq 0 $((NUM_CEREBRAS - 1))); do
        WORKER_RUNS=""
        for j in "${!CEREBRAS_RUN_ARR[@]}"; do
            if [ $((j % NUM_CEREBRAS)) -eq $i ]; then
                WORKER_RUNS="$WORKER_RUNS ${CEREBRAS_RUN_ARR[$j]}"
            fi
        done
        WORKER_RUNS=$(echo $WORKER_RUNS | xargs)
        if [ -n "$WORKER_RUNS" ]; then
            echo "Worker $WID (cerebras#$i): $WORKER_RUNS"
            run_worker $WID cerebras CEREBRAS_API_KEY "${CEREBRAS_KEYS[$i]}" "$WORKER_RUNS" &
            PIDS+=($!)
            WID=$((WID + 1))
        fi
    done
fi

# NVIDIA workers
if [ $NUM_NVIDIA -gt 0 ] && [ -n "$NVIDIA_RUNS" ]; then
    NVIDIA_RUN_ARR=($NVIDIA_RUNS)
    for i in $(seq 0 $((NUM_NVIDIA - 1))); do
        WORKER_RUNS=""
        for j in "${!NVIDIA_RUN_ARR[@]}"; do
            if [ $((j % NUM_NVIDIA)) -eq $i ]; then
                WORKER_RUNS="$WORKER_RUNS ${NVIDIA_RUN_ARR[$j]}"
            fi
        done
        WORKER_RUNS=$(echo $WORKER_RUNS | xargs)
        if [ -n "$WORKER_RUNS" ]; then
            echo "Worker $WID (nvidia#$i): $WORKER_RUNS"
            run_worker $WID nvidia NVIDIA_API_KEY "${NVIDIA_KEYS[$i]}" "$WORKER_RUNS" &
            PIDS+=($!)
            WID=$((WID + 1))
        fi
    done
fi

# Google workers
if [ $NUM_GOOGLE -gt 0 ] && [ -n "$GOOGLE_RUNS" ]; then
    GOOGLE_RUN_ARR=($GOOGLE_RUNS)
    for i in $(seq 0 $((NUM_GOOGLE - 1))); do
        WORKER_RUNS=""
        for j in "${!GOOGLE_RUN_ARR[@]}"; do
            if [ $((j % NUM_GOOGLE)) -eq $i ]; then
                WORKER_RUNS="$WORKER_RUNS ${GOOGLE_RUN_ARR[$j]}"
            fi
        done
        WORKER_RUNS=$(echo $WORKER_RUNS | xargs)
        if [ -n "$WORKER_RUNS" ]; then
            echo "Worker $WID (google#$i): $WORKER_RUNS"
            run_worker $WID google GOOGLE_API_KEY "${GOOGLE_KEYS[$i]}" "$WORKER_RUNS" &
            PIDS+=($!)
            WID=$((WID + 1))
        fi
    done
fi

notify "🔬 Launched $WID workers across $TOTAL_KEYS keys (5 providers)"
echo "$(date): Waiting for $WID workers..."

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

notify "🏁 ALL simulations COMPLETE"
echo "$(date): ALL DONE"
ls -lh "$OUTDIR"/results_*.csv
