#!/bin/bash
# Real K8s autoscaling experiment v2 — multi-provider, parallel workloads
#
# Runs CPU and I/O workload experiments in parallel (independent deployments).
# Each workload track runs models sequentially (can't share a deployment).
# Multi-provider: reads api_keys.conf, assigns models to available providers.
# Supports resume: skips runs that already have enough steps.
#
# Usage:
#   nohup bash run_k8s_v2.sh >> logs/k8s_v2.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
AUTOSCALER="$SCRIPT_DIR/k8s_autoscaler.py"
LOADGEN="$SCRIPT_DIR/load_generator.py"
OUTDIR="$SCRIPT_DIR/results_k8s_v2"
LOGDIR="$SCRIPT_DIR/logs"
KEYS_FILE="$SCRIPT_DIR/api_keys.conf"
NTFY_TOPIC="YOUR_NTFY_TOPIC"

STEPS=120
INTERVAL=60
SCALE_FACTOR=1.0

mkdir -p "$OUTDIR" "$LOGDIR"

COMPLETED_RUNS=0

notify() {
    curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
    echo "$(date): [NOTIFY] $1"
}

# Human-friendly model names
friendly_name() {
    case "$1" in
        llama4-scout)     echo "Llama 4 Scout 17B" ;;
        llama-8b)         echo "Llama 3.1 8B" ;;
        llama-70b)        echo "Llama 3.3 70B" ;;
        mistral-small4)   echo "Mistral Small 4" ;;
        gpt-oss-120b)     echo "GPT-OSS 120B" ;;
        deepseek-v4-flash) echo "DeepSeek V4 Flash" ;;
        qwen3-80b)        echo "Qwen 3 80B" ;;
        hpa)              echo "HPA Baseline" ;;
        keda)             echo "KEDA Baseline" ;;
        rl-dqn)           echo "RL (DQN)" ;;
        rl-ppo)           echo "RL (PPO)" ;;
        *)                echo "$1" ;;
    esac
}

friendly_workload() {
    case "$1" in
        cpu) echo "CPU-intensive" ;;
        io)  echo "I/O-bound" ;;
        *)   echo "$1" ;;
    esac
}

friendly_variant() {
    case "$1" in
        zero_shot) echo "zero-shot" ;;
        history_5) echo "5-step history" ;;
        cot)       echo "chain-of-thought" ;;
        domain)    echo "domain prompt" ;;
        baseline)  echo "" ;;
        *)         echo "$1" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse api_keys.conf
# ---------------------------------------------------------------------------
declare -A PROVIDER_KEYS

while IFS=: read -r provider key; do
    provider=$(echo "$provider" | xargs 2>/dev/null || echo "$provider")
    key=$(echo "$key" | xargs 2>/dev/null || echo "$key")
    [[ -z "$provider" || "$provider" =~ ^# ]] && continue
    [[ -z "$key" ]] && continue
    PROVIDER_KEYS[$provider]="$key"
done < "$KEYS_FILE"

echo "$(date): Available providers: ${!PROVIDER_KEYS[*]}"

# ---------------------------------------------------------------------------
# Model → provider assignment (first available wins)
# ---------------------------------------------------------------------------
# Format: "model:provider:env_var"
RUNS=()

assign() {
    local model=$1; shift
    for provider in "$@"; do
        if [ -n "${PROVIDER_KEYS[$provider]}" ]; then
            RUNS+=("$model:$provider")
            echo "  $model → $provider"
            return 0
        fi
    done
    echo "  WARNING: No provider for $model (tried: $*)"
    return 1
}

echo "$(date): Assigning models to providers..."
# Scout excluded — Groq-only, heavy rate limits. Scout has K8s v1 real-cluster data instead.
assign llama-8b        nvidia groq
assign llama-70b       nvidia groq sambanova
assign mistral-small4  nvidia
# deepseek-v4-flash removed — NVIDIA rate limits made it infeasible (5 steps in 28h)
# assign deepseek-v4-flash nvidia
# gpt-oss-120b removed — all sim variants failed (40-61% SLA, 8-13× over latency threshold)
# assign gpt-oss-120b   cerebras nvidia groq
assign qwen3-80b       nvidia

echo "$(date): ${#RUNS[@]} LLM models assigned"

VARIANTS="zero_shot domain history_5 cot"

get_variants() {
    echo "$VARIANTS"
}

# ---------------------------------------------------------------------------
# Check if a run is already complete
# ---------------------------------------------------------------------------
is_complete() {
    local outfile=$1
    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        local lines
        lines=$(wc -l < "$outfile")
        # header + STEPS data rows
        if [ "$lines" -ge $((STEPS + 1)) ]; then
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Run one experiment: load generator + autoscaler in parallel
# ---------------------------------------------------------------------------
run_one() {
    local workload=$1 model=$2 variant=$3 provider=$4
    local deployment="workload-${workload}"
    local trace="$SCRIPT_DIR/traces/trace_${workload}.npy"
    local outfile="$OUTDIR/k8s_${workload}_${model}_${variant}.csv"
    local loadfile="$OUTDIR/load_${workload}_${model}_${variant}.csv"

    if is_complete "$outfile"; then
        echo "$(date): SKIP $workload/$model/$variant (complete)"
        return 0
    fi

    # Determine env var for this provider
    local env_var
    case "$provider" in
        groq)      env_var="GROQ_API_KEY" ;;
        nvidia)    env_var="NVIDIA_API_KEY" ;;
        cerebras)  env_var="CEREBRAS_API_KEY" ;;
        sambanova) env_var="SAMBANOVA_API_KEY" ;;
        google)    env_var="GOOGLE_API_KEY" ;;
    esac

    local SVC_IP
    SVC_IP=$(kubectl get svc "$deployment" -o jsonpath='{.spec.clusterIP}')

    local fname vname wname run_label
    fname=$(friendly_name "$model")
    vname=$(friendly_variant "$variant")
    wname=$(friendly_workload "$workload")
    if [ -n "$vname" ]; then
        run_label="$fname ($vname) on $wname workload"
    else
        run_label="$fname on $wname workload"
    fi

    echo "$(date): START $workload / $model / $variant ($provider)"
    notify "Starting: $run_label [$COMPLETED_RUNS/$total done]"

    # Reset deployment
    kubectl scale deployment "$deployment" --replicas=3
    sleep 15

    # Load generator in background
    $PYTHON "$LOADGEN" \
        --service-ip "$SVC_IP" \
        --trace "$trace" \
        --steps $STEPS --interval $INTERVAL \
        --scale-factor $SCALE_FACTOR \
        --output "$loadfile" > "$LOGDIR/load_${workload}_${model}_${variant}.log" 2>&1 &
    local LOAD_PID=$!

    # Autoscaler (with provider env var set)
    local extra_args=""
    if [ "$model" != "hpa" ] && [ "$model" != "keda" ] && \
       [ "$model" != "rl-dqn" ] && [ "$model" != "rl-ppo" ]; then
        extra_args="--provider $provider"
    fi

    env "$env_var=${PROVIDER_KEYS[$provider]}" \
        $PYTHON "$AUTOSCALER" \
        --workload "$workload" --model "$model" --variant "$variant" \
        --trace "$trace" --steps $STEPS --interval $INTERVAL \
        --output-dir "$OUTDIR" --resume \
        $extra_args > "$LOGDIR/auto_${workload}_${model}_${variant}.log" 2>&1 &
    local AUTO_PID=$!

    wait $LOAD_PID $AUTO_PID 2>/dev/null
    local RET=$?

    if [ $RET -eq 0 ] && is_complete "$outfile"; then
        COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        notify "Done: $run_label — $COMPLETED_RUNS/$total complete"
    else
        notify "FAILED: $run_label — check logs/auto_${workload}_${model}_${variant}.log"
    fi
    return $RET
}

# ---------------------------------------------------------------------------
# Run one workload track (sequential: baselines → RL → LLMs)
# ---------------------------------------------------------------------------
run_workload() {
    local workload=$1
    local track_log="$LOGDIR/track_${workload}.log"

    local wname
    wname=$(friendly_workload "$workload")
    echo "$(date): === TRACK $workload STARTED ===" | tee -a "$track_log"
    notify "Starting $wname workload track (baselines → RL → LLMs)"

    # 1. Baselines (no API needed)
    for baseline in hpa keda; do
        run_one "$workload" "$baseline" "baseline" "none" 2>&1 | tee -a "$track_log" || true
    done

    # 2. RL baselines — wait for V2 models if training is still running
    local dqn_model="$SCRIPT_DIR/models_v2/DQN_autoscaler_v2.zip"
    local ppo_model="$SCRIPT_DIR/models_v2/PPO_autoscaler_v2.zip"
    if [ ! -f "$dqn_model" ] || [ ! -f "$ppo_model" ]; then
        echo "$(date): Waiting for RL V2 models to finish training..."
        notify "RL models still training — K8s experiment paused until DQN+PPO finish"
        while [ ! -f "$dqn_model" ] || [ ! -f "$ppo_model" ]; do
            sleep 60
        done
        echo "$(date): RL V2 models ready"
        notify "RL models ready — resuming K8s experiments with DQN and PPO"
    fi
    for rl in rl-dqn rl-ppo; do
        run_one "$workload" "$rl" "baseline" "none" 2>&1 | tee -a "$track_log" || true
    done

    # 3. LLM models
    for run_spec in "${RUNS[@]}"; do
        local model="${run_spec%%:*}"
        local provider="${run_spec##*:}"
        local variants
        variants=$(get_variants "$model")

        for variant in $variants; do
            run_one "$workload" "$model" "$variant" "$provider" 2>&1 | tee -a "$track_log" || true
        done
    done

    echo "$(date): === TRACK $workload DONE ===" | tee -a "$track_log"
    notify "All $wname workload experiments finished ($COMPLETED_RUNS/$total total)"
}

# ---------------------------------------------------------------------------
# Main: run CPU and I/O tracks in parallel
# ---------------------------------------------------------------------------

# Count total runs
total=0
for workload in cpu io; do
    total=$((total + 4))  # hpa, keda, rl-dqn, rl-ppo
    for run_spec in "${RUNS[@]}"; do
        model="${run_spec%%:*}"
        variants=$(get_variants "$model")
        for v in $variants; do
            total=$((total + 1))
        done
    done
done

echo "$(date): Total runs planned: $total ($STEPS steps × ${INTERVAL}s each)"
echo "$(date): Estimated time: ~$((total * STEPS * INTERVAL / 3600 / 2)) hours (2 parallel tracks)"
notify "K8s experiment started — $total runs across CPU + I/O workloads. Each run is 2 hours (120 steps x 60s). ETA ~20 hours."

# Launch both workload tracks in parallel
run_workload "cpu" &
CPU_PID=$!

run_workload "io" &
IO_PID=$!

wait $CPU_PID
wait $IO_PID

notify "ALL EXPERIMENTS COMPLETE — $total runs finished. Results in results_k8s_v2/"
echo "$(date): ALL DONE"
echo "Results in: $OUTDIR/"
ls -lh "$OUTDIR"/k8s_*.csv 2>/dev/null
