#!/bin/bash
# LLM autoscaler experiments v2 — proper infrastructure model
# Skip gpt-oss-120b (daily token quota exhausted), run 4 remaining models
# 4 models × 4 variants = 16 runs, 150 steps each
# Estimated: ~3 hours at 2.5s cooldown

set -e
export PYTHONUNBUFFERED=1
# GROQ_API_KEY must be set in the environment before running this script

PYTHON=/users/ffarazug/gym-sfu/venv/bin/python
SCRIPT=/users/ffarazug/gym-sfu/llm_autoscaler/llm_autoscaler.py
TRACE=/users/ffarazug/gym-sfu/llm_autoscaler/trace_alibaba_v2.npy
STEPS=150
COOLDOWN=2.5
OUTDIR=/users/ffarazug/gym-sfu/llm_autoscaler/results
LOG=/users/ffarazug/gym-sfu/llm_autoscaler/logs/run_all_v2.log

mkdir -p "$OUTDIR" "$(dirname $LOG)"
echo "$(date): Starting v2 experiments (4 models, $STEPS steps)" >> "$LOG"

# Baselines first
$PYTHON $SCRIPT --baselines-only --trace $TRACE --steps $STEPS --output-dir $OUTDIR >> "$LOG" 2>&1

# Skip gpt-oss-120b (daily quota hit). Run tomorrow with:
#   run_all_gptoss.sh or manual --model gpt-oss-120b --all-variants
MODELS="llama-70b llama4-scout llama-8b qwen3-32b"
VARIANTS="zero_shot history_5 cot domain"

total=16
done=0

for model in $MODELS; do
    for variant in $VARIANTS; do
        done=$((done + 1))
        outfile="$OUTDIR/results_${model}_${variant}.csv"
        if [ -f "$outfile" ] && [ -s "$outfile" ]; then
            echo "[$done/$total] SKIP: $model/$variant (exists)" >> "$LOG"
            continue
        fi
        rm -f "$outfile"
        echo "[$done/$total] $(date): Starting $model / $variant" >> "$LOG"
        $PYTHON $SCRIPT --model $model --variant $variant \
            --trace $TRACE --steps $STEPS --cooldown $COOLDOWN \
            --output-dir $OUTDIR >> "$LOG" 2>&1
        echo "[$done/$total] $(date): Finished $model / $variant" >> "$LOG"
    done
done

echo "" >> "$LOG"
echo "$(date): ALL DONE (4 models). Run gpt-oss-120b tomorrow." >> "$LOG"
ls -lh $OUTDIR/*.csv >> "$LOG" 2>&1
