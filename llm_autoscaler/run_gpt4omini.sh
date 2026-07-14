#!/bin/bash
# Run all 4 variants of GPT-4o-mini on GitHub Models sequentially
set -e

export GITHUB_API_KEY=$(grep '^github:' /users/ffarazug/gym-sfu/llm_autoscaler/api_keys.conf | cut -d: -f2)

VARIANTS="zero_shot domain history_5 cot"
TRACE="/users/ffarazug/gym-sfu/llm_autoscaler/trace_alibaba_v2.npy"
OUTDIR="/users/ffarazug/gym-sfu/llm_autoscaler/results_long"
PYTHON="/users/ffarazug/gym-sfu/venv/bin/python"
SCRIPT="/users/ffarazug/gym-sfu/llm_autoscaler/llm_autoscaler.py"

for v in $VARIANTS; do
    echo "$(date): START gpt-4o-mini / $v"
    $PYTHON -u $SCRIPT \
        --model gpt-4o-mini --variant $v --provider github \
        --trace $TRACE --steps 1440 --cooldown 2.5 \
        --output-dir $OUTDIR --resume
    echo "$(date): DONE gpt-4o-mini / $v"
done

echo "$(date): ALL GPT-4o-mini variants complete"
