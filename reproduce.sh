#!/usr/bin/env bash
# Regenerate all figures/tables/summaries from the committed result CSVs.
#   ./reproduce.sh
# (Re-running the experiments themselves needs LLM API keys + a k8s cluster; see below.)
set -euo pipefail
cd "$(dirname "$0")"

echo "== deps =="
python3 -c "import pandas, numpy, matplotlib" 2>/dev/null || pip install -r requirements.txt

echo "== regenerate plots 1-16 + LaTeX tables + CSV summaries from results/ =="
python3 plot_all.py

echo "== regenerate LLM inference-time figures 16-17 + table =="
python3 plot_inference_times.py

echo
echo "done -> plots/  (18 figures, LaTeX tables, CSV summaries)"
echo
echo "To re-run the experiments (needs provider keys + a Kubernetes cluster):"
echo "  cp api_keys.conf.example api_keys.conf      # add NVIDIA/Groq/Cerebras keys"
echo "  scripts/run_long.sh                          # 1440-step simulation"
echo "  scripts/run_k8s_v2.sh                        # real-cluster experiment (k8s)"
