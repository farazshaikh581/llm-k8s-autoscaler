#!/usr/bin/env bash
# Multi-seed RL baselines on the hardened simulator (issue #7 follow-up).
# 5 seeds x {DQN, PPO}, two algo lanes in parallel. Per-seed CSVs go to
# results/long_sim_rl_seeds/seed_N/; aggregate_rl_seeds.py folds them into
# the paper's results/long_sim/ with mean +/- std.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="0 1 2 3 4"
TIMESTEPS=500000
OUT=results/long_sim_rl_seeds
mkdir -p logs "$OUT"

run_lane() {
  local algo=$1
  for s in $SEEDS; do
    echo "[$(date -u +%H:%M:%S)] $algo seed $s START"
    python3 train_rl_v3.py --algo "$algo" --seed "$s" --timesteps "$TIMESTEPS" \
      --output-dir "$OUT/seed_$s" --models-dir "models_v3_seeds/seed_$s" \
      >> "logs/rl_multiseed_${algo}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] $algo seed $s DONE"
  done
}

run_lane DQN &
DQN_PID=$!
run_lane PPO &
PPO_PID=$!
wait $DQN_PID $PPO_PID
echo "[$(date -u +%H:%M:%S)] ALL SEEDS DONE"
