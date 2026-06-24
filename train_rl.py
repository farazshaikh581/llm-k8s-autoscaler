#!/usr/bin/env python3
"""Train DQN and PPO autoscalers with proper train/test split, then evaluate.

Train/test methodology:
  - TRAIN on trace_alibaba_v2.npy first 1080 steps (75%, 18 hours)
  - TEST in-distribution: trace_alibaba_v2.npy last 360 steps (unseen)
  - TEST out-of-distribution: trace_cpu.npy, trace_io.npy (different workloads)
  - TEST real-world: K8s cluster (handled by k8s_autoscaler.py --model rl-dqn/ppo)

Usage:
  python train_rl.py --timesteps 500000
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from autoscale_env import AutoscaleEnv

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_long"
MODELS_DIR = SCRIPT_DIR / "models"

TRAIN_SPLIT = 1080  # first 18 hours for training

CSV_COLUMNS = [
    "step", "replicas", "ready_replicas", "latency_p90", "cpu_pct",
    "requests", "success_rate", "vcpu_minutes", "scale_event",
    "llm_model", "llm_variant", "llm_tokens_used", "llm_latency_ms",
]


def train(algo_name: str, trace: np.ndarray, total_timesteps: int):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(AutoscaleEnv(trace))
    eval_env = Monitor(AutoscaleEnv(trace))

    print(f"\n{'='*60}")
    print(f"Training {algo_name} — {total_timesteps} timesteps")
    print(f"Train trace: {len(trace)} steps (first {TRAIN_SPLIT} of alibaba_v2)")
    print(f"{'='*60}")

    t0 = time.time()

    if algo_name == "DQN":
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=1,
        )
    elif algo_name == "PPO":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / f"best_{algo_name}"),
        eval_freq=5000,
        n_eval_episodes=3,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    elapsed = time.time() - t0

    save_path = MODELS_DIR / f"{algo_name}_autoscaler"
    model.save(str(save_path))
    print(f"\nTraining done: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Model saved: {save_path}")

    return model, elapsed


def evaluate(model, algo_name: str, trace: np.ndarray, output_dir: Path,
             trace_name: str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{trace_name}" if trace_name else ""
    out_file = output_dir / f"results_{algo_name.lower()}_rl{suffix}.csv"

    env = AutoscaleEnv(trace)
    obs, _ = env.reset()
    prev_replicas = 3

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for step in range(len(trace)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            scale_event = 1 if info["replicas"] != prev_replicas else 0
            prev_replicas = info["replicas"]

            writer.writerow({
                "step": step,
                "replicas": info["replicas"],
                "ready_replicas": info["replicas"],
                "latency_p90": round(info["latency_p90"], 1),
                "cpu_pct": round(info["cpu_pct"], 1),
                "requests": info["rps"],
                "success_rate": round(info["success_rate"], 4),
                "vcpu_minutes": round(info["vcpu_minutes"], 2),
                "scale_event": scale_event,
                "llm_model": algo_name.lower(),
                "llm_variant": "rl",
                "llm_tokens_used": 0,
                "llm_latency_ms": 0,
            })

            if done:
                break

    print(f"  Saved: {out_file} ({step + 1} steps)")
    return out_file


def summarize(path: Path, label: str):
    import pandas as pd
    df = pd.read_csv(path)
    sla_viol = (df["latency_p90"] > 200).sum()
    sla_pct = round((1 - sla_viol / len(df)) * 100, 1)
    return {
        "label": label,
        "steps": len(df),
        "sla_pct": sla_pct,
        "mean_lat": df["latency_p90"].mean(),
        "max_lat": df["latency_p90"].max(),
        "mean_rep": df["ready_replicas"].mean(),
        "cost": df["vcpu_minutes"].iloc[-1],
        "scales": df["scale_event"].sum(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL autoscaler baselines")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--trace", type=str,
                        default=str(SCRIPT_DIR / "trace_alibaba_v2.npy"))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    full_trace = np.load(args.trace)
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Train/test split
    train_trace = full_trace[:TRAIN_SPLIT]
    test_indist = full_trace[TRAIN_SPLIT:]

    # Out-of-distribution traces
    traces_dir = SCRIPT_DIR / "traces"
    test_cpu = np.load(traces_dir / "trace_cpu.npy")
    test_io = np.load(traces_dir / "trace_io.npy")

    print(f"Train trace:      {len(train_trace)} steps, RPS [{train_trace.min()}, {train_trace.max()}]")
    print(f"Test in-dist:     {len(test_indist)} steps, RPS [{test_indist.min()}, {test_indist.max()}]")
    print(f"Test OOD (cpu):   {len(test_cpu)} steps, RPS [{test_cpu.min()}, {test_cpu.max()}]")
    print(f"Test OOD (io):    {len(test_io)} steps, RPS [{test_io.min()}, {test_io.max()}]")

    all_results = {}

    for algo in ["DQN", "PPO"]:
        model, train_time = train(algo, train_trace, args.timesteps)

        print(f"\nEvaluating {algo}:")

        # 1. In-distribution test (unseen portion of same trace)
        r1 = evaluate(model, algo, test_indist, output_dir, "indist")
        # 2. Full alibaba trace (for comparison with LLM long-sim results)
        r2 = evaluate(model, algo, full_trace, output_dir)
        # 3. CPU workload trace (out-of-distribution)
        r3 = evaluate(model, algo, test_cpu, output_dir, "cpu")
        # 4. I/O workload trace (out-of-distribution)
        r4 = evaluate(model, algo, test_io, output_dir, "io")

        all_results[algo] = {
            "train_time": train_time,
            "indist": summarize(r1, "In-dist (held-out)"),
            "full": summarize(r2, "Full alibaba trace"),
            "cpu": summarize(r3, "OOD: CPU workload"),
            "io": summarize(r4, "OOD: I/O workload"),
        }

    print(f"\n{'='*80}")
    print("RL BASELINE RESULTS — TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Training: first {TRAIN_SPLIT} steps of trace_alibaba_v2.npy")
    print()

    for algo, data in all_results.items():
        print(f"  {algo} (trained in {data['train_time']:.0f}s)")
        print(f"  {'Test Set':<25} {'Steps':>6} {'SLA%':>6} {'AvgLat':>7} {'MaxLat':>7} {'AvgRep':>7} {'Cost':>8} {'Scales':>7}")
        print(f"  {'-'*75}")
        for key in ["indist", "full", "cpu", "io"]:
            r = data[key]
            print(f"  {r['label']:<25} {r['steps']:>6} {r['sla_pct']:>5.1f}% {r['mean_lat']:>6.1f}ms {r['max_lat']:>6.0f}ms {r['mean_rep']:>6.1f} {r['cost']:>8.0f} {r['scales']:>7}")
        print()

    print("KEY INSIGHT: Compare SLA% across test sets.")
    print("If SLA degrades from in-dist → OOD → real K8s, it proves that")
    print("RL requires environment-specific training while LLMs generalize zero-shot.")


if __name__ == "__main__":
    main()
