#!/usr/bin/env python3
"""Train DQN and PPO on the hardened simulator (autoscale_env_v3, issue #7).

Same protocol as the paper's original RL baselines: train on the first 1080
steps of the Alibaba trace, evaluate on the held-out tail and the full trace.
Evaluation CSVs use the simulator's column layout so plot_all.py picks them
up like any other controller run.

Usage:
  python train_rl_v3.py --timesteps 500000 --output-dir <dir>
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from autoscale_env_v3 import AutoscaleEnvV3
from llm_autoscaler import CSV_COLUMNS

SCRIPT_DIR = Path(__file__).parent
TRAIN_SPLIT = 1080

torch.set_num_threads(1)  # be a polite neighbor; MLP training is tiny


def train(algo_name: str, trace: np.ndarray, total_timesteps: int,
          models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(AutoscaleEnvV3(trace, seed=42))
    eval_env = Monitor(AutoscaleEnvV3(trace, seed=123))

    print(f"\n{'='*60}")
    print(f"Training {algo_name} on V3 env (hardened sim) — {total_timesteps} timesteps")
    print(f"Train trace: {len(trace)} steps (first {TRAIN_SPLIT} of alibaba_v2)")
    print(f"{'='*60}")

    t0 = time.time()

    if algo_name == "DQN":
        model = DQN(
            "MlpPolicy", env,
            learning_rate=5e-5,
            buffer_size=100_000,
            learning_starts=2000,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.4,
            exploration_final_eps=0.05,
            target_update_interval=1000,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
        )
    elif algo_name == "PPO":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / f"best_{algo_name}"),
        eval_freq=10000,
        n_eval_episodes=3,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    elapsed = time.time() - t0

    save_path = models_dir / f"{algo_name}_autoscaler_v3"
    model.save(str(save_path))
    print(f"Training done: {elapsed:.0f}s ({elapsed/60:.1f} min) -> {save_path}")
    return model, elapsed


def evaluate(model, algo_name: str, trace: np.ndarray, output_dir: Path,
             suffix: str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"results_{algo_name.lower()}_rl{suffix}.csv"

    env = AutoscaleEnvV3(trace, seed=999)
    obs, _ = env.reset(seed=999)

    def row(step, m, cluster, scale_event):
        return {
            "step": step,
            "replicas": cluster.target_replicas,
            "ready_replicas": cluster.ready_replicas,
            "latency_p90": m["latency_p90"],
            "cpu_pct": m["cpu_pct"],
            "requests": m["rps"],
            "success_rate": m["success_rate"],
            "vcpu_minutes": round(cluster.cumulative_vcpu_min, 2),
            "scale_event": scale_event,
            "llm_model": algo_name.lower(),
            "llm_variant": "rl",
            "llm_tokens_used": 0,
            "llm_latency_ms": 0,
        }

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row(0, env.last_metrics, env.cluster, 0))

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            writer.writerow(row(info["step"], env.last_metrics, env.cluster,
                                info["scale_event"]))

    print(f"  Saved: {out_file} ({env.t + 1} steps)")
    return out_file


def summarize(path: Path, label: str) -> dict:
    import pandas as pd
    df = pd.read_csv(path)
    sla_viol = (df["latency_p90"] > 200).sum()
    return {
        "label": label,
        "steps": len(df),
        "sla_pct": round((1 - sla_viol / len(df)) * 100, 1),
        "mean_lat": df["latency_p90"].mean(),
        "max_lat": df["latency_p90"].max(),
        "mean_rep": df["ready_replicas"].mean(),
        "cost": df["vcpu_minutes"].iloc[-1],
        "scales": df["scale_event"].sum(),
        "min_succ": df["success_rate"].min(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL on the hardened sim")
    parser.add_argument("--algo", choices=["DQN", "PPO", "both"], default="both",
                        help="train one algorithm (lets DQN and PPO run in parallel)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--trace", type=str,
                        default=str(SCRIPT_DIR / "traces" / "trace_alibaba_v2.npy"))
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "results" / "long_sim"))
    parser.add_argument("--models-dir", type=str,
                        default=str(SCRIPT_DIR / "models_v3"))
    args = parser.parse_args()

    full_trace = np.load(args.trace)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)

    train_trace = full_trace[:TRAIN_SPLIT]
    test_indist = full_trace[TRAIN_SPLIT:]

    print(f"Train trace:  {len(train_trace)} steps, RPS [{train_trace.min()}, {train_trace.max()}]")
    print(f"Held-out:     {len(test_indist)} steps, RPS [{test_indist.min()}, {test_indist.max()}]")

    algos = ["DQN", "PPO"] if args.algo == "both" else [args.algo]
    all_results = {}
    for algo in algos:
        model, train_time = train(algo, train_trace, args.timesteps, models_dir)
        print(f"\nEvaluating {algo}:")
        r_full = evaluate(model, algo, full_trace, output_dir)
        r_held = evaluate(model, algo, test_indist, output_dir, "_indist")
        all_results[algo] = {
            "train_time": train_time,
            "full": summarize(r_full, "Full alibaba trace"),
            "indist": summarize(r_held, "In-dist (held-out)"),
        }

    print(f"\n{'='*90}")
    print("RL BASELINES — HARDENED SIM (V3 env)")
    print(f"{'='*90}")
    for algo, data in all_results.items():
        print(f"  {algo} (trained in {data['train_time']:.0f}s)")
        print(f"  {'Test Set':<25} {'Steps':>6} {'SLA%':>6} {'AvgLat':>7} {'MaxLat':>7} {'AvgRep':>7} {'Cost':>8} {'Scales':>7} {'MinSucc':>8}")
        print(f"  {'-'*85}")
        for key in ["full", "indist"]:
            r = data[key]
            print(f"  {r['label']:<25} {r['steps']:>6} {r['sla_pct']:>5.1f}% "
                  f"{r['mean_lat']:>6.1f}ms {r['max_lat']:>6.0f}ms {r['mean_rep']:>6.1f} "
                  f"{r['cost']:>8.0f} {r['scales']:>7} {r['min_succ']:>8.3f}")
        print()


if __name__ == "__main__":
    main()
