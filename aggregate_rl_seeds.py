#!/usr/bin/env python3
"""Aggregate the multi-seed RL baselines (run_rl_multiseed.sh) into the paper.

For each algorithm we have 5 seeds in results/long_sim_rl_seeds/seed_*/.
This script:
  1. Summarizes each seed's full-trace run (SLA, latency, cost, scales, ...).
  2. Reports mean +/- std per algorithm -> plots/rl_seed_variance.csv.
  3. Picks the *medoid* seed (closest to the per-algo mean, z-scored over the
     headline metrics) and copies its full + _indist CSVs into
     results/long_sim/ so the time-series plots stay a single real trajectory
     while the tables carry the mean +/- std.

Usage: python aggregate_rl_seeds.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SEEDS_DIR = BASE / "results" / "long_sim_rl_seeds"
PAPER_DIR = BASE / "results" / "long_sim"
PLOTS_DIR = BASE / "plots"
ALGOS = ["dqn", "ppo"]
SEEDS = [0, 1, 2, 3, 4]
# metrics used to pick the representative (medoid) seed
MEDOID_METRICS = ["sla_pct", "cost", "scales", "mean_lat"]


def summarize(path: Path) -> dict:
    d = pd.read_csv(path)
    return {
        "sla_pct": (1 - (d["latency_p90"] > 200).mean()) * 100,
        "mean_lat": d["latency_p90"].mean(),
        "cost": d["vcpu_minutes"].iloc[-1],
        "scales": int(d["scale_event"].sum()),
        "min_succ": d["success_rate"].min(),
        "mean_rep": d["ready_replicas"].mean(),
    }


def main():
    variance_rows = []
    for algo in ALGOS:
        rows = []
        for s in SEEDS:
            r = summarize(SEEDS_DIR / f"seed_{s}" / f"results_{algo}_rl.csv")
            r["seed"] = s
            rows.append(r)
        df = pd.DataFrame(rows).set_index("seed")

        # medoid: z-score the headline metrics, nearest seed to the mean vector
        z = (df[MEDOID_METRICS] - df[MEDOID_METRICS].mean()) / df[MEDOID_METRICS].std(ddof=0)
        dist = np.sqrt((z ** 2).sum(axis=1))
        medoid = int(dist.idxmin())

        print(f"\n=== {algo.upper()} (n={len(SEEDS)} seeds) ===")
        print(df.round(2).to_string())
        print(f"medoid seed = {medoid} (nearest to mean over {MEDOID_METRICS})")

        # copy medoid full + indist into the paper dir
        for suffix in ["", "_indist"]:
            src = SEEDS_DIR / f"seed_{medoid}" / f"results_{algo}_rl{suffix}.csv"
            dst = PAPER_DIR / f"results_{algo}_rl{suffix}.csv"
            shutil.copyfile(src, dst)
            print(f"  copied {src.relative_to(BASE)} -> {dst.relative_to(BASE)}")

        row = {"algo": algo.upper(), "n_seeds": len(SEEDS), "medoid_seed": medoid}
        for m in ["sla_pct", "mean_lat", "cost", "scales", "min_succ", "mean_rep"]:
            row[f"{m}_mean"] = round(df[m].mean(), 3)
            row[f"{m}_std"] = round(df[m].std(ddof=1), 3)
        variance_rows.append(row)

    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / "rl_seed_variance.csv"
    pd.DataFrame(variance_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(pd.DataFrame(variance_rows).to_string(index=False))


if __name__ == "__main__":
    main()
