#!/usr/bin/env python3
"""A 'perfect' (oracle) autoscaling baseline derived from REAL cluster capacity.

No simulator, no queueing model. We measure, from the real per-step logs, the
sustainable per-replica request rate at the SLA boundary, then compute the
minimum-cost replica trajectory that serves a given trace within SLA -- with
perfect foresight (it pre-scales for the next step's load, covering the ~60s
pod start-up). This is the honest upper bound: the least capacity any scaler
would need to hold ~100% SLA on this workload.

Capacity model:
  c = q-th percentile of (rps / ready_replicas) over all SLA-met steps
      (SLA-met = latency_p90 < 200 ms and success_rate >= 0.99).
  A replica sustains ~c req/s at SLA; replicas(L) = ceil(L / c), clipped [1,20].

Usage:
  # validate/illustrate on existing real data + a trace:
  python compute_oracle_baseline.py --results-dir results_k8s_v2 \
      --pattern 'k8s_cpu_*.csv' --trace traces/trace_cpu.npy --steps 120
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

REPLICA_MIN, REPLICA_MAX = 1, 20
SLA_MS = 200


def pooled(results_dir, pattern):
    frames = []
    for f in glob.glob(os.path.join(results_dir, pattern)):
        if os.path.getsize(f) < 100:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) > 10 and {"rps_target", "ready_replicas",
                              "latency_p90_ms", "success_rate"} <= set(df.columns):
            frames.append(df)
    if not frames:
        raise SystemExit(f"no usable CSVs in {results_dir}/{pattern}")
    return pd.concat(frames, ignore_index=True), len(frames)


def per_replica_capacity(d, q):
    m = (d.latency_p90_ms < SLA_MS) & (d.success_rate >= 0.99) & (d.ready_replicas >= 1)
    s = d[m]
    rate = s.rps_target / s.ready_replicas
    return float(np.quantile(rate, q)), int(m.sum()), len(d)


def oracle_trace(trace, c):
    """Min-cost replicas with 1-step lookahead (pre-scale for next step)."""
    n = len(trace)
    reps = np.empty(n, dtype=int)
    for t in range(n):
        load = max(trace[t], trace[t + 1] if t + 1 < n else trace[t])
        reps[t] = int(np.clip(np.ceil(load / c), REPLICA_MIN, REPLICA_MAX))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_k8s_v2")
    ap.add_argument("--pattern", default="k8s_cpu_*.csv")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--q", type=float, default=0.90,
                    help="per-replica capacity percentile; LOWER = more conservative "
                         "(assumes less per replica -> provisions more). 0.90 nominal.")
    ap.add_argument("--max-replicas", type=int, default=20,
                    help="schedulable ceiling (match the run's --max-replicas)")
    ap.add_argument("--out", default=None, help="write oracle per-step CSV here")
    args = ap.parse_args()

    global REPLICA_MAX
    REPLICA_MAX = args.max_replicas

    d, nruns = pooled(args.results_dir, args.pattern)
    c, nsla, ntot = per_replica_capacity(d, args.q)
    trace = np.load(args.trace)[:args.steps].astype(float)

    reps = oracle_trace(trace, c)
    # sanity: fraction of real SLA-met points the capacity model would have covered
    m = (d.latency_p90_ms < SLA_MS) & (d.success_rate >= 0.99) & (d.ready_replicas >= 1)
    covered = (d[m].ready_replicas >= np.ceil(d[m].rps_target / c)).mean()

    print(f"Real data: {nruns} runs, {ntot} steps ({nsla} SLA-met)")
    print(f"Per-replica capacity c (q={args.q}): {c:.0f} rps/replica")
    print(f"Capacity model covers {100*covered:.0f}% of real SLA-met steps")
    print(f"\nOracle on {os.path.basename(args.trace)} ({len(trace)} steps, "
          f"RPS [{trace.min():.0f},{trace.max():.0f}]):")
    print(f"  replicas: mean {reps.mean():.1f}, peak {reps.max()}, min {reps.min()}")
    print(f"  scale events: {int((np.diff(reps) != 0).sum())}")
    print(f"  cost (replica-steps): {int(reps.sum())}  |  SLA by construction: ~100%")

    if args.out:
        pd.DataFrame({
            "step": range(len(trace)), "rps_target": trace.astype(int),
            "ready_replicas": reps, "replicas": reps,
            "latency_p90_ms": 0, "success_rate": 1.0,
            "llm_model": "oracle", "llm_variant": "perfect",
            "scale_event": np.concatenate([[0], (np.diff(reps) != 0).astype(int)]),
        }).to_csv(args.out, index=False)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
