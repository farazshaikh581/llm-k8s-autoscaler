#!/usr/bin/env python3
"""Business-case analysis for REAL K8s cluster runs (not the simulator).

Mirrors plot_business_case.py's cost/SLA/stability methodology, but reads
real-cluster CSVs (results_k8s_v2/, results_richer/) instead of results_long_v2.
Real runs don't have a fixed 24h horizon or a vcpu_minutes column, so cost is
computed directly from (replicas x 0.25 vCPU request x actual elapsed wall time
between steps), which also absorbs the real-cluster step/wall-clock desync
documented in README (llama-70b latency spikes stretch some steps far past the
nominal 60s interval).

Usage:
  python plot_business_case_real.py --results-dir results_k8s_v2 [--workload cpu]
  python plot_business_case_real.py --results-dir results_richer [--workload cpu]
"""
import argparse
import glob
import os

import pandas as pd

VCPU_REQUEST = 0.25       # cpu: 250m request per replica (k8s/deployment.yaml, k8s/workloads.yaml)
VCPU_HOUR_USD = 0.04       # on-demand $/vCPU-hour (same basis as the sim analysis)
HOURS_PER_YEAR = 8760.0
SLA_MS = 200.0             # P90 latency SLA threshold (ms) — same as sim
SLA_DEPLOYABLE = 99.0
MIN_STEPS = 120            # only complete runs

BASELINE_MODELS = {"hpa": "HPA", "keda": "KEDA", "rl-dqn": "DQN", "rl-ppo": "PPO"}
CORE_LLM_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b", "llama4-scout"]
DISPLAY = {"llama-8b": "Llama-8B", "llama-70b": "Llama-70B", "mistral-small4": "Mistral",
           "qwen3-80b": "Qwen-80B", "llama4-scout": "Scout"}


def parse_name(path):
    """k8s_{workload}_{model}_{variant}.csv -> (workload, model, variant)."""
    base = os.path.basename(path)[len("k8s_"):-4]
    workload, rest = base.split("_", 1)
    for m in list(BASELINE_MODELS) + CORE_LLM_MODELS + ["deepseek-v4-flash"]:
        if rest == f"{m}_baseline":
            return workload, m, "baseline"
        if rest.startswith(m + "_"):
            return workload, m, rest[len(m) + 1:]
    model, _, variant = rest.partition("_")
    return workload, model, variant


def load_run(path):
    if os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    if len(df) < MIN_STEPS or "latency_p90_ms" not in df:
        return None
    ts = pd.to_datetime(df["timestamp"])
    dt_hours = ts.diff().dt.total_seconds().fillna(0) / 3600.0
    # cost of a step = replicas held *during* that step x request x elapsed hours
    # since the row's replicas describe the state entering the step, use replicas
    # shifted by one (state held since the previous timestamp) — approximate with
    # replicas.shift(1), first row gets 0 (no prior interval).
    vcpu_hours = (df["replicas"].shift(1).fillna(df["replicas"].iloc[0]) * VCPU_REQUEST * dt_hours).sum()
    run_hours = dt_hours.sum()
    lat_sla = (df["latency_p90_ms"] <= SLA_MS).mean() * 100
    scales = int(df["scale_event"].sum())
    return dict(vcpu_hours=vcpu_hours, run_hours=run_hours, lat_sla=lat_sla,
                mean_lat=df["latency_p90_ms"].mean(), scales=scales, n_steps=len(df))


def annualize(vcpu_hours, run_hours):
    if run_hours <= 0:
        return 0.0
    return (vcpu_hours * VCPU_HOUR_USD) * (HOURS_PER_YEAR / run_hours)


def load_summary(results_dir, workload_filter):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "k8s_*.csv"), recursive=True)):
        workload, model, variant = parse_name(f)
        if workload_filter and workload != workload_filter:
            continue
        r = load_run(f)
        if r is None:
            continue
        cls = BASELINE_MODELS.get(model, "LLM")
        label = BASELINE_MODELS.get(model, f"{model}_{variant}")
        rows.append(dict(
            path=f, workload=workload, model=model, variant=variant, label=label, cls=cls,
            lat_sla=round(r["lat_sla"], 1), mean_lat=round(r["mean_lat"], 1),
            cost_vcpu_h=round(r["vcpu_hours"], 2), cost_usd=round(annualize(r["vcpu_hours"], r["run_hours"])),
            scales=r["scales"], run_hours=round(r["run_hours"], 2), n_steps=r["n_steps"],
        ))
    return pd.DataFrame(rows)


def flat_label(row):
    return row.label if row.cls != "LLM" else f"{DISPLAY.get(row.model, row.model)}/{row.variant}"


def two_regime_verdict(s):
    lines = ["\n" + "=" * 78, "REAL-CLUSTER BUSINESS CASE", "=" * 78]
    if "hpa" not in set(s.model):
        lines.append("(no complete HPA baseline in this data — skipping HPA-relative framing)")
        print("\n".join(lines))
        return
    hpa = s[s.model == "hpa"].iloc[0]

    a_pool = s[s.lat_sla >= 99.5].sort_values("cost_usd")
    lines.append("\nA) PERFORMANCE-CRITICAL  (minimize cost s.t. SLA >= 99.5%)")
    if not a_pool.empty:
        w = a_pool.iloc[0]
        pct = (hpa.cost_usd - w.cost_usd) / hpa.cost_usd * 100 if hpa.cost_usd else 0
        lines.append(f"   -> WINNER: {flat_label(w)}  (SLA {w.lat_sla}%, ${w.cost_usd:,}/yr, "
                      f"{abs(pct):.0f}% {'cheaper than' if pct >= 0 else 'pricier than'} HPA, {w.scales} scales)")
    else:
        lines.append("   -> no config clears 99.5% SLA on real cluster yet")
    for _, r in a_pool.head(6).iterrows():
        lines.append(f"     {flat_label(r):24s} SLA {r.lat_sla:5.1f}%  ${r.cost_usd:>6,}/yr  {r.scales:>4} scales")

    b_pool = s[(s.lat_sla >= SLA_DEPLOYABLE) & (s.cost_usd <= hpa.cost_usd)].sort_values("scales")
    lines.append(f"\nB) COST-&-STABILITY-SENSITIVE  (min scales s.t. SLA >= 99% and cost <= HPA ${hpa.cost_usd:,}/yr)")
    if not b_pool.empty:
        w = b_pool.iloc[0]
        lines.append(f"   -> WINNER: {flat_label(w)}  (SLA {w.lat_sla}%, ${w.cost_usd:,}/yr, {w.scales} scales)")
    else:
        lines.append("   -> no config both clears 99% SLA and undercuts HPA cost yet")
    for _, r in b_pool.head(6).iterrows():
        lines.append(f"     {flat_label(r):24s} SLA {r.lat_sla:5.1f}%  ${r.cost_usd:>6,}/yr  {r.scales:>4} scales")
    lines.append(f"     {'HPA (reference)':24s} SLA {hpa.lat_sla:5.1f}%  ${hpa.cost_usd:>6,}/yr  {hpa.scales:>4} scales")
    lines.append("=" * 78)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_k8s_v2")
    ap.add_argument("--workload", default="cpu", help="cpu, io, or '' for all")
    ap.add_argument("--out", default=None, help="CSV output path (default: <results-dir>_business_case.csv)")
    args = ap.parse_args()

    s = load_summary(args.results_dir, args.workload or None)
    print(f"Loaded {len(s)} complete real-cluster runs (>= {MIN_STEPS} steps) "
          f"from {args.results_dir}/ (workload={args.workload or 'all'})")
    if s.empty:
        print("No complete runs found.")
        return
    out = args.out or f"{args.results_dir.rstrip('/')}_business_case.csv"
    s.sort_values("cost_usd").to_csv(out, index=False)
    print(f"Summary table -> {out}")
    two_regime_verdict(s.sort_values("cost_usd"))


if __name__ == "__main__":
    main()
