#!/usr/bin/env python3
"""Sim-vs-real validation: replay both simulator performance models against
real-cluster observations (results/k8s_v2, CPU workload).

For every observed step (ready_replicas, rps_target) the pre-rework model
(utilization-curve, at cbd5264) and the hardened model (per-pod M/M/1/K,
llm_autoscaler.py) each predict latency P90 and CPU%. Predictions are compared
against what the real cluster measured at that same operating point.

Outputs:
  plots/19_sim_validation.png          three-panel figure (issue #6 closure gate)
  plots/sim_validation_summary.csv     error metrics per model version

The real data is the same source the hardened model's constants were calibrated
on (CPU demand slope + latency floor), so this is an in-sample consistency
check of the full queueing model, not an out-of-sample generalization test.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_autoscaler import POD, compute_metrics as compute_metrics_v2

ROOT = Path(__file__).parent
K8S_DIR = ROOT / "results" / "k8s_v2"
PLOTS = ROOT / "plots"
SLA_MS = 200.0

# ---------------------------------------------------------------------------
# Pre-rework performance model, verbatim from llm_autoscaler.py at cbd5264
# (the last commit before the queueing rework), jitter removed so the replay
# is deterministic.
# ---------------------------------------------------------------------------

V1_SERVICE_TIME_MS = 8.0
V1_CAPACITY_PER_REPLICA = 200


def compute_metrics_v1(ready_replicas: int, rps: int) -> dict:
    c = max(ready_replicas, 1)
    total_capacity = c * V1_CAPACITY_PER_REPLICA
    rho = min(rps / max(total_capacity, 1), 2.0)

    cpu_pct = round(min(rho * 100.0, 100.0), 1)

    if rho <= 1.0:
        queue_factor = (rho / (1.0 - rho + 0.1)) ** 1.5
        latency_p90 = V1_SERVICE_TIME_MS * (1.0 + queue_factor)
    else:
        lat_at_capacity = V1_SERVICE_TIME_MS * (1.0 + (1.0 / 0.1) ** 1.5)
        latency_p90 = lat_at_capacity * math.exp(3.0 * (rho - 1.0))
    latency_p90 = round(max(V1_SERVICE_TIME_MS, min(latency_p90, 10000.0)), 1)

    if rho <= 1.0:
        success = 1.0
    elif rho <= 1.3:
        success = 1.0 - 0.5 * (rho - 1.0)
    else:
        success = max(0.3, 0.85 - 0.5 * (rho - 1.3))

    return {"cpu_pct": cpu_pct, "latency_p90": latency_p90,
            "success_rate": round(success, 4)}

# ---------------------------------------------------------------------------
# Real observations
# ---------------------------------------------------------------------------

def load_real() -> pd.DataFrame:
    frames = []
    for f in sorted(K8S_DIR.glob("k8s_cpu_*.csv")):
        if f.stat().st_size < 100:
            continue
        df = pd.read_csv(f)
        df["run"] = f.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # drop warmup and malformed rows; keep transitions (the plant model takes
    # the pods that are actually ready, so scale transients are valid points)
    df = df[(df.step >= 3) & (df.ready_replicas >= 1) & (df.rps_target > 0)]
    df = df.dropna(subset=["latency_p90_ms", "cpu_millicores"])
    # observed CPU, HPA-style: total millicores against summed pod requests
    df["cpu_pct_obs"] = df.cpu_millicores / (df.ready_replicas * POD["cpu_request_m"]) * 100.0
    df["load_per_pod"] = df.rps_target / df.ready_replicas
    # cpu_millicores <= 5 while serving load is a metrics-server read failure
    # (those rows show hundreds of req/min/pod at normal latency); exclude them
    # from the CPU comparison only, keep them for latency/SLA
    df["cpu_valid"] = df.cpu_millicores > 5
    return df.reset_index(drop=True)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)  # hardened model adds 3% jitter; fix it for reproducibility
    v1 = [compute_metrics_v1(int(r), int(q)) for r, q in zip(df.ready_replicas, df.rps_target)]
    v2 = [compute_metrics_v2(int(r), int(q)) for r, q in zip(df.ready_replicas, df.rps_target)]
    df["lat_v1"] = [m["latency_p90"] for m in v1]
    df["lat_v2"] = [m["latency_p90"] for m in v2]
    df["cpu_v1"] = [m["cpu_pct"] for m in v1]
    df["cpu_v2"] = [m["cpu_pct"] for m in v2]
    df["succ_v1"] = [m["success_rate"] for m in v1]
    df["succ_v2"] = [m["success_rate"] for m in v2]
    return df

# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def error_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ver in ["v1", "v2"]:
        for metric, pred_col, obs_col in [
            ("latency_p90_ms", f"lat_{ver}", "latency_p90_ms"),
            ("cpu_pct", f"cpu_{ver}", "cpu_pct_obs"),
            ("success_rate", f"succ_{ver}", "success_rate"),
        ]:
            d = df[df.cpu_valid] if metric == "cpu_pct" else df
            err = d[pred_col] - d[obs_col]
            row = {
                "model": "pre-rework" if ver == "v1" else "hardened",
                "metric": metric,
                "mae": err.abs().mean(),
                "median_ae": err.abs().median(),
                "bias": err.mean(),
            }
            if metric == "latency_p90_ms":
                row["sla_agreement"] = (
                    (df[pred_col] <= SLA_MS) == (df[obs_col] <= SLA_MS)
                ).mean()
            rows.append(row)
    return pd.DataFrame(rows).round(3)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

C_OBS, C_V1, C_V2 = "#37474F", "#FF9800", "#2196F3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 100,
})


def binned(df: pd.DataFrame, col: str, nbins: int = 18):
    """Median and IQR of `col` in quantile bins of per-pod load."""
    bins = pd.qcut(df.load_per_pod, nbins, duplicates="drop")
    g = df.groupby(bins, observed=True)
    x = g.load_per_pod.median()
    return x, g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)


def make_figure(df: pd.DataFrame, errors: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) latency vs per-pod load
    ax = axes[0]
    x, med, q1, q3 = binned(df, "latency_p90_ms")
    ax.fill_between(x, q1, q3, color=C_OBS, alpha=0.18, linewidth=0)
    ax.plot(x, med, color=C_OBS, linewidth=2, label="Real cluster (median, IQR)")
    for col, c, ls, lbl in [("lat_v1", C_V1, "--", "Pre-rework sim"),
                            ("lat_v2", C_V2, "-", "Hardened sim")]:
        xs, ms, _, _ = binned(df, col)
        ax.plot(xs, ms, color=c, linestyle=ls, linewidth=2, label=lbl)
    ax.axhline(SLA_MS, color="red", linestyle=":", alpha=0.6, linewidth=1.5, label="SLA (200 ms)")
    ax.set_yscale("log")
    ax.set_xlabel("Load per pod (req/min)")
    ax.set_ylabel("Latency P90 (ms)")
    ax.set_title("(a) Latency vs load")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()

    # (b) CPU% vs per-pod load (rows with failed CPU reads excluded)
    ax = axes[1]
    dcpu = df[df.cpu_valid]
    x, med, q1, q3 = binned(dcpu, "cpu_pct_obs")
    ax.fill_between(x, q1, q3, color=C_OBS, alpha=0.18, linewidth=0)
    ax.plot(x, med, color=C_OBS, linewidth=2, label="Real cluster (median, IQR)")
    for col, c, ls, lbl in [("cpu_v1", C_V1, "--", "Pre-rework sim"),
                            ("cpu_v2", C_V2, "-", "Hardened sim")]:
        xs, ms, _, _ = binned(dcpu, col)
        ax.plot(xs, ms, color=c, linestyle=ls, linewidth=2, label=lbl)
    ax.axhline(100, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, label="CPU request")
    ax.set_xlabel("Load per pod (req/min)")
    ax.set_ylabel("CPU (% of pod request)")
    ax.set_title("(b) CPU utilization vs load")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()

    # (c) predicted vs observed latency
    ax = axes[2]
    lims = [5, 12000]
    ax.plot(lims, lims, color="gray", linewidth=1, linestyle="-", alpha=0.7)
    ax.scatter(df.latency_p90_ms, df.lat_v1, s=7, color=C_V1, alpha=0.25,
               edgecolors="none", label="Pre-rework sim")
    ax.scatter(df.latency_p90_ms, df.lat_v2, s=7, color=C_V2, alpha=0.25,
               edgecolors="none", label="Hardened sim")
    for _, r in errors[errors.metric == "latency_p90_ms"].iterrows():
        y = 0.14 if r.model == "pre-rework" else 0.06
        c = C_V1 if r.model == "pre-rework" else C_V2
        ax.text(0.97, y, f"{r.model}: MAE {r.mae:.0f} ms", transform=ax.transAxes,
                ha="right", fontsize=9, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed latency P90 (ms)")
    ax.set_ylabel("Predicted latency P90 (ms)")
    ax.set_title("(c) Predicted vs observed (y = x ideal)")
    ax.grid(alpha=0.25, linewidth=0.5)
    leg = ax.legend(loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.suptitle("Simulator validation against the real cluster "
                 f"({len(df)} steps, CPU workload, results/k8s_v2)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure: {out}")


def main():
    PLOTS.mkdir(exist_ok=True)
    df = load_real()
    print(f"Loaded {len(df)} real observations "
          f"({df.run.nunique()} runs, per-pod load {df.load_per_pod.min():.0f}"
          f"-{df.load_per_pod.max():.0f} req/min)")
    df = predict(df)
    errors = error_rows(df)

    lossy = (df.success_rate < 0.999).sum()
    bad_cpu = (~df.cpu_valid).sum()
    print(f"\nNotes: only {lossy} real rows show losses (success-rate "
          "comparison dominated by the no-loss regime); "
          f"{bad_cpu} rows with failed CPU reads (<=5 mc under load) excluded "
          "from the CPU comparison only.\n")
    print(errors.to_string(index=False))

    out_csv = PLOTS / "sim_validation_summary.csv"
    errors.to_csv(out_csv, index=False)
    print(f"\nSummary: {out_csv}")
    make_figure(df, errors, PLOTS / "19_sim_validation.png")


if __name__ == "__main__":
    main()
