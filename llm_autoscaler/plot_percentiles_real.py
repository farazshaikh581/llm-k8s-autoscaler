#!/usr/bin/env python3
"""Latency percentile analysis for REAL K8s cluster runs (P50 / P90 / P99).

load_generator.py records latency_mean_ms, latency_p50_ms, latency_p90_ms and
latency_p99_ms per step (in load_{workload}_{model}_{variant}.csv) -- richer
than the single P90 the business-case scripts use. This computes, per run, the
mean of each per-step percentile across the run (a run-level summary of "how
big is P50/P90/P99 typically"), SLA attainment at each percentile, and the
P99/P50 ratio (tail amplification -- how much worse the tail is than the
median for that controller).

NOTE: P95 is not available. load_generator.py only ever computed and persisted
P50/P90/P99 per step (see compute_stats()); the raw per-request latencies used
to derive them are discarded after each step, so P95 cannot be recovered
retroactively from completed runs. It can be added to load_generator.py for
future runs (see bottom of this file's usage note).

Usage:
  python plot_percentiles_real.py --results-dir results_k8s_v2 --workload cpu
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLA_MS = 200.0
MIN_STEPS = 120
OUT_DIR = "business_case_real"

BASELINE_MODELS = {"hpa": "HPA", "keda": "KEDA", "rl-dqn": "DQN", "rl-ppo": "PPO"}
CORE_LLM_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b", "llama4-scout", "deepseek-v4-flash", "gpt-oss-120b"]
DISPLAY = {"llama-8b": "Llama-8B", "llama-70b": "Llama-70B", "mistral-small4": "Mistral",
           "qwen3-80b": "Qwen-80B", "llama4-scout": "Scout", "deepseek-v4-flash": "DeepSeek",
           "gpt-oss-120b": "GPT-OSS-120B"}

# Sequential single-hue ramp (ColorBrewer Blues), light -> dark, one step per
# percentile magnitude (P50 lightest / smallest -> P99 darkest / largest).
P_COLORS = {"p50": "#9ECAE1", "p90": "#4292C6", "p99": "#08306B"}
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"
SOURCE_TAG = "REAL K8S CLUSTER — measured, not simulated"
TAG_COLOR = "#CC3311"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "figure.dpi": 130,
})


def parse_name(path, prefix):
    base = os.path.basename(path)[len(prefix):-4]
    workload, rest = base.split("_", 1)
    for m in list(BASELINE_MODELS) + CORE_LLM_MODELS:
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
    need = {"latency_p50_ms", "latency_p90_ms", "latency_p99_ms"}
    if len(df) < MIN_STEPS or not need.issubset(df.columns):
        return None
    out = {}
    for p in ("p50", "p90", "p99"):
        col = df[f"latency_{p}_ms"]
        out[f"mean_{p}"] = round(col.mean(), 1)
        out[f"sla_{p}"] = round((col <= SLA_MS).mean() * 100, 1)
    out["mean_mean"] = round(df["latency_mean_ms"].mean(), 1) if "latency_mean_ms" in df else None
    out["tail_ratio"] = round(out["mean_p99"] / out["mean_p50"], 2) if out["mean_p50"] > 0 else float("nan")
    out["n_steps"] = len(df)
    return out


def load_summary(results_dir, workload_filter):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "load_*.csv"), recursive=True)):
        workload, model, variant = parse_name(f, "load_")
        if workload_filter and workload != workload_filter:
            continue
        r = load_run(f)
        if r is None:
            continue
        cls = BASELINE_MODELS.get(model, "LLM")
        label = BASELINE_MODELS.get(model, f"{model}_{variant}")
        rows.append(dict(path=f, workload=workload, model=model, variant=variant,
                          label=label, cls=cls, **r))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # average across reps (rep1/rep2/rep3 subdirs) so each config is one row,
    # matching this script's original single-rep assumption.
    agg = df.groupby(["workload", "model", "variant", "label", "cls"], as_index=False).agg(
        mean_p50=("mean_p50", "mean"), mean_p90=("mean_p90", "mean"), mean_p99=("mean_p99", "mean"),
        sla_p50=("sla_p50", "mean"), sla_p90=("sla_p90", "mean"), sla_p99=("sla_p99", "mean"),
        mean_mean=("mean_mean", "mean"), n_steps=("n_steps", "mean"), n_reps=("path", "count"),
    )
    for c in ("mean_p50", "mean_p90", "mean_p99", "sla_p50", "sla_p90", "sla_p99", "mean_mean"):
        agg[c] = agg[c].round(1)
    agg["tail_ratio"] = (agg["mean_p99"] / agg["mean_p50"]).round(2)
    return agg


def flat_label(row):
    return row.label if row.cls != "LLM" else f"{DISPLAY.get(row.model, row.model)}/{row.variant}"


def _tag(fig):
    fig.text(0.5, 1.1, SOURCE_TAG, fontsize=10, fontweight="bold", color=TAG_COLOR,
              va="bottom", ha="center",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#FBEAE5", edgecolor=TAG_COLOR, linewidth=1.1))


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.png")


def fig_ladder(s, source_label, suffix):
    """Dot-and-line 'ladder' per config: P50 -> P90 -> P99, sorted by P99."""
    s = s.sort_values("mean_p99", ascending=True).reset_index(drop=True)
    labels = [flat_label(r) for _, r in s.iterrows()]
    y = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(s) + 2))
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.plot([r.mean_p50, r.mean_p99], [yi, yi], color=MUTED, lw=1.4, zorder=2)
        ax.scatter(r.mean_p50, yi, s=70, color=P_COLORS["p50"], edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter(r.mean_p90, yi, s=95, color=P_COLORS["p90"], edgecolors="white", linewidths=0.7, zorder=3)
        ax.scatter(r.mean_p99, yi, s=120, color=P_COLORS["p99"], edgecolors="white", linewidths=0.7, zorder=4)

    ax.axvline(SLA_MS, color="#E69F00", ls="--", lw=1.2, zorder=1)
    ax.text(SLA_MS, len(s) - 0.3, " 200ms SLA", color="#E69F00", fontsize=8.5, va="top", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Latency (ms, log scale)  —  mean of each per-step percentile over the run")
    ax.set_xscale("log")
    ax.set_title(f"Real cluster ({source_label}): P50 → P90 → P99 latency spread per config",
                 fontsize=12.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="o", color="w", markerfacecolor=P_COLORS[p],
                          markersize=9, label=p.upper(), markeredgecolor="gray") for p in ("p50", "p90", "p99")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _tag(fig)
    _save(fig, f"fig_percentile_ladder_real{suffix}")


def fig_tail_ratio(s, source_label, suffix):
    """P99/P50 ratio per config -- how much worse the tail is than the median."""
    s = s.sort_values("tail_ratio", ascending=False).reset_index(drop=True)
    labels = [flat_label(r) for _, r in s.iterrows()]
    x = np.arange(len(s))
    colors = ["#CC3311" if v >= 3 else ("#E69F00" if v >= 2 else "#4292C6") for v in s.tail_ratio]

    fig, ax = plt.subplots(figsize=(max(11, 0.5 * len(s)), 5))
    ax.bar(x, s.tail_ratio, color=colors, width=0.68, zorder=3)
    for xi, v in zip(x, s.tail_ratio):
        ax.text(xi, v + s.tail_ratio.max() * 0.015, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color=MUTED, lw=1, zorder=1)
    ax.set_ylabel("P99 / P50 latency ratio  (1x = flat, higher = worse tail)")
    ax.set_title(f"Real cluster ({source_label}): tail amplification — how much worse P99 is than P50",
                 fontsize=12.5, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#4292C6", "#E69F00", "#CC3311"]]
    ax.legend(handles, ["< 2x", "2x – 3x", "≥ 3x"], loc="upper right", frameon=False, fontsize=9, title="tail severity")
    fig.tight_layout()
    _tag(fig)
    _save(fig, f"fig_percentile_tail_ratio_real{suffix}")


def print_table(s):
    s = s.sort_values("mean_p99")
    hdr = f"{'config':26s} {'P50':>7s} {'P90':>7s} {'P99':>7s} {'mean':>7s}  {'SLA50':>6s} {'SLA90':>6s} {'SLA99':>6s}  {'P99/P50':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in s.iterrows():
        print(f"{flat_label(r):26s} {r.mean_p50:7.1f} {r.mean_p90:7.1f} {r.mean_p99:7.1f} {r.mean_mean:7.1f}  "
              f"{r.sla_p50:5.1f}% {r.sla_p90:5.1f}% {r.sla_p99:5.1f}%  {r.tail_ratio:7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_k8s_v2")
    ap.add_argument("--workload", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    s = load_summary(args.results_dir, args.workload or None)
    print(f"Loaded {len(s)} complete real-cluster runs from {args.results_dir}/ (workload={args.workload or 'all'})")
    if s.empty:
        print("No complete runs found.")
        return
    out = args.out or f"{args.results_dir.rstrip('/')}_{args.workload or 'all'}_percentiles.csv"
    s.sort_values("mean_p99").to_csv(out, index=False)
    print(f"Table -> {out}\n")
    print_table(s)

    if not args.no_plots:
        source_label = f"{args.results_dir}, {args.workload or 'all'} workload"
        suffix = f"_{args.workload}" if args.workload else ""
        fig_ladder(s, source_label, suffix)
        fig_tail_ratio(s, source_label, suffix)


if __name__ == "__main__":
    main()
