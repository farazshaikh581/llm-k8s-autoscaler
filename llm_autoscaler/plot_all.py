#!/usr/bin/env python3
"""
Comprehensive plotting & table generation for the LLM Autoscaler paper.

Produces all figures and LaTeX tables covering:
  - Long simulation (1440 steps, Alibaba trace)
  - K8s v2 real cluster (120 steps × 60s, CPU + IO workloads)
  - Workload characterization & comparison
  - Prompt variant ablation
  - Model size scaling
  - Cost-efficiency Pareto frontiers
  - Time-series overlays
  - LLM inference overhead

Usage:
    python plot_all.py              # generate all plots + tables
    python plot_all.py --dpi 300    # publication quality

Output: plots/ directory (PNGs + LaTeX .tex files + summary CSVs)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
LONG_DIR = BASE / "results_long"
K8S_DIR = BASE / "results_k8s_v2"
TRACE_FILE = BASE / "trace_alibaba_v2.npy"
TRACE_CPU = BASE / "traces" / "trace_cpu.npy"
TRACE_IO = BASE / "traces" / "trace_io.npy"
PLOTS_DIR = BASE / "plots"

SLA_MS = 200
LONG_STEPS = 1440
K8S_STEPS = 120

CORE_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b"]
SUPPLEMENTARY = ["gpt-oss-120b", "llama4-scout"]
BASELINES_LONG = ["hpa", "keda", "dqn", "ppo"]
VARIANT_ORDER = ["zero_shot", "domain", "history_5", "cot"]

MODEL_SIZES = {
    "llama-8b": 8, "llama4-scout": 17, "llama-70b": 70,
    "qwen3-80b": 80, "mistral-small4": 119, "gpt-oss-120b": 120,
}
MODEL_LABELS = {
    "llama-8b": "Llama 3.1 8B", "llama-70b": "Llama 3.3 70B",
    "mistral-small4": "Mistral Small 4", "qwen3-80b": "Qwen 3 80B",
    "gpt-oss-120b": "GPT-OSS 120B", "llama4-scout": "Llama 4 Scout",
    "hpa": "HPA", "keda": "KEDA", "dqn": "DQN (RL)", "ppo": "PPO (RL)",
}
VARIANT_LABELS = {
    "zero_shot": "Zero-Shot", "domain": "Domain", "history_5": "History-5",
    "cot": "Chain-of-Thought", "baseline": "Baseline", "rl": "RL",
}
MODEL_COLORS = {
    "llama-8b": "#2196F3", "llama-70b": "#FF9800", "mistral-small4": "#4CAF50",
    "qwen3-80b": "#E91E63", "gpt-oss-120b": "#9C27B0", "llama4-scout": "#00BCD4",
    "hpa": "#607D8B", "keda": "#795548", "dqn": "#FF5722", "ppo": "#8BC34A",
}
VARIANT_MARKERS = {"zero_shot": "o", "domain": "s", "history_5": "^", "cot": "D"}
VARIANT_LINESTYLES = {"zero_shot": "-", "domain": "--", "history_5": "-.", "cot": ":"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
})

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_long_sim():
    frames = []
    for f in sorted(LONG_DIR.glob("results_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) < LONG_STEPS:
            continue
        df = df.head(LONG_STEPS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_k8s_v2():
    frames = []
    known_variants = ["zero_shot", "history_5", "cot", "domain", "baseline"]
    for f in sorted(K8S_DIR.glob("k8s_*.csv")):
        if f.stat().st_size < 500:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) < K8S_STEPS:
            continue
        name = f.stem
        parts = name.split("_", 2)
        workload = parts[1].upper()
        rest = parts[2]
        model, variant = None, None
        for v in known_variants:
            if rest.endswith("_" + v):
                model = rest[:-(len(v) + 1)].replace("rl-", "")
                variant = v
                break
        if model is None:
            continue
        df["workload_type"] = workload
        df["model"] = model
        df["variant"] = variant
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_long(df):
    g = df.groupby(["llm_model", "llm_variant"])
    s = g.agg(
        mean_replicas=("ready_replicas", "mean"),
        std_replicas=("ready_replicas", "std"),
        mean_latency=("latency_p90", "mean"),
        p95_latency=("latency_p90", lambda x: x.quantile(0.95)),
        p99_latency=("latency_p90", lambda x: x.quantile(0.99)),
        max_latency=("latency_p90", "max"),
        mean_cpu=("cpu_pct", "mean"),
        mean_success=("success_rate", "mean"),
        sla_violations=("latency_p90", lambda x: (x > SLA_MS).sum()),
        total_vcpu=("vcpu_minutes", "last"),
        scale_events=("scale_event", "sum"),
        total_tokens=("llm_tokens_used", "sum"),
        mean_llm_lat=("llm_latency_ms", "mean"),
        steps=("step", "count"),
        mean_rps=("requests", "mean"),
    ).reset_index()
    s["sla_pct"] = round((1 - s["sla_violations"] / s["steps"]) * 100, 2)
    return s


def summarize_k8s(df):
    g = df.groupby(["model", "variant", "workload_type"])
    s = g.agg(
        mean_replicas=("ready_replicas", "mean"),
        std_replicas=("ready_replicas", "std"),
        mean_latency=("latency_p90_ms", "mean"),
        p95_latency=("latency_p90_ms", lambda x: x.quantile(0.95)),
        p99_latency=("latency_p90_ms", lambda x: x.quantile(0.99)),
        max_latency=("latency_p90_ms", "max"),
        mean_cpu_mc=("cpu_millicores", "mean"),
        mean_success=("success_rate", "mean"),
        sla_violations=("latency_p90_ms", lambda x: (x > SLA_MS).sum()),
        scale_events=("scale_event", "sum"),
        total_tokens=("llm_tokens", "sum"),
        mean_llm_lat=("llm_latency_ms", "mean"),
        steps=("step", "count"),
        mean_rps=("rps_target", "mean"),
    ).reset_index()
    s["sla_pct"] = round((1 - s["sla_violations"] / s["steps"]) * 100, 2)
    return s

# ─── Plot Helpers ────────────────────────────────────────────────────────────

def _label(model):
    return MODEL_LABELS.get(model, model)

def _save(fig, name, dpi):
    out = PLOTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Workload Trace Characterization
# ═══════════════════════════════════════════════════════════════════════════

def plot_workload_traces(dpi):
    trace_main = np.load(TRACE_FILE)
    trace_cpu = np.load(TRACE_CPU)
    trace_io = np.load(TRACE_IO)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    for ax, trace, title, color in [
        (axes[0], trace_main, "Combined Trace (Long Simulation)", "#2196F3"),
        (axes[1], trace_cpu, "CPU-Intensive Trace (K8s v2)", "#E91E63"),
        (axes[2], trace_io, "I/O-Intensive Trace (K8s v2)", "#4CAF50"),
    ]:
        steps = np.arange(len(trace))
        ax.fill_between(steps, trace, alpha=0.3, color=color)
        ax.plot(steps, trace, color=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel("RPS")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

        stats_text = f"min={trace.min()}, max={trace.max()}, mean={trace.mean():.0f}, std={trace.std():.0f}"
        ax.text(0.98, 0.92, stats_text, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[-1].set_xlabel("Step (minutes)")
    fig.suptitle("Alibaba Cluster Trace 2018 — Workload Characterization", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "01_workload_traces", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Long Sim — Main Summary Bar Chart (SLA + Cost + Scale Events)
# ═══════════════════════════════════════════════════════════════════════════

def plot_long_summary_bars(ls, dpi):
    core = ls[ls["llm_model"].isin(CORE_MODELS)].copy()
    base = ls[ls["llm_model"].isin(BASELINES_LONG)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models_in_order = [m for m in CORE_MODELS if m in core["llm_model"].values]
    variants_present = [v for v in VARIANT_ORDER if v in core["llm_variant"].values]
    x = np.arange(len(models_in_order))
    width = 0.18
    variant_colors = {"zero_shot": "#2196F3", "domain": "#4CAF50", "history_5": "#FF9800", "cot": "#E91E63"}

    for metric_idx, (metric, ylabel, title) in enumerate([
        ("sla_pct", "SLA Compliance (%)", "SLA Compliance"),
        ("total_vcpu", "vCPU-minutes", "Infrastructure Cost"),
        ("scale_events", "Number of Scale Events", "Scaling Stability"),
    ]):
        ax = axes[metric_idx]
        for vi, v in enumerate(variants_present):
            vals = []
            for m in models_in_order:
                row = core[(core["llm_model"] == m) & (core["llm_variant"] == v)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (vi - len(variants_present)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=VARIANT_LABELS[v],
                          color=variant_colors[v], alpha=0.85, edgecolor="white", linewidth=0.5)

        for _, br in base.iterrows():
            ax.axhline(y=br[metric], linestyle="--", alpha=0.5, linewidth=1.5,
                       color=MODEL_COLORS.get(br["llm_model"], "gray"),
                       label=_label(br["llm_model"]))

        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in models_in_order], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Long Simulation (1440 steps) — Core Models x Prompt Variants", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "02_long_sim_summary_bars", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Long Sim — Time-Series Comparison (Best LLM vs Baselines)
# ═══════════════════════════════════════════════════════════════════════════

def plot_long_timeseries(long_df, dpi):
    SMOOTH_W = 15
    trace = np.load(TRACE_FILE)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 1, 1], hspace=0.30, wspace=0.25)

    ax_rps = fig.add_subplot(gs[0, :])
    ax_rep_llm = fig.add_subplot(gs[1, 0], sharex=ax_rps)
    ax_rep_base = fig.add_subplot(gs[1, 1], sharex=ax_rps)
    ax_lat_llm = fig.add_subplot(gs[2, 0], sharex=ax_rps)
    ax_lat_base = fig.add_subplot(gs[2, 1], sharex=ax_rps)

    steps = np.arange(LONG_STEPS)
    ax_rps.fill_between(steps, trace[:LONG_STEPS], alpha=0.2, color="gray")
    ax_rps.plot(steps, trace[:LONG_STEPS], "gray", alpha=0.5, linewidth=0.6)
    ax_rps.set_ylabel("RPS")
    ax_rps.set_title("Workload Trace (Alibaba 2018)")
    ax_rps.grid(True, alpha=0.2)
    plt.setp(ax_rps.get_xticklabels(), visible=False)

    optimal = np.ceil(trace[:LONG_STEPS] / 200).astype(int).clip(1, 20)
    optimal_smooth = pd.Series(optimal).rolling(SMOOTH_W, center=True, min_periods=1).mean()
    for ax in [ax_rep_llm, ax_rep_base]:
        ax.plot(steps, optimal_smooth, "k:", alpha=0.35, label="Optimal", linewidth=1)

    llm_configs = [
        ("llama-70b", "domain", "-", 2.5, 0.9),
        ("llama-8b", "domain", "-", 2.0, 0.85),
        ("mistral-small4", "domain", "-.", 2.0, 0.85),
        ("qwen3-80b", "domain", "--", 2.0, 0.85),
    ]
    baseline_configs = [
        ("hpa", "baseline", "-", 2.5, 0.9),
        ("keda", "baseline", "--", 2.0, 0.85),
        ("dqn", "rl", "-.", 2.0, 0.85),
        ("ppo", "rl", ":", 2.0, 0.85),
    ]

    def _smooth(series):
        return series.rolling(SMOOTH_W, center=True, min_periods=1).mean()

    for model, variant, linestyle, lw, alpha in llm_configs:
        mask = (long_df["llm_model"] == model) & (long_df["llm_variant"] == variant)
        d = long_df[mask].sort_values("step")
        if d.empty:
            continue
        color = MODEL_COLORS.get(model, "gray")
        lab = _label(model)
        ax_rep_llm.plot(d["step"], _smooth(d["ready_replicas"]), linestyle, color=color,
                        alpha=alpha, linewidth=lw, label=lab)
        ax_lat_llm.plot(d["step"], _smooth(d["latency_p90"]), linestyle, color=color,
                        alpha=alpha, linewidth=lw, label=lab)

    for model, variant, linestyle, lw, alpha in baseline_configs:
        mask = (long_df["llm_model"] == model) & (long_df["llm_variant"] == variant)
        d = long_df[mask].sort_values("step")
        if d.empty:
            continue
        color = MODEL_COLORS.get(model, "gray")
        lab = _label(model)
        ax_rep_base.plot(d["step"], _smooth(d["ready_replicas"]), linestyle, color=color,
                         alpha=alpha, linewidth=lw, label=lab)
        ax_lat_base.plot(d["step"], _smooth(d["latency_p90"]), linestyle, color=color,
                         alpha=alpha, linewidth=lw, label=lab)

    for ax in [ax_lat_llm, ax_lat_base]:
        ax.axhline(y=SLA_MS, color="red", linestyle=":", alpha=0.6, linewidth=1.5, label="SLA (200ms)")

    for ax in [ax_rep_llm, ax_rep_base]:
        ax.set_ylabel("Ready Replicas")
        ax.set_ylim(0, 21)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_rep_llm.set_title("Replicas — LLM Autoscalers (domain)")
    ax_rep_base.set_title("Replicas — Baselines")

    for ax in [ax_lat_llm, ax_lat_base]:
        ax.set_ylabel("Latency P90 (ms)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Step (minutes)")
        ax.legend(fontsize=8, loc="upper right")
    ax_lat_llm.set_title("Tail Latency — LLM Autoscalers (domain)")
    ax_lat_base.set_title("Tail Latency — Baselines")

    fig.suptitle("Long Simulation (1440 steps) — Time-Series: LLMs vs Baselines", fontsize=14, y=1.01)
    _save(fig, "03_long_sim_timeseries", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Prompt Variant Ablation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_heatmap(ls, dpi):
    core = ls[ls["llm_model"].isin(CORE_MODELS + SUPPLEMENTARY)]
    if core.empty:
        return

    models = [m for m in CORE_MODELS + SUPPLEMENTARY if m in core["llm_model"].values]
    variants = [v for v in VARIANT_ORDER if v in core["llm_variant"].values]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("sla_pct", "SLA Compliance (%)", "RdYlGn", False),
        ("mean_latency", "Mean Latency P90 (ms)", "RdYlGn_r", False),
        ("total_vcpu", "Cost (vCPU-min)", "RdYlGn_r", False),
        ("scale_events", "Scale Events", "RdYlGn_r", True),
        ("mean_replicas", "Mean Replicas", "Blues", False),
        ("mean_llm_lat", "LLM Inference Latency (ms)", "Oranges", False),
    ]

    for ax, (metric, title, cmap, use_log) in zip(axes.flat, metrics):
        pivot = core.pivot(index="llm_model", columns="llm_variant", values=metric)
        pivot = pivot.reindex(index=models, columns=variants)
        labels_y = [_label(m) for m in pivot.index]

        data = pivot.values.copy()
        if use_log:
            im_data = np.log10(data + 1)
        else:
            im_data = data

        im = ax.imshow(im_data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=8)
        ax.set_title(title, fontsize=10)

        for i in range(len(labels_y)):
            for j in range(len(variants)):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
                else:
                    fmt = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                            color="white" if im_data[i, j] > np.nanmean(im_data) * 1.3 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Prompt Variant Ablation — Long Simulation (1440 steps)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "04_ablation_heatmap", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Model Size vs Performance (zero_shot)
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_size_scaling(ls, dpi):
    all_models = CORE_MODELS + SUPPLEMENTARY
    zs = ls[(ls["llm_model"].isin(all_models)) & (ls["llm_variant"] == "zero_shot")].copy()
    base = ls[ls["llm_model"].isin(BASELINES_LONG)]
    if zs.empty:
        return

    zs["size"] = zs["llm_model"].map(MODEL_SIZES)
    zs = zs.sort_values("size")

    model_order = zs["llm_model"].tolist()
    x_labels = [f"{_label(m)}\n({MODEL_SIZES[m]}B)" for m in model_order]
    x = np.arange(len(model_order))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    metrics = [
        ("sla_pct", "SLA Compliance (%)", False),
        ("mean_latency", "Mean Latency P90 (ms)", True),
        ("total_vcpu", "Cost (vCPU-min)", False),
        ("scale_events", "Scale Events", True),
    ]

    for ax, (metric, ylabel, use_log) in zip(axes, metrics):
        colors = [MODEL_COLORS.get(m, "gray") for m in model_order]
        vals = [zs[zs["llm_model"] == m][metric].values[0] for m in model_order]
        bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor="white",
                      linewidth=0.8, zorder=5)

        for bar, v in zip(bars, vals):
            fmt = f"{v:.1f}" if v < 100 else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")

        for _, br in base.iterrows():
            ax.axhline(y=br[metric], linestyle="--", alpha=0.6, linewidth=1.5,
                       color=MODEL_COLORS.get(br["llm_model"], "gray"),
                       label=_label(br["llm_model"]))

        if use_log and max(vals) > 10 * np.median(vals):
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Zero-Shot Performance vs Model Size (Larger ≠ Better)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "05_model_size_scaling", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: Cost-Efficiency Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_sla_pareto(ls, dpi):
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(14, 9))

    core = ls[ls["llm_model"].isin(CORE_MODELS)]
    base = ls[ls["llm_model"].isin(BASELINES_LONG)]

    texts = []
    all_x, all_y = [], []

    for _, r in core.iterrows():
        color = MODEL_COLORS[r["llm_model"]]
        marker = VARIANT_MARKERS.get(r["llm_variant"], "o")
        ax.scatter(r["total_vcpu"], r["sla_pct"], s=140, c=color, marker=marker,
                   alpha=0.85, edgecolors="white", linewidth=0.8, zorder=5)
        short_model = _label(r["llm_model"]).split()[-1]
        short_variant = VARIANT_LABELS.get(r["llm_variant"], r["llm_variant"])
        txt = ax.text(r["total_vcpu"], r["sla_pct"],
                      f" {short_model}/{short_variant}", fontsize=7, ha="left", va="center")
        texts.append(txt)
        all_x.append(r["total_vcpu"])
        all_y.append(r["sla_pct"])

    for _, r in base.iterrows():
        color = MODEL_COLORS.get(r["llm_model"], "gray")
        ax.scatter(r["total_vcpu"], r["sla_pct"], s=220, c=color, marker="*",
                   alpha=0.9, edgecolors="black", linewidth=0.5, zorder=6)
        txt = ax.text(r["total_vcpu"], r["sla_pct"],
                      f" {_label(r['llm_model'])}", fontsize=8, fontweight="bold",
                      ha="left", va="center")
        texts.append(txt)
        all_x.append(r["total_vcpu"])
        all_y.append(r["sla_pct"])

    pareto_pts = sorted(zip(all_x, all_y), key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    best_sla = -1
    for cx, cy in pareto_pts:
        if cy > best_sla:
            frontier_x.append(cx)
            frontier_y.append(cy)
            best_sla = cy
    if len(frontier_x) > 1:
        ax.plot(frontier_x, frontier_y, "k--", alpha=0.4, linewidth=1.5, zorder=3,
                label="Pareto frontier")

    ax.axhline(y=99, color="green", linestyle=":", alpha=0.5, linewidth=1.5, label="99% SLA target")
    ax.axhline(y=95, color="orange", linestyle=":", alpha=0.5, linewidth=1.5, label="95% SLA target")

    y_min = min(all_y) - 2
    ax.set_ylim(max(y_min, 75), 101)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
                force_text=(0.8, 0.8), force_points=(0.5, 0.5), expand=(1.5, 1.5))

    legend_elements = []
    for m in CORE_MODELS:
        legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m],
                                      markersize=8, label=_label(m)))
    for m in BASELINES_LONG:
        legend_elements.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=MODEL_COLORS[m],
                                      markersize=10, label=_label(m)))
    for v in VARIANT_ORDER:
        legend_elements.append(Line2D([0], [0], marker=VARIANT_MARKERS[v], color="w",
                                      markerfacecolor="gray", markersize=8, label=VARIANT_LABELS[v]))
    legend_elements.append(Line2D([0], [0], linestyle="--", color="black", alpha=0.4, label="Pareto frontier"))
    ax.legend(handles=legend_elements, fontsize=7, ncol=3, loc="lower left")

    ax.set_xlabel("Cost (vCPU-minutes)")
    ax.set_ylabel("SLA Compliance (%)")
    ax.set_title("Cost-Efficiency Frontier: LLM Autoscalers vs Baselines (Long Sim)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "06_cost_sla_pareto", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7: K8s v2 — Summary Grouped Bars (CPU vs IO)
# ═══════════════════════════════════════════════════════════════════════════

def plot_k8s_summary_bars(ks, dpi):
    core = ks[ks["model"].isin(CORE_MODELS)]
    base = ks[ks["model"].isin(BASELINES_LONG)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    variant_colors = {"zero_shot": "#2196F3", "domain": "#4CAF50", "history_5": "#FF9800",
                      "cot": "#E91E63", "baseline": "#607D8B"}

    for row_idx, wl in enumerate(["CPU", "IO"]):
        wl_core = core[core["workload_type"] == wl]
        wl_base = base[base["workload_type"] == wl]
        models_present = [m for m in CORE_MODELS if m in wl_core["model"].values]
        variants_present = [v for v in VARIANT_ORDER if v in wl_core["variant"].values]
        x = np.arange(len(models_present))
        width = 0.18

        for col_idx, (metric, ylabel, title) in enumerate([
            ("sla_pct", "SLA Compliance (%)", f"{wl} Workload — SLA"),
            ("mean_latency", "Mean Latency P90 (ms)", f"{wl} Workload — Latency"),
            ("scale_events", "Scale Events", f"{wl} Workload — Stability"),
        ]):
            ax = axes[row_idx, col_idx]
            for vi, v in enumerate(variants_present):
                vals = []
                for m in models_present:
                    row = wl_core[(wl_core["model"] == m) & (wl_core["variant"] == v)]
                    vals.append(row[metric].values[0] if len(row) > 0 else 0)
                offset = (vi - len(variants_present)/2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=VARIANT_LABELS[v],
                       color=variant_colors[v], alpha=0.85, edgecolor="white", linewidth=0.5)

            for _, br in wl_base.iterrows():
                ax.axhline(y=br[metric], linestyle="--", alpha=0.5, linewidth=1.5,
                           color=MODEL_COLORS.get(br["model"], "gray"),
                           label=_label(br["model"]))

            ax.set_xticks(x)
            ax.set_xticklabels([_label(m) for m in models_present], rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("K8s v2 Real Cluster (120 steps x 60s) — CPU vs I/O Workloads", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "07_k8s_summary_bars", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8: K8s v2 — CPU vs IO Workload Comparison (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════

def plot_k8s_workload_comparison(ks, dpi):
    core = ks[ks["model"].isin(CORE_MODELS)]

    rows = []
    for model in CORE_MODELS:
        for variant in VARIANT_ORDER:
            cpu_row = core[(core["model"] == model) & (core["variant"] == variant) &
                           (core["workload_type"] == "CPU")]
            io_row = core[(core["model"] == model) & (core["variant"] == variant) &
                          (core["workload_type"] == "IO")]
            if cpu_row.empty or io_row.empty:
                continue
            rows.append({
                "model": model, "variant": variant,
                "label": f"{_label(model)} / {VARIANT_LABELS[variant]}",
                "cpu_sla": cpu_row["sla_pct"].values[0],
                "io_sla": io_row["sla_pct"].values[0],
                "cpu_lat": cpu_row["mean_latency"].values[0],
                "io_lat": io_row["mean_latency"].values[0],
                "cpu_scale": cpu_row["scale_events"].values[0],
                "io_scale": io_row["scale_events"].values[0],
            })
    data = pd.DataFrame(rows)
    data = data.sort_values(["model", "variant"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    metrics = [
        ("cpu_sla", "io_sla", "SLA Compliance (%)"),
        ("cpu_lat", "io_lat", "Mean Latency P90 (ms)"),
        ("cpu_scale", "io_scale", "Scale Events"),
    ]

    y_pos = np.arange(len(data))

    for ax, (cpu_col, io_col, xlabel) in zip(axes, metrics):
        for i, row in data.iterrows():
            color = MODEL_COLORS[row["model"]]
            cpu_val = row[cpu_col]
            io_val = row[io_col]
            ax.plot([cpu_val, io_val], [i, i], "-", color=color, alpha=0.4, linewidth=2, zorder=3)
            ax.scatter(cpu_val, i, s=100, c=color, marker="s", edgecolors="white",
                       linewidth=0.8, zorder=5, label="CPU" if i == 0 else None)
            ax.scatter(io_val, i, s=100, c=color, marker="o", edgecolors="white",
                       linewidth=0.8, zorder=5, alpha=0.6, label="IO" if i == 0 else None)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["label"], fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.15, axis="x")
        ax.invert_yaxis()

        for m_idx, model in enumerate(CORE_MODELS):
            block = data[data["model"] == model]
            if block.empty:
                continue
            y_start = block.index[0] - 0.5
            y_end = block.index[-1] + 0.5
            if m_idx % 2 == 0:
                ax.axhspan(y_start, y_end, color="gray", alpha=0.04, zorder=0)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=9, label="CPU workload"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=9, alpha=0.6, label="I/O workload"),
        Line2D([0], [0], linestyle="-", color="gray", alpha=0.4, linewidth=2,
               label="CPU-IO gap"),
    ]
    for m in CORE_MODELS:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=MODEL_COLORS[m], markersize=8,
                                      label=_label(m)))
    axes[1].legend(handles=legend_elements, fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("CPU vs I/O Workload Performance — Dumbbell Comparison (K8s v2 Real Cluster)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "08_k8s_workload_comparison", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 9: K8s v2 — Time-Series per Workload
# ═══════════════════════════════════════════════════════════════════════════

def plot_k8s_timeseries(k8s_df, dpi):
    SMOOTH_W = 5

    llm_styles = [
        ("llama-8b", "domain", "-", 2.5, 0.9),
        ("llama-70b", "domain", "--", 2.0, 0.85),
        ("mistral-small4", "domain", "-.", 2.0, 0.85),
        ("qwen3-80b", "domain", ":", 2.0, 0.85),
    ]
    base_styles = [
        ("hpa", "baseline", "-", 2.5, 0.9),
        ("keda", "baseline", "--", 2.0, 0.85),
        ("dqn", "baseline", "-.", 2.0, 0.85),
        ("ppo", "baseline", ":", 2.0, 0.85),
    ]

    def _sm(series):
        return series.rolling(SMOOTH_W, center=True, min_periods=1).mean()

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    col_titles = ["Replicas — LLMs", "Replicas — Baselines",
                  "Tail Latency — LLMs", "Tail Latency — Baselines"]

    for row_idx, wl in enumerate(["CPU", "IO"]):
        wl_data = k8s_df[k8s_df["workload_type"] == wl]

        for model, variant, ls_style, lw, alpha in llm_styles:
            d = wl_data[(wl_data["model"] == model) & (wl_data["variant"] == variant)].sort_values("step")
            if d.empty:
                continue
            color = MODEL_COLORS[model]
            lab = _label(model)
            axes[row_idx, 0].plot(d["step"], _sm(d["ready_replicas"]), ls_style, color=color,
                                  alpha=alpha, linewidth=lw, label=lab)
            axes[row_idx, 2].plot(d["step"], _sm(d["latency_p90_ms"]), ls_style, color=color,
                                  alpha=alpha, linewidth=lw, label=lab)

        for model, variant, ls_style, lw, alpha in base_styles:
            d = wl_data[(wl_data["model"] == model) & (wl_data["variant"] == variant)].sort_values("step")
            if d.empty:
                continue
            color = MODEL_COLORS.get(model, "gray")
            lab = _label(model)
            axes[row_idx, 1].plot(d["step"], _sm(d["ready_replicas"]), ls_style, color=color,
                                  alpha=alpha, linewidth=lw, label=lab)
            axes[row_idx, 3].plot(d["step"], _sm(d["latency_p90_ms"]), ls_style, color=color,
                                  alpha=alpha, linewidth=lw, label=lab)

        for col in [2, 3]:
            axes[row_idx, col].axhline(y=SLA_MS, color="red", linestyle=":", alpha=0.6,
                                        linewidth=1.5, label="SLA (200ms)")

        axes[row_idx, 0].set_ylabel(f"{wl} Workload\nReady Replicas")
        axes[row_idx, 2].set_ylabel("Latency P90 (ms)")

        for col in range(4):
            axes[row_idx, col].legend(fontsize=7, loc="upper right")
            axes[row_idx, col].grid(True, alpha=0.2)
            if row_idx == 1:
                axes[row_idx, col].set_xlabel("Step (x60s)")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)

    fig.suptitle("K8s v2 Real Cluster — Time-Series (domain variant)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "09_k8s_timeseries", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 10: LLM Inference Overhead
# ═══════════════════════════════════════════════════════════════════════════

def plot_llm_overhead(ls, dpi):
    llm = ls[ls["llm_model"].isin(CORE_MODELS)].copy()
    if llm.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = [m for m in CORE_MODELS if m in llm["llm_model"].values]
    variants_present = [v for v in VARIANT_ORDER if v in llm["llm_variant"].values]
    x = np.arange(len(models))
    width = 0.18
    variant_colors = {"zero_shot": "#2196F3", "domain": "#4CAF50", "history_5": "#FF9800", "cot": "#E91E63"}

    for ax, (metric, ylabel, title) in zip(axes, [
        ("total_tokens", "Total Tokens", "Token Usage"),
        ("mean_llm_lat", "Mean LLM Latency (ms)", "Inference Latency"),
        ("mean_latency", "App Latency P90 (ms)", "Application Latency Impact"),
    ]):
        for vi, v in enumerate(variants_present):
            vals = []
            for m in models:
                row = llm[(llm["llm_model"] == m) & (llm["llm_variant"] == v)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (vi - len(variants_present)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=VARIANT_LABELS[v],
                   color=variant_colors[v], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in models], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("LLM Inference Overhead — Long Simulation", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "10_llm_inference_overhead", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 11: Simulation vs Real Cluster Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_sim_vs_real(ls, ks, dpi):
    from adjustText import adjust_text

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    metrics = [
        ("sla_pct", "SLA Compliance (%)"),
        ("mean_latency", "Mean Latency P90 (ms)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        texts = []
        for model in CORE_MODELS:
            for variant in VARIANT_ORDER:
                sim_row = ls[(ls["llm_model"] == model) & (ls["llm_variant"] == variant)]
                k8s_row = ks[(ks["model"] == model) & (ks["variant"] == variant) &
                             (ks["workload_type"] == "CPU")]
                if sim_row.empty or k8s_row.empty:
                    continue
                sim_val = sim_row[metric].values[0]
                k8s_val = k8s_row["mean_latency"].values[0] if metric == "mean_latency" else k8s_row[metric].values[0]

                ax.scatter(sim_val, k8s_val, s=140, c=MODEL_COLORS[model],
                           marker=VARIANT_MARKERS.get(variant, "o"), alpha=0.85,
                           edgecolors="white", linewidth=1.0, zorder=5)
                short_model = _label(model).split()[-1]
                short_variant = VARIANT_LABELS.get(variant, variant)[:3]
                txt = ax.text(sim_val, k8s_val, f" {short_model}/{short_variant}",
                              fontsize=7, ha="left", va="center")
                texts.append(txt)

        for bm in BASELINES_LONG:
            sim_row = ls[ls["llm_model"] == bm]
            k8s_row = ks[(ks["model"] == bm) & (ks["workload_type"] == "CPU")]
            if sim_row.empty or k8s_row.empty:
                continue
            sim_val = sim_row[metric].values[0]
            k8s_val = k8s_row["mean_latency"].values[0] if metric == "mean_latency" else k8s_row[metric].values[0]
            ax.scatter(sim_val, k8s_val, s=250, c=MODEL_COLORS.get(bm, "gray"),
                       marker="*", alpha=0.9, edgecolors="black", linewidth=0.5, zorder=6)
            txt = ax.text(sim_val, k8s_val, f" {_label(bm)}",
                          fontsize=8.5, fontweight="bold", ha="left", va="center")
            texts.append(txt)

        lims_all = list(ax.get_xlim()) + list(ax.get_ylim())
        lo, hi = min(lims_all), max(lims_all)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k:", alpha=0.3, linewidth=1.5)
        ax.fill_between([lo, hi], [lo, hi], hi, alpha=0.04, color="green", label="Real > Sim")
        ax.fill_between([lo, hi], lo, [lo, hi], alpha=0.04, color="red", label="Sim > Real")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Simulation — {ylabel}")
        ax.set_ylabel(f"Real Cluster (CPU) — {ylabel}")
        ax.grid(True, alpha=0.2)

        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
                    force_text=(1.0, 1.0), force_points=(0.6, 0.6), expand=(1.8, 1.8),
                    iterations=200)

    legend_elements = []
    for m in CORE_MODELS:
        legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m],
                                      markersize=9, label=_label(m)))
    for m in BASELINES_LONG:
        legend_elements.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=MODEL_COLORS.get(m, "gray"),
                                      markersize=11, label=_label(m)))
    for v in VARIANT_ORDER:
        legend_elements.append(Line2D([0], [0], marker=VARIANT_MARKERS[v], color="w",
                                      markerfacecolor="gray", markersize=9, label=VARIANT_LABELS[v]))
    axes[0].legend(handles=legend_elements, fontsize=8, ncol=3, loc="lower right")

    fig.suptitle("Simulation Fidelity: Simulated vs Real K8s Cluster (CPU Workload)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "11_sim_vs_real", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 12: Radar / Spider Chart — Per-Model Profile
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar_profiles(ls, dpi):
    core = ls[ls["llm_model"].isin(CORE_MODELS) & (ls["llm_variant"] == "domain")]
    base_hpa = ls[ls["llm_model"] == "hpa"]
    if core.empty:
        return

    categories = ["SLA %", "1/Latency", "1/Cost", "Stability\n(1/Scales)", "CPU Eff"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    def normalize(series, higher_better=True):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        if higher_better:
            return (series - mn) / (mx - mn)
        return (mx - series) / (mx - mn)

    all_data = pd.concat([core, base_hpa])
    sla_norm = normalize(all_data["sla_pct"])
    lat_norm = normalize(all_data["mean_latency"], higher_better=False)
    cost_norm = normalize(all_data["total_vcpu"], higher_better=False)
    scale_norm = normalize(all_data["scale_events"], higher_better=False)
    cpu_norm = normalize(all_data["mean_cpu"])

    for idx, (_, r) in enumerate(all_data.iterrows()):
        model = r["llm_model"]
        values = [sla_norm.iloc[idx], lat_norm.iloc[idx], cost_norm.iloc[idx],
                  scale_norm.iloc[idx], cpu_norm.iloc[idx]]
        values += values[:1]
        color = MODEL_COLORS.get(model, "gray")
        ls_style = "--" if model in BASELINES_LONG else "-"
        ax.plot(angles, values, ls_style, linewidth=2, color=color, label=_label(model))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Model Performance Profiles (domain variant vs HPA)", fontsize=12, pad=20)
    plt.tight_layout()
    _save(fig, "12_radar_profiles", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 13: Per-Variant Time-Series Comparison (one model, all variants)
# ═══════════════════════════════════════════════════════════════════════════

def plot_variant_timeseries(long_df, dpi):
    SMOOTH_W = 15
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)

    def _sm(series):
        return series.rolling(SMOOTH_W, center=True, min_periods=1).mean()

    for col_idx, model in enumerate(["llama-8b", "qwen3-80b"]):
        for vi, variant in enumerate(VARIANT_ORDER):
            d = long_df[(long_df["llm_model"] == model) & (long_df["llm_variant"] == variant)].sort_values("step")
            if d.empty:
                continue
            color = {"zero_shot": "#2196F3", "domain": "#4CAF50", "history_5": "#FF9800", "cot": "#E91E63"}[variant]
            axes[vi, col_idx].plot(d["step"], _sm(d["ready_replicas"]), "-", color=color,
                                    alpha=0.9, linewidth=2.0, label="Replicas")
            ax2 = axes[vi, col_idx].twinx()
            ax2.fill_between(d["step"], 0, _sm(d["latency_p90"]), color="red", alpha=0.08)
            ax2.plot(d["step"], _sm(d["latency_p90"]), "-", color="red", alpha=0.6, linewidth=1.2, label="Latency P90")
            ax2.axhline(y=SLA_MS, color="red", linestyle="--", alpha=0.4, linewidth=1.0)
            ax2.set_ylabel("Latency (ms)", color="red", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="red", labelsize=8)

            axes[vi, col_idx].set_ylabel("Replicas")
            title = f"{_label(model)} — {VARIANT_LABELS[variant]}"
            sla_row = long_df[(long_df["llm_model"] == model) & (long_df["llm_variant"] == variant)]
            sla_val = (1 - (sla_row["latency_p90"] > SLA_MS).sum() / len(sla_row)) * 100
            axes[vi, col_idx].set_title(f"{title}  (SLA={sla_val:.1f}%)", fontsize=10)
            axes[vi, col_idx].grid(True, alpha=0.2)
            axes[vi, col_idx].set_ylim(0, 21)

        axes[-1, col_idx].set_xlabel("Step (minutes)")

    fig.suptitle("Prompt Variant Effect on Scaling — Llama 8B (stable) vs Qwen 80B (volatile)", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "13_variant_timeseries", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 14: K8s v2 — Workload Difficulty Analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_workload_difficulty(ks, dpi):
    all_data = ks.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, wl, color in [(axes[0], "CPU", "#E91E63"), (axes[1], "IO", "#4CAF50")]:
        wl_data = all_data[all_data["workload_type"] == wl]
        models = list(CORE_MODELS) + list(BASELINES_LONG)
        models_present = [m for m in models if m in wl_data["model"].values]

        sla_vals = []
        labels = []
        colors_list = []
        for m in models_present:
            for v in (VARIANT_ORDER if m in CORE_MODELS else ["baseline"]):
                row = wl_data[(wl_data["model"] == m) & (wl_data["variant"] == v)]
                if row.empty:
                    continue
                sla_vals.append(row["sla_pct"].values[0])
                vl = VARIANT_LABELS.get(v, v)
                labels.append(f"{_label(m)}\n{vl}")
                colors_list.append(MODEL_COLORS.get(m, "gray"))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, sla_vals, color=colors_list, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("SLA Compliance (%)")
        ax.set_title(f"{wl} Workload")
        ax.axvline(x=99, color="green", linestyle=":", alpha=0.5, label="99% target")
        ax.axvline(x=95, color="orange", linestyle=":", alpha=0.5, label="95% target")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_xlim(50, 101)
        ax.invert_yaxis()

    fig.suptitle("K8s v2 — Workload Difficulty: CPU (hard) vs I/O (easy)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "14_workload_difficulty", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 15: Box plots — Latency Distribution per Model
# ═══════════════════════════════════════════════════════════════════════════

def plot_latency_distributions(long_df, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models_to_plot = CORE_MODELS + BASELINES_LONG

    ax = axes[0]
    data_boxes = []
    labels_boxes = []
    colors_boxes = []
    for m in models_to_plot:
        if m in CORE_MODELS:
            d = long_df[(long_df["llm_model"] == m) & (long_df["llm_variant"] == "domain")]
        else:
            d = long_df[long_df["llm_model"] == m]
        if d.empty:
            continue
        data_boxes.append(d["latency_p90"].values)
        labels_boxes.append(_label(m))
        colors_boxes.append(MODEL_COLORS.get(m, "gray"))

    bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors_boxes):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=SLA_MS, color="red", linestyle=":", alpha=0.5, label="SLA (200ms)")
    ax.set_ylabel("Latency P90 (ms)")
    ax.set_title("Latency Distribution by Model (best variant)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    data_boxes = []
    labels_boxes = []
    for v in VARIANT_ORDER:
        d = long_df[(long_df["llm_model"].isin(CORE_MODELS)) & (long_df["llm_variant"] == v)]
        if d.empty:
            continue
        data_boxes.append(d["latency_p90"].values)
        labels_boxes.append(VARIANT_LABELS.get(v, v))

    bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    variant_colors_list = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for patch, color in zip(bp["boxes"], variant_colors_list[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=SLA_MS, color="red", linestyle=":", alpha=0.5, label="SLA (200ms)")
    ax.set_ylabel("Latency P90 (ms)")
    ax.set_title("Latency Distribution by Prompt Variant (core models)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Latency Distributions — Long Simulation (1440 steps)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "15_latency_distributions", dpi)


# ═══════════════════════════════════════════════════════════════════════════
# TABLES: LaTeX
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_tables(ls, ks):
    # Table 1: Long sim summary
    tex = PLOTS_DIR / "table_long_sim.tex"
    with open(tex, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{Long Simulation Results (1440 steps, Alibaba Cluster Trace 2018). "
                "SLA threshold: P90 latency $<$200\\,ms.}\n")
        f.write("\\label{tab:long_sim}\n")
        f.write("\\begin{tabular}{llrrrrrrrr}\n\\toprule\n")
        f.write("Model & Variant & Replicas & Lat P90 & P99 & CPU\\% & SLA\\% & "
                "vCPU-min & Scales & Tokens \\\\\n")
        f.write("\\midrule\n")
        core = ls[ls["llm_model"].isin(CORE_MODELS)].sort_values(
            ["llm_model", "llm_variant"],
            key=lambda s: s.map({v: i for i, v in enumerate(CORE_MODELS)}) if s.name == "llm_model"
            else s.map({v: i for i, v in enumerate(VARIANT_ORDER)}))
        prev = None
        for _, r in core.iterrows():
            if prev and r["llm_model"] != prev:
                f.write("\\midrule\n")
            prev = r["llm_model"]
            sla_fmt = f"\\textbf{{{r['sla_pct']:.1f}}}" if r["sla_pct"] >= 99.5 else f"{r['sla_pct']:.1f}"
            f.write(f"{_label(r['llm_model'])} & {VARIANT_LABELS.get(r['llm_variant'], r['llm_variant'])} & "
                    f"{r['mean_replicas']:.1f} & {r['mean_latency']:.1f} & {r['p99_latency']:.0f} & "
                    f"{r['mean_cpu']:.0f} & {sla_fmt} & {r['total_vcpu']:.0f} & "
                    f"{r['scale_events']:.0f} & {r['total_tokens']:.0f} \\\\\n")
        f.write("\\midrule\n")
        base = ls[ls["llm_model"].isin(BASELINES_LONG)]
        for _, r in base.iterrows():
            sla_fmt = f"\\textbf{{{r['sla_pct']:.1f}}}" if r["sla_pct"] >= 99.5 else f"{r['sla_pct']:.1f}"
            f.write(f"{_label(r['llm_model'])} & — & "
                    f"{r['mean_replicas']:.1f} & {r['mean_latency']:.1f} & {r['p99_latency']:.0f} & "
                    f"{r['mean_cpu']:.0f} & {sla_fmt} & {r['total_vcpu']:.0f} & "
                    f"{r['scale_events']:.0f} & — \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"  Saved: {tex}")

    # Table 2: K8s v2 summary
    tex = PLOTS_DIR / "table_k8s_v2.tex"
    with open(tex, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{Real Kubernetes Cluster Results (120 steps $\\times$ 60s). "
                "k3s cluster: 2 worker nodes, 3 vCPU / 16 GiB each.}\n")
        f.write("\\label{tab:k8s_v2}\n")
        f.write("\\begin{tabular}{lllrrrrr}\n\\toprule\n")
        f.write("Workload & Model & Variant & Replicas & Lat P90 & SLA\\% & Scales & Tokens \\\\\n")
        f.write("\\midrule\n")
        for wl in ["CPU", "IO"]:
            wl_data = ks[ks["workload_type"] == wl]
            core_k = wl_data[wl_data["model"].isin(CORE_MODELS)].sort_values(
                ["model", "variant"],
                key=lambda s: s.map({v: i for i, v in enumerate(CORE_MODELS)}) if s.name == "model"
                else s.map({v: i for i, v in enumerate(VARIANT_ORDER)}))
            base_k = wl_data[wl_data["model"].isin(BASELINES_LONG)]
            first = True
            prev = None
            for _, r in core_k.iterrows():
                if not first and r["model"] != prev:
                    f.write("\\cmidrule{2-8}\n")
                first = False
                prev = r["model"]
                wl_label = wl if r["model"] == core_k.iloc[0]["model"] and r["variant"] == core_k.iloc[0]["variant"] else ""
                sla_fmt = f"\\textbf{{{r['sla_pct']:.1f}}}" if r["sla_pct"] >= 99.5 else f"{r['sla_pct']:.1f}"
                tok = f"{r['total_tokens']:.0f}" if r['total_tokens'] > 0 else "—"
                f.write(f"{wl_label} & {_label(r['model'])} & {VARIANT_LABELS.get(r['variant'], r['variant'])} & "
                        f"{r['mean_replicas']:.1f} & {r['mean_latency']:.0f} & {sla_fmt} & "
                        f"{r['scale_events']:.0f} & {tok} \\\\\n")
            f.write("\\cmidrule{2-8}\n")
            for _, r in base_k.iterrows():
                sla_fmt = f"\\textbf{{{r['sla_pct']:.1f}}}" if r["sla_pct"] >= 99.5 else f"{r['sla_pct']:.1f}"
                f.write(f" & {_label(r['model'])} & — & "
                        f"{r['mean_replicas']:.1f} & {r['mean_latency']:.0f} & {sla_fmt} & "
                        f"{r['scale_events']:.0f} & — \\\\\n")
            if wl == "CPU":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"  Saved: {tex}")

    # Table 3: Prompt variant ablation
    tex = PLOTS_DIR / "table_ablation.tex"
    core = ls[ls["llm_model"].isin(CORE_MODELS)]
    with open(tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{Prompt Variant Ablation (averaged across core models, long simulation).}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Variant & Avg SLA\\% & Avg Latency & Avg Cost & Avg Scales \\\\\n")
        f.write("\\midrule\n")
        agg = core.groupby("llm_variant").agg(
            sla=("sla_pct", "mean"), lat=("mean_latency", "mean"),
            cost=("total_vcpu", "mean"), scales=("scale_events", "mean"),
        ).reindex([v for v in VARIANT_ORDER if v in core["llm_variant"].values])
        for v, r in agg.iterrows():
            sla_fmt = f"\\textbf{{{r['sla']:.1f}}}" if r["sla"] >= 99.5 else f"{r['sla']:.1f}"
            f.write(f"{VARIANT_LABELS.get(v, v)} & {sla_fmt} & {r['lat']:.1f} & "
                    f"{r['cost']:.0f} & {r['scales']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV Summaries
# ═══════════════════════════════════════════════════════════════════════════

def save_csv_summaries(ls, ks):
    ls.to_csv(PLOTS_DIR / "summary_long_sim.csv", index=False)
    ks.to_csv(PLOTS_DIR / "summary_k8s_v2.csv", index=False)
    print(f"  Saved: {PLOTS_DIR / 'summary_long_sim.csv'}")
    print(f"  Saved: {PLOTS_DIR / 'summary_k8s_v2.csv'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    dpi = args.dpi

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    long_df = load_long_sim()
    k8s_df = load_k8s_v2()
    ls = summarize_long(long_df)
    ks = summarize_k8s(k8s_df)
    print(f"  Long sim: {len(long_df)} rows, {long_df['llm_model'].nunique()} models")
    print(f"  K8s v2:   {len(k8s_df)} rows, {k8s_df['model'].nunique()} models")

    print("\nGenerating plots...")
    plot_workload_traces(dpi)
    plot_long_summary_bars(ls, dpi)
    plot_long_timeseries(long_df, dpi)
    plot_ablation_heatmap(ls, dpi)
    plot_model_size_scaling(ls, dpi)
    plot_cost_sla_pareto(ls, dpi)
    plot_k8s_summary_bars(ks, dpi)
    plot_k8s_workload_comparison(ks, dpi)
    plot_k8s_timeseries(k8s_df, dpi)
    plot_llm_overhead(ls, dpi)
    plot_sim_vs_real(ls, ks, dpi)
    plot_radar_profiles(ls, dpi)
    plot_variant_timeseries(long_df, dpi)
    plot_workload_difficulty(ks, dpi)
    plot_latency_distributions(long_df, dpi)

    print("\nGenerating LaTeX tables...")
    generate_latex_tables(ls, ks)

    print("\nSaving CSV summaries...")
    save_csv_summaries(ls, ks)

    print(f"\nAll done! {len(list(PLOTS_DIR.glob('*.png')))} plots, "
          f"{len(list(PLOTS_DIR.glob('*.tex')))} LaTeX tables, "
          f"{len(list(PLOTS_DIR.glob('*.csv')))} CSV summaries in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
