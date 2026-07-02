#!/usr/bin/env python3
"""Plot 16: Inference time comparison — all models + baselines, with 6G context."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
LONG_DIR = BASE / "results" / "long_sim"
K8S_DIR = BASE / "results" / "k8s_v2"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALL_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b",
              "llama4-scout", "gpt-oss-120b"]
BASELINES = ["dqn", "ppo", "hpa", "keda"]
VARIANT_ORDER = ["zero_shot", "domain", "history_5", "cot"]

MODEL_LABELS = {
    "llama-8b": "Llama 3.1\n8B", "llama-70b": "Llama 3.3\n70B",
    "mistral-small4": "Mistral\nSmall 4", "qwen3-80b": "Qwen 3\n80B",
    "gpt-oss-120b": "GPT-OSS\n120B", "llama4-scout": "Scout\n17B",
    "hpa": "HPA", "keda": "KEDA", "dqn": "DQN\n(RL)", "ppo": "PPO\n(RL)",
}
VARIANT_LABELS = {
    "zero_shot": "Zero-Shot", "domain": "Domain",
    "history_5": "History-5", "cot": "CoT",
}
VARIANT_COLORS = {
    "zero_shot": "#2196F3", "domain": "#4CAF50",
    "history_5": "#FF9800", "cot": "#E91E63",
}
BASELINE_COLOR = "#607D8B"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
})


def load_long_sim():
    frames = []
    for f in sorted(LONG_DIR.glob("results_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) < 1440:
            continue
        df = df.head(1440)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_stats(raw):
    rows = []
    for (model, variant), g in raw.groupby(["llm_model", "llm_variant"]):
        lats = g["llm_latency_ms"]
        if model not in BASELINES:
            lats = lats[lats > 0]
        rows.append({
            "model": model, "variant": variant,
            "mean_ms": lats.mean(),
            "median_ms": lats.median(),
            "p5_ms": lats.quantile(0.05),
            "p95_ms": lats.quantile(0.95),
            "p99_ms": lats.quantile(0.99),
            "min_ms": lats.min(),
            "max_ms": lats.max(),
            "std_ms": lats.std(),
        })
    return pd.DataFrame(rows)


def plot_inference_bars(stats, dpi=300):
    fig, (ax_llm, ax_base) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [4, 1.2], "wspace": 0.08},
    )

    # --- Left panel: LLM models grouped by variant ---
    models = [m for m in ALL_MODELS if m in stats["model"].values]
    x = np.arange(len(models))
    width = 0.18

    for vi, v in enumerate(VARIANT_ORDER):
        means, lo_err, hi_err = [], [], []
        for m in models:
            row = stats[(stats["model"] == m) & (stats["variant"] == v)]
            if len(row) > 0:
                mean = row["mean_ms"].values[0]
                p5 = row["p5_ms"].values[0]
                p95 = row["p95_ms"].values[0]
            else:
                mean = p5 = p95 = 0
            means.append(mean)
            lo_err.append(max(0, mean - p5))
            hi_err.append(max(0, p95 - mean))
        offset = (vi - len(VARIANT_ORDER) / 2 + 0.5) * width
        ax_llm.bar(x + offset, means, width,
                   label=VARIANT_LABELS[v], color=VARIANT_COLORS[v],
                   alpha=0.85, edgecolor="white", linewidth=0.5,
                   yerr=[lo_err, hi_err], capsize=2, error_kw={"lw": 0.8})

    # 6G requirement zones
    ax_llm.axhspan(0, 100, alpha=0.08, color="green", zorder=0)
    ax_llm.axhspan(100, 1000, alpha=0.06, color="orange", zorder=0)
    ax_llm.axhspan(1000, 50000, alpha=0.04, color="red", zorder=0)

    ax_llm.axhline(100, color="green", ls="--", lw=1.2, alpha=0.7)
    ax_llm.axhline(1000, color="orange", ls="--", lw=1.2, alpha=0.7)
    ax_llm.axhline(10000, color="red", ls="--", lw=1.0, alpha=0.5)

    ax_llm.text(len(models) - 0.3, 70, "Edge-viable (<100ms)", fontsize=7,
                color="green", ha="right", fontstyle="italic")
    ax_llm.text(len(models) - 0.3, 700, "Cloud-viable (<1s)", fontsize=7,
                color="#CC7700", ha="right", fontstyle="italic")
    ax_llm.text(len(models) - 0.3, 7000, "Too slow for real-time", fontsize=7,
                color="red", ha="right", fontstyle="italic")

    ax_llm.set_yscale("log")
    ax_llm.set_xticks(x)
    ax_llm.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=0, ha="center")
    ax_llm.set_ylabel("Inference Latency (ms, log scale)")
    ax_llm.set_title("LLM Autoscaler Models")
    ax_llm.legend(loc="upper left", framealpha=0.9)
    ax_llm.grid(True, alpha=0.15, axis="y", which="both")
    ax_llm.set_ylim(50, 50000)

    # --- Right panel: Baselines ---
    baselines = [b for b in BASELINES if b in stats["model"].values]
    xb = np.arange(len(baselines))
    baseline_vals = []
    for b in baselines:
        row = stats[stats["model"] == b]
        baseline_vals.append(max(row["mean_ms"].values[0], 0.01))

    bars = ax_base.bar(xb, baseline_vals, 0.5, color=BASELINE_COLOR, alpha=0.85,
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, baseline_vals):
        if val < 1:
            ax_base.text(bar.get_x() + bar.get_width() / 2, 0.15,
                         "<0.01ms", ha="center", va="bottom", fontsize=7,
                         fontweight="bold", color="white")

    ax_base.axhspan(0, 100, alpha=0.08, color="green", zorder=0)
    ax_base.axhline(100, color="green", ls="--", lw=1.2, alpha=0.7)

    ax_base.set_yscale("log")
    ax_base.set_xticks(xb)
    ax_base.set_xticklabels([MODEL_LABELS.get(b, b) for b in baselines],
                            rotation=0, ha="center")
    ax_base.set_title("Baselines (RL + Rule)")
    ax_base.set_ylim(0.001, 50000)
    ax_base.set_yticklabels([])
    ax_base.grid(True, alpha=0.15, axis="y", which="both")

    fig.suptitle("Decision Inference Latency — All Autoscaling Approaches",
                 fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "16_inference_time_comparison.png",
                dpi=dpi, bbox_inches="tight")
    print(f"Saved: {PLOTS_DIR / '16_inference_time_comparison.png'}")
    plt.close(fig)


def plot_inference_vs_interval(stats, dpi=300):
    """Scatter: inference time vs decision interval budget, with feasibility zones."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {
        "llama-8b": "#2196F3", "llama-70b": "#FF9800", "mistral-small4": "#4CAF50",
        "qwen3-80b": "#E91E63", "gpt-oss-120b": "#9C27B0", "llama4-scout": "#00BCD4",
    }
    variant_markers = {"zero_shot": "o", "domain": "s", "history_5": "^", "cot": "D"}

    intervals = [1, 5, 10, 30, 60]
    ax.axhspan(0, 1, alpha=0.08, color="green", zorder=0)
    ax.axhline(1.0, color="red", ls="-", lw=2, alpha=0.8)
    ax.text(0.55, 1.05, "Infeasible (inference > interval)", fontsize=9,
            color="red", transform=ax.transAxes, fontstyle="italic")

    for _, row in stats.iterrows():
        if row["model"] in BASELINES:
            continue
        for interval_s in intervals:
            ratio = (row["mean_ms"] / 1000) / interval_s
            color = model_colors.get(row["model"], "gray")
            marker = variant_markers.get(row["variant"], "x")
            label_m = MODEL_LABELS.get(row["model"], row["model"]).replace("\n", " ")
            label_v = VARIANT_LABELS.get(row["variant"], row["variant"])
            ax.scatter(interval_s, ratio, c=color, marker=marker, s=50, alpha=0.7,
                       edgecolors="black", linewidths=0.3)

    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                            markersize=8, label=MODEL_LABELS[m].replace("\n", " "))
                     for m, c in model_colors.items()]
    variant_handles = [Line2D([0], [0], marker=variant_markers[v], color="w",
                              markerfacecolor="gray", markersize=8,
                              label=VARIANT_LABELS[v])
                       for v in VARIANT_ORDER]

    leg1 = ax.legend(handles=model_handles, loc="upper right", title="Model",
                     fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, loc="center right", title="Variant",
              fontsize=7, title_fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(intervals)
    ax.set_xticklabels([f"{i}s" for i in intervals])
    ax.set_xlabel("Decision Interval")
    ax.set_ylabel("Inference Time / Interval Ratio")
    ax.set_title("Inference Feasibility Across Decision Intervals", fontweight="bold")
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "17_inference_feasibility.png",
                dpi=dpi, bbox_inches="tight")
    print(f"Saved: {PLOTS_DIR / '17_inference_feasibility.png'}")
    plt.close(fig)


def generate_latex_table(stats):
    tex = PLOTS_DIR / "table_inference_times.tex"
    with open(tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Decision inference latency across autoscaling approaches. "
                "LLM values are per-step means over 1440 steps (Alibaba trace). "
                "RL and rule-based baselines have sub-millisecond latency. "
                "Edge-viable threshold: $<$100\\,ms; cloud-viable: $<$1\\,s.}\n")
        f.write("\\label{tab:inference-times}\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Model & Variant & Mean (ms) & Median (ms) & P95 (ms) & P99 (ms) \\\\\n")
        f.write("\\midrule\n")

        prev = None
        for m in ALL_MODELS:
            for v in VARIANT_ORDER:
                row = stats[(stats["model"] == m) & (stats["variant"] == v)]
                if row.empty:
                    continue
                r = row.iloc[0]
                label = MODEL_LABELS.get(m, m).replace("\n", " ")
                if prev and m != prev:
                    f.write("\\midrule\n")
                prev = m
                mean = r["mean_ms"]
                if mean > 1000:
                    mean_s = f"{mean/1000:.1f}s"
                else:
                    mean_s = f"{mean:.0f}"
                f.write(f"{label} & {VARIANT_LABELS.get(v, v)} & "
                        f"{r['mean_ms']:.0f} & {r['median_ms']:.0f} & "
                        f"{r['p95_ms']:.0f} & {r['p99_ms']:.0f} \\\\\n")

        f.write("\\midrule\n")
        for b in BASELINES:
            row = stats[stats["model"] == b]
            if row.empty:
                continue
            label = MODEL_LABELS.get(b, b).replace("\n", " ")
            f.write(f"{label} & --- & $<$1 & $<$1 & $<$1 & $<$1 \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {tex}")


def print_summary(stats):
    print("\n" + "=" * 90)
    print("INFERENCE TIME SUMMARY (Long Simulation, 1440 steps)")
    print("=" * 90)
    print(f"{'Model':<22s} {'Variant':<12s} {'Mean':>8s} {'Median':>8s} "
          f"{'P95':>8s} {'P99':>8s} {'Max':>8s}")
    print("-" * 90)
    for m in ALL_MODELS + BASELINES:
        for v in VARIANT_ORDER + ["rl", "baseline"]:
            row = stats[(stats["model"] == m) & (stats["variant"] == v)]
            if row.empty:
                continue
            r = row.iloc[0]
            label = MODEL_LABELS.get(m, m).replace("\n", " ")
            def fmt(x):
                if x < 1:
                    return "<1ms"
                elif x > 1000:
                    return f"{x/1000:.1f}s"
                else:
                    return f"{x:.0f}ms"
            print(f"{label:<22s} {VARIANT_LABELS.get(v, v):<12s} "
                  f"{fmt(r['mean_ms']):>8s} {fmt(r['median_ms']):>8s} "
                  f"{fmt(r['p95_ms']):>8s} {fmt(r['p99_ms']):>8s} "
                  f"{fmt(r['max_ms']):>8s}")


if __name__ == "__main__":
    raw = load_long_sim()
    stats = compute_stats(raw)
    print_summary(stats)
    plot_inference_bars(stats, dpi=300)
    plot_inference_vs_interval(stats, dpi=300)
    generate_latex_table(stats)
