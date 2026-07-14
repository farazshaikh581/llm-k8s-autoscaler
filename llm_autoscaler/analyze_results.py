#!/usr/bin/env python3
"""Analyze and plot LLM autoscaler experiment results (v2 — infra-aware).

Produces:
  1. Summary table (terminal + LaTeX) — mean metrics per model×variant
  2. Time-series plot — replicas + latency + RPS over trace steps
  3. Prompt ablation heatmap — metric × variant × model
  4. Model size scaling plot — performance vs parameter count
  5. Cost-efficiency scatter — SLA compliance vs vCPU cost
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

MODEL_SIZES = {
    "gpt-oss-120b": 120,
    "llama-70b": 70,
    "qwen3-32b": 32,
    "llama4-scout": 17,
    "llama-8b": 8,
    "hpa": 0,
    "keda": 0,
}

MODEL_LABELS = {
    "gpt-oss-120b": "GPT-OSS 120B",
    "llama-70b": "Llama 3.3 70B",
    "qwen3-32b": "Qwen 3 32B",
    "llama4-scout": "Llama 4 Scout 17B",
    "llama-8b": "Llama 3.1 8B",
    "hpa": "HPA (CPU 50%)",
    "keda": "KEDA (RPS)",
}

MODEL_ORDER = ["gpt-oss-120b", "llama-70b", "qwen3-32b", "llama4-scout", "llama-8b"]
VARIANT_ORDER = ["zero_shot", "history_5", "cot", "domain"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    frames = []
    for f in sorted(RESULTS_DIR.glob("results_*.csv")):
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        print("No result CSVs found in", RESULTS_DIR)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["llm_model", "llm_variant"])
    summary = groups.agg(
        mean_ready=("ready_replicas", "mean"),
        mean_target=("replicas", "mean"),
        mean_latency=("latency_p90", "mean"),
        p99_latency=("latency_p90", lambda x: x.quantile(0.99)),
        mean_cpu=("cpu_pct", "mean"),
        mean_success=("success_rate", "mean"),
        sla_violations=("latency_p90", lambda x: (x > 200).sum()),
        total_vcpu_min=("vcpu_minutes", "last"),
        scale_events=("scale_event", "sum"),
        total_tokens=("llm_tokens_used", "sum"),
        mean_llm_latency=("llm_latency_ms", "mean"),
        steps=("step", "count"),
    ).reset_index()

    summary["sla_pct"] = round(
        (1 - summary["sla_violations"] / summary["steps"]) * 100, 1
    )

    optimal_replicas = df.groupby(["llm_model", "llm_variant"]).apply(
        lambda g: np.ceil(g["requests"].values / 200).mean()
    ).values
    summary["over_provision"] = round(summary["mean_ready"] - optimal_replicas, 1)

    return summary

# ---------------------------------------------------------------------------
# Plot 1: Summary table
# ---------------------------------------------------------------------------

def print_summary_table(summary: pd.DataFrame):
    print("\n" + "=" * 110)
    print("SUMMARY — Mean metrics per model × variant")
    print("=" * 110)
    print(f"{'Model':<20s} {'Variant':<12s} {'Ready':>5s} {'LatP90':>7s} {'CPU%':>5s} "
          f"{'Succ':>6s} {'SLA%':>5s} {'vCPU-m':>7s} {'Scales':>6s} {'OverProv':>8s}")
    print("-" * 110)
    for _, r in summary.iterrows():
        label = MODEL_LABELS.get(r["llm_model"], r["llm_model"])[:19]
        print(f"{label:<20s} {r['llm_variant']:<12s} {r['mean_ready']:5.1f} "
              f"{r['mean_latency']:7.1f} {r['mean_cpu']:5.1f} "
              f"{r['mean_success']:6.4f} {r['sla_pct']:5.1f} "
              f"{r['total_vcpu_min']:7.0f} {r['scale_events']:6.0f} "
              f"{r['over_provision']:+8.1f}")

    # LaTeX
    tex_file = PLOTS_DIR / "summary_table.tex"
    with open(tex_file, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Autoscaling performance on Alibaba Cluster Trace 2018 (300 steps). "
                "Cluster: 5 nodes $\\times$ 4\\,vCPU; pods: 500m CPU; "
                "SLA: P90 latency $<$200\\,ms.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        f.write("Model & Variant & Ready & Lat P90 & CPU\\% & SLA\\% & "
                "vCPU-min & Scales & Over-prov \\\\\n")
        f.write("\\midrule\n")
        prev_model = None
        for _, r in summary.iterrows():
            label = MODEL_LABELS.get(r["llm_model"], r["llm_model"])
            if prev_model and r["llm_model"] != prev_model:
                f.write("\\midrule\n")
            prev_model = r["llm_model"]
            f.write(
                f"{label} & {r['llm_variant']} & "
                f"{r['mean_ready']:.1f} & {r['mean_latency']:.0f} & "
                f"{r['mean_cpu']:.0f} & {r['sla_pct']:.0f} & "
                f"{r['total_vcpu_min']:.0f} & {r['scale_events']:.0f} & "
                f"{r['over_provision']:+.1f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"\nLaTeX table: {tex_file}")

# ---------------------------------------------------------------------------
# Plot 2: Time-series
# ---------------------------------------------------------------------------

def plot_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    baselines = df[df["llm_variant"] == "baseline"]
    llm_best = df[df["llm_variant"] == "zero_shot"]

    # RPS trace (top panel)
    sample = baselines.groupby("llm_model").first()
    rps_source = baselines[baselines["llm_model"] == baselines["llm_model"].iloc[0]].sort_values("step")
    axes[0].fill_between(rps_source["step"], rps_source["requests"], alpha=0.3, color="gray")
    axes[0].plot(rps_source["step"], rps_source["requests"], "gray", alpha=0.5, linewidth=0.8)
    axes[0].set_ylabel("Requests/min")
    axes[0].set_title("Workload Trace (Alibaba 2018) + Autoscaler Response")

    # Replicas (middle panel)
    optimal = np.ceil(rps_source["requests"].values / 200).astype(int).clip(1, 20)
    axes[1].plot(rps_source["step"].values, optimal, "k:", alpha=0.4, label="Optimal", linewidth=1)

    for model, grp in baselines.groupby("llm_model"):
        grp = grp.sort_values("step")
        axes[1].plot(grp["step"], grp["ready_replicas"], "--",
                     label=MODEL_LABELS.get(model, model), alpha=0.8, linewidth=1.5)

    for model, grp in llm_best.groupby("llm_model"):
        grp = grp.sort_values("step")
        axes[1].plot(grp["step"], grp["ready_replicas"], "-",
                     label=MODEL_LABELS.get(model, model), alpha=0.7, linewidth=1)

    axes[1].set_ylabel("Ready Replicas")
    axes[1].legend(fontsize=6, ncol=4, loc="upper right")
    axes[1].set_ylim(0, 21)

    # Latency (bottom panel)
    axes[2].axhline(y=200, color="red", linestyle=":", alpha=0.5, label="SLA (200ms)")

    for model, grp in baselines.groupby("llm_model"):
        grp = grp.sort_values("step")
        axes[2].plot(grp["step"], grp["latency_p90"], "--",
                     label=MODEL_LABELS.get(model, model), alpha=0.8, linewidth=1.5)

    for model, grp in llm_best.groupby("llm_model"):
        grp = grp.sort_values("step")
        axes[2].plot(grp["step"], grp["latency_p90"], "-",
                     label=MODEL_LABELS.get(model, model), alpha=0.7, linewidth=1)

    axes[2].set_ylabel("Latency P90 (ms)")
    axes[2].set_xlabel("Step (minutes)")
    axes[2].legend(fontsize=6, ncol=4, loc="upper right")
    axes[2].set_yscale("log")

    plt.tight_layout()
    out = PLOTS_DIR / "timeseries.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 3: Ablation heatmap
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(summary: pd.DataFrame):
    llm = summary[~summary["llm_model"].isin(["hpa", "keda"])]
    if llm.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [
        ("mean_latency", "Latency P90 (ms)", "Reds"),
        ("sla_pct", "SLA Compliance %", "Greens"),
        ("total_vcpu_min", "Cost (vCPU-min)", "Oranges"),
        ("over_provision", "Over-provisioning", "RdYlGn_r"),
    ]

    models = [m for m in MODEL_ORDER if m in llm["llm_model"].values]
    variants = [v for v in VARIANT_ORDER if v in llm["llm_variant"].values]

    for ax, (metric, title, cmap) in zip(axes, metrics):
        pivot = llm.pivot(index="llm_model", columns="llm_variant", values=metric)
        pivot = pivot.reindex(index=models, columns=variants)
        pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(pivot)):
            for j in range(len(variants)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.0f}" if abs(val) > 10 else f"{val:.1f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Prompt Variant Ablation (5 models × 4 variants)", fontsize=12)
    plt.tight_layout()
    out = PLOTS_DIR / "ablation_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 4: Model size scaling
# ---------------------------------------------------------------------------

def plot_size_scaling(summary: pd.DataFrame):
    llm_zs = summary[
        (~summary["llm_model"].isin(["hpa", "keda"]))
        & (summary["llm_variant"] == "zero_shot")
    ].copy()
    if llm_zs.empty:
        return

    llm_zs["size_b"] = llm_zs["llm_model"].map(MODEL_SIZES)
    llm_zs = llm_zs.sort_values("size_b")

    # get baseline values for reference lines
    baselines = summary[summary["llm_model"].isin(["hpa", "keda"])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("mean_latency", "Mean Latency P90 (ms)"),
        ("sla_pct", "SLA Compliance (%)"),
        ("total_vcpu_min", "Cost (vCPU-min)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(llm_zs["size_b"], llm_zs[metric], "o-", markersize=8, color="tab:blue",
                label="LLM (zero_shot)")
        for _, row in llm_zs.iterrows():
            ax.annotate(
                MODEL_LABELS[row["llm_model"]].split()[0],
                (row["size_b"], row[metric]),
                textcoords="offset points", xytext=(0, 10), fontsize=7, ha="center",
            )
        for _, br in baselines.iterrows():
            ax.axhline(y=br[metric], linestyle="--", alpha=0.5,
                       label=MODEL_LABELS.get(br["llm_model"], br["llm_model"]))
        ax.set_xlabel("Model Size (B params)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Zero-Shot Autoscaling Quality vs Model Size", fontsize=12)
    plt.tight_layout()
    out = PLOTS_DIR / "model_size_scaling.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 5: Cost vs SLA (Pareto)
# ---------------------------------------------------------------------------

def plot_cost_vs_sla(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in summary.iterrows():
        label = MODEL_LABELS.get(row["llm_model"], row["llm_model"])
        variant = row["llm_variant"]
        is_baseline = row["llm_model"] in ("hpa", "keda")
        marker = "s" if is_baseline else "o"
        size = 100 if is_baseline else 60

        ax.scatter(row["total_vcpu_min"], row["sla_pct"],
                   s=size, marker=marker, alpha=0.7, zorder=3)
        ax.annotate(f"{label}\n({variant})",
                    (row["total_vcpu_min"], row["sla_pct"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=6)

    ax.axhline(y=99, color="green", linestyle=":", alpha=0.5, label="99% SLA target")
    ax.set_xlabel("Cost (vCPU-minutes)")
    ax.set_ylabel("SLA Compliance (%)")
    ax.set_title("Cost-Efficiency Frontier: LLM Autoscalers vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "cost_vs_sla.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    df = load_all_results()
    models = df["llm_model"].nunique()
    variants = df["llm_variant"].nunique()
    print(f"Loaded {len(df)} rows: {models} models, {variants} variants")

    summary = summarize(df)
    print_summary_table(summary)

    summary.to_csv(PLOTS_DIR / "summary.csv", index=False)
    print(f"Summary CSV: {PLOTS_DIR / 'summary.csv'}")

    print("\nGenerating plots...")
    plot_timeseries(df)
    plot_ablation_heatmap(summary)
    plot_size_scaling(summary)
    plot_cost_vs_sla(summary)

    print("\nAll done.")


if __name__ == "__main__":
    main()
