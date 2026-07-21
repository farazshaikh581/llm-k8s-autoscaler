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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VCPU_REQUEST = 0.25       # cpu: 250m request per replica (k8s/deployment.yaml, k8s/workloads.yaml)
VCPU_HOUR_USD = 0.04       # on-demand $/vCPU-hour (same basis as the sim analysis)
HOURS_PER_YEAR = 8760.0
SLA_MS = 200.0             # P90 latency SLA threshold (ms) — same as sim
SLA_DEPLOYABLE = 99.0
MIN_STEPS = 120            # only complete runs
OUT_DIR = "business_case_real"

BASELINE_MODELS = {"hpa": "HPA", "keda": "KEDA", "rl-dqn": "DQN", "rl-ppo": "PPO"}
CORE_LLM_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b", "llama4-scout", "gpt-oss-120b"]
DISPLAY = {"llama-8b": "Llama-8B", "llama-70b": "Llama-70B", "mistral-small4": "Mistral",
           "qwen3-80b": "Qwen-80B", "llama4-scout": "Scout", "gpt-oss-120b": "GPT-OSS-120B"}

# Okabe-Ito, colorblind-safe — same palette as plot_business_case.py (sim) for
# visual consistency; the REAL CLUSTER tag on every figure is what disambiguates.
COLORS = {"LLM": "#0072B2", "HPA": "#E69F00", "KEDA": "#D55E00", "DQN": "#009E73", "PPO": "#009E73"}
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"
SOURCE_TAG = "REAL K8S CLUSTER — measured, not simulated"
TAG_COLOR = "#CC3311"  # red-orange: visually distinct from the sim tag's blue
SCENARIO_LABELS = {"cpu_bursty": "CPU-bursty trace", "wiki_diurnal": "Wikipedia-diurnal trace"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "figure.dpi": 130,
})


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
        top = os.path.relpath(f, results_dir).split(os.sep)[0]
        if top.startswith("_"):
            continue  # archived/buggy runs, e.g. _buggy_*, _dropped_*
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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # average across reps (rep1/rep2/rep3 subdirs) so each config is one row,
    # matching this script's original single-rep assumption.
    agg = df.groupby(["workload", "model", "variant", "label", "cls"], as_index=False).agg(
        lat_sla=("lat_sla", "mean"), mean_lat=("mean_lat", "mean"),
        cost_vcpu_h=("cost_vcpu_h", "mean"), cost_usd=("cost_usd", "mean"),
        scales=("scales", "mean"), run_hours=("run_hours", "mean"), n_steps=("n_steps", "mean"),
        n_reps=("path", "count"),
    )
    for c in ("lat_sla", "mean_lat", "cost_vcpu_h", "scales"):
        agg[c] = agg[c].round(1)
    agg["cost_usd"] = agg["cost_usd"].round()
    return agg


def flat_label(row):
    return row.label if row.cls != "LLM" else f"{DISPLAY.get(row.model, row.model)}/{row.variant}"


def bar_label(row):
    return row.label if row.cls != "LLM" else f"{DISPLAY.get(row.model, row.model)}\n({row.variant})"


def pick_representative(s):
    """One config per core LLM model: cheapest with SLA >= threshold (else cheapest overall)."""
    reps = []
    for m in CORE_LLM_MODELS:
        sub = s[s.model == m]
        if sub.empty:
            continue
        ok = sub[sub.lat_sla >= SLA_DEPLOYABLE]
        pick = (ok if not ok.empty else sub).sort_values("cost_usd").iloc[0]
        reps.append(pick)
    bases = s[s.model.isin(BASELINE_MODELS)]
    if bases.empty and not reps:
        return s
    return pd.concat([pd.DataFrame(reps), bases]).reset_index(drop=True)


def _tag(fig):
    fig.text(0.01, 0.995, SOURCE_TAG, fontsize=9.5, fontweight="bold",
              color=TAG_COLOR, va="top", ha="left")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.png")


def fig_latency_cost(rep, source_label, suffix=""):
    rep = rep.sort_values("cost_usd").reset_index(drop=True)
    labels = [bar_label(r) for _, r in rep.iterrows()]
    colors = [COLORS.get(c, COLORS["LLM"]) for c in rep.cls]
    x = np.arange(len(rep))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    axL.bar(x, rep.lat_sla, color=colors, width=0.68, zorder=3)
    axL.set_ylim(0, 105)
    axL.set_ylabel("Latency SLA attainment (%)")
    axL.set_title(f"Robustness  —  % of time P90 < {SLA_MS:.0f} ms", fontsize=12, loc="left", pad=10)
    axL.axhline(99, color=MUTED, ls="--", lw=1, zorder=1)
    axL.text(-0.4, 99.5, "99% SLA", color=MUTED, fontsize=8, va="bottom", ha="left")
    for xi, v in zip(x, rep.lat_sla):
        axL.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    axR.bar(x, rep.cost_usd, color=colors, width=0.68, zorder=3)
    axR.set_ylabel("Annualized infra cost (USD / service)")
    axR.set_title(f"Cost  —  @ ${VCPU_HOUR_USD:.2f}/vCPU-hr, one service", fontsize=12, loc="left", pad=10)
    hpa_cost = rep[rep.model == "hpa"].cost_usd.values
    if hpa_cost.size:
        axR.axhline(hpa_cost[0], color=COLORS["HPA"], ls="--", lw=1, zorder=1)
        axR.text(len(rep) - 0.4, hpa_cost[0], "  HPA cost", color=COLORS["HPA"],
                 fontsize=8, va="bottom", ha="right")
    top = rep.cost_usd.max()
    for xi, v in zip(x, rep.cost_usd):
        axR.text(xi, v + top * 0.01, f"${v:,.0f}", ha="center", va="bottom", fontsize=8.5)
    axR.set_ylim(0, top * 1.14)

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    present = set(rep.cls)
    legend_order = [c for c in ["LLM", "HPA", "KEDA", "PPO", "DQN"] if c in present]
    legend_text = {"LLM": "LLM (frozen)", "HPA": "HPA", "KEDA": "KEDA", "PPO": "RL (PPO)", "DQN": "RL (DQN)"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in legend_order]
    fig.legend(handles, [legend_text[c] for c in legend_order],
               loc="lower center", ncol=len(legend_order), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Real cluster ({source_label}): LLM vs. rule-based autoscalers", fontsize=14, y=1.03)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    _tag(fig)
    _save(fig, f"fig_business_latency_cost_real{suffix}")


def fig_cost_stability(s, rep, source_label, suffix=""):
    core = s[(s.model.isin(CORE_LLM_MODELS) | s.model.isin(BASELINE_MODELS))]
    label_runs = set(rep.index)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    texts = []
    for _, r in core.iterrows():
        marker = "*" if r.cls == "LLM" else ("s" if r.cls in ("HPA", "KEDA") else "D")
        sz = 340 if marker == "*" else 150
        ax.scatter(r.cost_usd, r.scales, s=sz, c=COLORS.get(r.cls, COLORS["LLM"]), marker=marker,
                   edgecolors="white", linewidths=0.9, zorder=3, alpha=0.92)
        texts.append(ax.text(r.cost_usd, r.scales, flat_label(r), fontsize=8, color=INK))

    hpa_rows = rep[rep.model == "hpa"]
    if not hpa_rows.empty:
        hpa = hpa_rows.iloc[0]
        ax.axvline(hpa.cost_usd, color=COLORS["HPA"], ls="--", lw=1, zorder=1)
        texts.append(ax.text(hpa.cost_usd, ax.get_ylim()[1] * 0.02, " HPA cost", color=COLORS["HPA"],
                              fontsize=8, ha="left", va="bottom"))

    # Many configs land close together on cost and scale count, so plain offset
    # labels overlap into an unreadable block. adjustText spreads them out and
    # draws a thin leader line back to the point they belong to.
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    except ImportError:
        pass

    ax.set_xlabel("Annualized infra cost (USD / service)  → cheaper is left")
    ax.set_ylabel("Scaling actions over the run  → more stable is down")
    ax.set_title(f"Real cluster ({source_label}): cost vs. stability  —  good corner is bottom-left",
                 fontsize=12.5, loc="left")
    ax.set_ylim(-5, max(core.scales.max() * 1.35, 10))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=m, color="w", markerfacecolor=c,
                          markersize=12, label=l, markeredgecolor="gray")
               for m, c, l in [("*", COLORS["LLM"], "LLM (frozen)"), ("s", COLORS["HPA"], "HPA/KEDA"),
                                ("D", COLORS["PPO"], "RL")]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _tag(fig)
    _save(fig, f"fig_business_cost_stability_real{suffix}")


def fig_stability(rep, source_label, suffix=""):
    rep = rep.sort_values("scales").reset_index(drop=True)
    labels = [bar_label(r) for _, r in rep.iterrows()]
    colors = [COLORS.get(c, COLORS["LLM"]) for c in rep.cls]
    x = np.arange(len(rep))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, rep.scales, color=colors, width=0.68, zorder=3)
    for xi, v in zip(x, rep.scales):
        ax.text(xi, v + max(rep.scales.max(), 1) * 0.01, f"{v}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Scaling actions over the run")
    ax.set_title(f"Real cluster ({source_label}): operational stability  —  fewer scaling actions = less churn",
                 fontsize=13, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True)
    fig.tight_layout()
    _tag(fig)
    _save(fig, f"fig_business_stability_real{suffix}")


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
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation, table + verdict only")
    ap.add_argument("--tag", default=None,
                     help="figure filename suffix (default: workload); set this when "
                          "--results-dir points at one scenario, since the workload token "
                          "in these filenames is always 'cpu' for both cpu_bursty and wiki_diurnal")
    args = ap.parse_args()

    s = load_summary(args.results_dir, args.workload or None)
    print(f"Loaded {len(s)} complete real-cluster runs (>= {MIN_STEPS} steps) "
          f"from {args.results_dir}/ (workload={args.workload or 'all'})")
    if s.empty:
        print("No complete runs found.")
        return
    out = args.out or f"{args.results_dir.rstrip('/')}_business_case.csv"
    s = s.sort_values("cost_usd")
    s.to_csv(out, index=False)
    print(f"Summary table -> {out}")
    two_regime_verdict(s)

    if not args.no_plots:
        tag = args.tag or args.workload
        source_label = SCENARIO_LABELS.get(tag, f"{args.results_dir}, {args.workload or 'all'} workload")
        suffix = f"_{tag}" if tag else ""
        rep = pick_representative(s)
        fig_latency_cost(rep, source_label, suffix)
        fig_cost_stability(s, rep, source_label, suffix)
        fig_stability(rep, source_label, suffix)


if __name__ == "__main__":
    main()
