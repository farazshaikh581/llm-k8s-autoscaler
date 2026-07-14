#!/usr/bin/env python3
"""Business-case analysis: latency robustness vs. cost for LLM vs. RL/HPA/KEDA autoscalers.

Reframes cost as a first-class objective alongside the latency SLA, on the hardened
simulator (results_long_v2). Produces:
  - fig_business_latency_cost.png : two panels (latency SLA | annualized cost)
  - fig_business_pareto.png       : cost($)-vs-SLA frontier, stability encoded
  - fig_business_stability.png    : scaling actions (operational churn)
  - business_case_summary.csv     : the underlying table
and prints the two-regime verdict.

Cost model (identical for every controller in results_long_v2):
  0.25 vCPU per ready replica per step, 1 step = 1 minute, 1440 steps = 24 h.
Dollar figures assume an on-demand rate of $/vCPU-hour below (state it in the paper),
annualized from the 24 h run.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results_long_v2"
OUT_DIR = "business_case"
SLA_MS = 200.0                 # P90 latency SLA threshold (ms)
VCPU_HOUR_USD = 0.04           # on-demand $/vCPU-hour (AWS general-purpose, mid-range)
RUN_HOURS = 24.0               # 1440 steps x 1 min
HOURS_PER_YEAR = 8760.0
MIN_STEPS = 1440               # only complete runs (partial runs understate cost)
SLA_DEPLOYABLE = 99.0          # min latency-SLA to be "business deployable"

# Okabe-Ito, colorblind-safe. Color by controller CLASS (identity), fixed order.
COLORS = {"LLM": "#0072B2", "HPA": "#E69F00", "KEDA": "#D55E00", "RL": "#009E73"}
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"
SOURCE_TAG = "SIMULATED — hardened simulator, 24h Alibaba trace (not real cluster)"
TAG_COLOR = "#0072B2"  # blue: visually distinct from the real-cluster tag's red-orange

CORE_MODELS = ["llama-8b", "llama-70b", "mistral-small4", "qwen3-80b"]
BASELINES = {"hpa_baseline": "HPA", "keda_baseline": "KEDA",
             "dqn_rl": "DQN", "ppo_rl": "PPO"}
DISPLAY = {"llama-8b": "Llama-8B", "llama-70b": "Llama-70B",
           "mistral-small4": "Mistral", "qwen3-80b": "Qwen-80B", "llama4-scout": "Scout"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "figure.dpi": 130,
})


def annual_usd(vcpu_min):
    """vCPU-minutes over the 24 h run -> annualized on-demand dollars."""
    run_cost = (vcpu_min / 60.0) * VCPU_HOUR_USD
    return run_cost * (HOURS_PER_YEAR / RUN_HOURS)


def load_summary():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))):
        name = os.path.basename(f)[len("results_"):-4]
        if name.endswith("_indist"):
            continue
        df = pd.read_csv(f)
        if len(df) < MIN_STEPS or "latency_p90" not in df:
            continue
        lat_sla = (df["latency_p90"] <= SLA_MS).mean() * 100
        cost = df["vcpu_minutes"].iloc[-1]
        scales = int(df["scale_event"].sum())
        # classify + split model/variant
        if name in BASELINES:
            label, cls, model, variant = BASELINES[name], \
                ("RL" if name.endswith("_rl") else BASELINES[name]), name, ""
            cls = "RL" if name in ("dqn_rl", "ppo_rl") else BASELINES[name]
            model = name
        else:
            model, _, variant = name.partition("_")
            label, cls = name, "LLM"
        rows.append(dict(run=name, label=label, cls=cls, model=model, variant=variant,
                         lat_sla=round(lat_sla, 1), mean_lat=round(df["latency_p90"].mean(), 1),
                         cost_vcpu=round(cost), cost_usd=round(annual_usd(cost)),
                         scales=scales))
    return pd.DataFrame(rows)


def pick_representative(s):
    """One deployable config per core LLM model: cheapest with SLA >= threshold."""
    reps = []
    for m in CORE_MODELS:
        sub = s[s.model == m]
        if sub.empty:
            continue
        ok = sub[sub.lat_sla >= SLA_DEPLOYABLE]
        pick = (ok if not ok.empty else sub).sort_values("cost_vcpu").iloc[0]
        reps.append(pick)
    bases = s[s.model.isin(BASELINES)]
    return pd.concat([pd.DataFrame(reps), bases]).reset_index(drop=True)


def bar_label(row):
    if row.cls == "LLM":
        return f"{DISPLAY.get(row.model, row.model)}\n({row.variant})"
    return {"hpa_baseline": "HPA", "keda_baseline": "KEDA",
            "dqn_rl": "DQN", "ppo_rl": "PPO"}[row.model]


def flat_label(row):
    return bar_label(row).replace("\n", " ")


def fig_latency_cost(rep):
    rep = rep.sort_values("cost_usd").reset_index(drop=True)
    labels = [bar_label(r) for _, r in rep.iterrows()]
    colors = [COLORS[c] for c in rep.cls]
    x = np.arange(len(rep))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: latency SLA ---
    axL.bar(x, rep.lat_sla, color=colors, width=0.68, zorder=3)
    axL.set_ylim(90, 100.6)
    axL.set_ylabel("Latency SLA attainment (%)")
    axL.set_title("Robustness  —  % of time P90 < 200 ms", fontsize=12, loc="left", pad=10)
    axL.axhline(99, color=MUTED, ls="--", lw=1, zorder=1)
    axL.text(len(rep) - 0.4, 99.05, "99% SLA", color=MUTED, fontsize=8, va="bottom", ha="right")
    for xi, v in zip(x, rep.lat_sla):
        axL.text(xi, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    # --- Right: annualized cost ---
    axR.bar(x, rep.cost_usd, color=colors, width=0.68, zorder=3)
    axR.set_ylabel("Annualized infra cost (USD / service)")
    axR.set_title(f"Cost  —  @ ${VCPU_HOUR_USD:.2f}/vCPU-hr, one service", fontsize=12, loc="left", pad=10)
    hpa_cost = rep[rep.model == "hpa_baseline"].cost_usd.values
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

    # class legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in ["LLM", "HPA", "KEDA", "RL"]]
    fig.legend(handles, ["LLM (frozen)", "HPA", "KEDA", "RL (DQN/PPO)"],
               loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LLM autoscalers match SLA robustness at comparable cost", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    _tag(fig)
    _save(fig, "fig_business_latency_cost")


def fig_cost_stability(s, rep):
    """Cost vs. stability at matched SLA — the two axes that actually vary.
    All core LLM configs + baselines are near-identical on SLA (>=98.8%), so the
    business tradeoff lives in the cost / scaling-churn plane. Good corner = bottom-left."""
    core = s[(s.model.isin(CORE_MODELS) | s.model.isin(BASELINES)) & (s.lat_sla >= 99.0)]
    label_runs = set(rep.run) | set(BASELINES)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for _, r in core.iterrows():
        marker = "*" if r.cls == "LLM" else ("s" if r.cls in ("HPA", "KEDA") else "D")
        sz = 340 if marker == "*" else 150
        ax.scatter(r.cost_usd, r.scales, s=sz, c=COLORS[r.cls], marker=marker,
                   edgecolors="white", linewidths=0.9, zorder=3, alpha=0.92)
        # label representatives + baselines, plus any high-churn LLM thrasher (teaches the outlier)
        thrasher = r.cls == "LLM" and r.scales > 400
        if r.run == "keda_baseline":
            continue  # merged into the HPA label to avoid overprint
        if r.run in label_runs or thrasher:
            lab = ({"hpa_baseline": "HPA/KEDA", "dqn_rl": "DQN", "ppo_rl": "PPO"}[r.model]
                   if r.cls != "LLM" else f"{DISPLAY.get(r.model, r.model)}/{r.variant}")
            dy = 6 if thrasher else 4
            ax.annotate(lab, (r.cost_usd, r.scales), fontsize=8,
                        color=(COLORS["HPA"] if thrasher else INK),
                        xytext=(7, dy), textcoords="offset points")

    # shade the "business sweet spot": cost <= HPA and churn well below HPA
    hpa = rep[rep.model == "hpa_baseline"].iloc[0]
    ax.axvspan(0, hpa.cost_usd, ymin=0, ymax=0.35, color=COLORS["LLM"], alpha=0.06, zorder=0)
    ax.axvline(hpa.cost_usd, color=COLORS["HPA"], ls="--", lw=1, zorder=1)
    ax.text(hpa.cost_usd, ax.get_ylim()[1] * 0.02, " HPA cost", color=COLORS["HPA"],
            fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Annualized infra cost (USD / service)  → cheaper is left")
    ax.set_ylabel("Scaling actions over 24 h  → more stable is down")
    ax.set_title("Cost vs. stability at matched SLA (all ≥ 99%)  —  good corner is bottom-left",
                 fontsize=12.5, loc="left")
    ax.set_ylim(-15, core.scales.max() * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=m, color="w", markerfacecolor=COLORS[c],
                          markersize=12, label=l, markeredgecolor="gray")
               for m, c, l in [("*", "LLM", "LLM (frozen)"), ("s", "HPA", "HPA/KEDA"), ("D", "RL", "RL")]]
    ax.legend(handles=handles, loc="center right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    _tag(fig)
    _save(fig, "fig_business_cost_stability")


def fig_stability(rep):
    rep = rep.sort_values("scales").reset_index(drop=True)
    labels = [bar_label(r) for _, r in rep.iterrows()]
    colors = [COLORS[c] for c in rep.cls]
    x = np.arange(len(rep))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, rep.scales, color=colors, width=0.68, zorder=3)
    for xi, v in zip(x, rep.scales):
        ax.text(xi, v + rep.scales.max() * 0.01, f"{v}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Scaling actions over 24 h")
    ax.set_title("Operational stability  —  fewer scaling actions = less churn / risk",
                 fontsize=13, loc="left")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True)
    fig.tight_layout()
    _tag(fig)
    _save(fig, "fig_business_stability")


def _row(r):
    return f"     {flat_label(r):26s} SLA {r.lat_sla:5.1f}%  ${r.cost_usd:>6,}/yr  {r.scales:>4} scales"


def two_regime_verdict(s, rep):
    lines = ["\n" + "=" * 78, "TWO BUSINESS REGIMES", "=" * 78]
    hpa = rep[rep.model == "hpa_baseline"].iloc[0]

    # Regime A: latency-sensitive product -> min cost s.t. SLA >= 99.5%
    a_pool = rep[rep.lat_sla >= 99.5].sort_values("cost_usd")
    lines.append("\nA) PERFORMANCE-CRITICAL  (latency-sensitive product)")
    lines.append("   rule: minimize cost subject to latency-SLA >= 99.5%")
    if not a_pool.empty:
        w = a_pool.iloc[0]
        pct = (hpa.cost_usd - w.cost_usd) / hpa.cost_usd * 100
        cheaper = f"{abs(pct):.0f}% {'cheaper than' if pct >= 0 else 'pricier than'} HPA"
        lines.append(f"   -> WINNER: {flat_label(w)}  (SLA {w.lat_sla}%, ${w.cost_usd:,}/yr, {cheaper}, {w.scales} scales)")
        for _, r in a_pool.head(4).iterrows():
            lines.append(_row(r))

    # Regime B: large fleet / ops-constrained -> min churn s.t. SLA>=99% AND cost<=HPA
    b_pool = rep[(rep.lat_sla >= SLA_DEPLOYABLE) & (rep.cost_usd <= hpa.cost_usd)].sort_values("scales")
    lines.append(f"\nB) COST-&-STABILITY-SENSITIVE  (large fleet / SRE-constrained)")
    lines.append(f"   rule: minimize scaling actions subject to SLA >= 99% and cost <= HPA (${hpa.cost_usd:,}/yr)")
    if not b_pool.empty:
        w = b_pool.iloc[0]
        vs = f"{hpa.scales // max(w.scales,1)}x fewer scales than HPA ({hpa.scales})"
        lines.append(f"   -> WINNER: {flat_label(w)}  (SLA {w.lat_sla}%, ${w.cost_usd:,}/yr, {w.scales} scales = {vs})")
        for _, r in b_pool.head(4).iterrows():
            lines.append(_row(r))
    lines.append(_row(hpa) + "   <- HPA reference")

    lines.append("\nHONEST NOTE: on (cost, SLA) alone HPA/KEDA sit ON the frontier (100% SLA).")
    lines.append("The LLM advantage is COST-AT-MATCHED-SLA (regime A) and STABILITY (regime B),")
    lines.append("not raw SLA dominance. Present all three axes together.")
    lines.append("=" * 78)
    print("\n".join(lines))


def _tag(fig):
    fig.text(0.5, 1.09, SOURCE_TAG, fontsize=10, fontweight="bold",
              color=TAG_COLOR, va="bottom", ha="center",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF2FA", edgecolor=TAG_COLOR, linewidth=1.1))


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/{name}.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = load_summary()
    print(f"Loaded {len(s)} completed runs (>= {MIN_STEPS} steps) from {RESULTS_DIR}/")
    rep = pick_representative(s)
    s.sort_values("cost_usd").to_csv(os.path.join(OUT_DIR, "business_case_summary.csv"), index=False)
    fig_latency_cost(rep)
    fig_cost_stability(s, rep)
    fig_stability(rep)
    two_regime_verdict(s, rep)
    print(f"\nSummary table -> {OUT_DIR}/business_case_summary.csv")


if __name__ == "__main__":
    main()
