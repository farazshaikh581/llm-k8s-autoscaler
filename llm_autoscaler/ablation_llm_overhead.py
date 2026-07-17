#!/usr/bin/env python3
"""Ablation: LLM decision latency now (real API, results_richer/cpu_bursty)
vs. an estimate of what it would be if the model were hosted locally.

We have not run these models on local GPUs, so "if local" is an estimate,
not a measurement. It is the median of each model's own observed decision
latency: half of all real API calls already come in at or below this, i.e.
it is the latency on a "nothing went wrong" call -- no rate-limit wait, no
504 retry, no provider-fallback rotation. Local hosting removes exactly
those things (no shared quota, no network hop to a third-party host, no
other tenants competing for the endpoint), so the median is a reasonable
floor for what a dedicated local deployment would look like. It is not a
lower bound on compute time itself -- a genuinely large model still takes
real GPU time to generate, which is why Llama-70B's estimate is still ~26s,
not near-zero.

Usage: python ablation_llm_overhead.py
Output: ablation_llm_overhead.csv, fig_ablation_llm_overhead.png
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["llama-8b", "mistral-small4", "gpt-oss-120b", "llama-70b", "qwen3-80b"]
DISPLAY = {"llama-8b": "Llama-8B", "mistral-small4": "Mistral-Small",
           "gpt-oss-120b": "GPT-OSS-120B", "llama-70b": "Llama-70B", "qwen3-80b": "Qwen3-80B"}


def latencies_for(model):
    lats = []
    for f in sorted(glob.glob(f"results_richer/cpu_bursty/rep*/k8s_cpu_{model}_*.csv")):
        df = pd.read_csv(f)
        if "llm_latency_ms" not in df or len(df) < 100:
            continue
        lats.extend((df["llm_latency_ms"] / 1000.0).tolist())
    return np.array(lats)


def main():
    rows = []
    for m in MODELS:
        arr = latencies_for(m)
        if arr.size == 0:
            continue
        now = arr.mean()
        if_local = np.median(arr)
        rows.append(dict(
            model=DISPLAY[m],
            latency_now_s=round(now, 1),
            latency_if_local_s=round(if_local, 1),
            reduction_s=round(now - if_local, 1),
            reduction_pct=round((now - if_local) / now * 100, 0),
        ))
    df = pd.DataFrame(rows).sort_values("latency_now_s")
    df.to_csv("ablation_llm_overhead.csv", index=False)
    print(df.to_string(index=False))
    print("\nWrote ablation_llm_overhead.csv")

    # ---- figure: latency now vs. if-local, side by side ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df.latency_now_s, width=w, label="Latency now (real API)", color="#CC3311", zorder=3)
    ax.bar(x + w / 2, df.latency_if_local_s, width=w, label="Estimated if hosted locally", color="#4292C6", zorder=3)
    for xi, (now, loc) in enumerate(zip(df.latency_now_s, df.latency_if_local_s)):
        ax.text(xi - w / 2, now, f"{now:.1f}s", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + w / 2, loc, f"{loc:.1f}s", ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylabel("LLM decision latency (s, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.model, fontsize=9.5)
    ax.set_title("Real cluster (cpu_bursty): LLM decision latency, now vs. estimated if local",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.99, "REAL API measured; LOCAL estimated, not run", fontsize=9, fontweight="bold",
              color="#CC3311", va="top", ha="left")
    fig.tight_layout()
    fig.savefig("fig_ablation_llm_overhead.png", bbox_inches="tight", dpi=150)
    fig.savefig("fig_ablation_llm_overhead.pdf", bbox_inches="tight")
    print("Wrote fig_ablation_llm_overhead.png")


if __name__ == "__main__":
    main()
