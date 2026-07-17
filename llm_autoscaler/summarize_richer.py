#!/usr/bin/env python3
"""Aggregate results_richer/{scenario}/rep{N}/*.csv into per-controller means
across completed reps, matching the methodology in plot_business_case_real.py /
plot_percentiles_real.py (same cost model, same SLA threshold) but averaged
per-controller across reps instead of one row per CSV file.

Ad hoc script for the 2026-07-17 collaborator meeting report refresh — not
part of the paper pipeline.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
VCPU_REQUEST = 0.25
VCPU_HOUR_USD = 0.04
HOURS_PER_YEAR = 8760.0
MIN_STEPS = 120

MODEL_ORDER = ["hpa", "keda", "rl-ppo", "rl-dqn",
               "llama-8b", "llama-70b", "mistral-small4", "qwen3-80b", "gpt-oss-120b"]
CLASS = {"hpa": "baseline", "keda": "baseline", "rl-ppo": "rl", "rl-dqn": "rl"}
DISPLAY = {"hpa": "HPA", "keda": "KEDA", "rl-ppo": "RL-PPO", "rl-dqn": "RL-DQN",
           "llama-8b": "Llama-8B", "llama-70b": "Llama-70B", "mistral-small4": "Mistral-Small",
           "qwen3-80b": "Qwen3-80B", "gpt-oss-120b": "GPT-OSS-120B"}


def parse_name(path):
    base = os.path.basename(path)
    prefix = "k8s_cpu_" if base.startswith("k8s_cpu_") else None
    if prefix is None:
        return None
    rest = base[len(prefix):-4]
    for m in MODEL_ORDER:
        if rest == f"{m}_baseline":
            return m, "baseline"
        if rest.startswith(m + "_"):
            return m, rest[len(m) + 1:]
    return None


def load_one(scen, rep, model, variant):
    outdir = os.path.join(BASE, "results_richer", scen, f"rep{rep}")
    if variant == "baseline":
        kfile = os.path.join(outdir, f"k8s_cpu_{model}_baseline.csv")
        lfile = os.path.join(outdir, f"load_cpu_{model}_baseline.csv")
    else:
        kfile = os.path.join(outdir, f"k8s_cpu_{model}_{variant}.csv")
        lfile = os.path.join(outdir, f"load_cpu_{model}_{variant}.csv")
    if not (os.path.exists(kfile) and os.path.exists(lfile)):
        return None
    if os.path.getsize(kfile) == 0 or os.path.getsize(lfile) == 0:
        return None
    kdf = pd.read_csv(kfile)
    ldf = pd.read_csv(lfile)
    if len(kdf) < MIN_STEPS or len(ldf) < MIN_STEPS:
        return None

    ts = pd.to_datetime(kdf["timestamp"])
    dt_hours = ts.diff().dt.total_seconds().fillna(0) / 3600.0
    vcpu_hours = (kdf["replicas"].shift(1).fillna(kdf["replicas"].iloc[0]) * VCPU_REQUEST * dt_hours).sum()
    run_hours = dt_hours.sum()
    cost_usd = (vcpu_hours * VCPU_HOUR_USD) * (HOURS_PER_YEAR / run_hours) if run_hours > 0 else 0.0

    dec_ms = kdf["llm_latency_ms"].mean() if "llm_latency_ms" in kdf and model not in ("hpa", "keda", "rl-ppo", "rl-dqn") else np.nan

    return dict(
        mean=ldf["latency_mean_ms"].mean(), p50=ldf["latency_p50_ms"].mean(),
        p90=ldf["latency_p90_ms"].mean(), p99=ldf["latency_p99_ms"].mean(),
        replicas=kdf["replicas"].mean(), cpu_mc=kdf["cpu_millicores"].mean(),
        mem_mib=kdf["memory_mib"].mean(), scales=int(kdf["scale_event"].sum()),
        dec_s=dec_ms / 1000.0 if pd.notna(dec_ms) else None,
        cost_usd=cost_usd, success=kdf["success_rate"].mean() * 100,
    )


def variants_for(model):
    if model in ("hpa", "keda", "rl-ppo", "rl-dqn"):
        return ["baseline"]
    return ["zero_shot", "domain", "history_5", "cot"]


def summarize(scen):
    rows = []
    for model in MODEL_ORDER:
        for variant in variants_for(model):
            runs = []
            reps_done = []
            for rep in (1, 2, 3):
                r = load_one(scen, rep, model, variant)
                if r is not None:
                    runs.append(r)
                    reps_done.append(rep)
            if not runs:
                continue
            agg = {k: np.mean([r[k] for r in runs]) for k in
                   ("mean", "p50", "p90", "p99", "replicas", "cpu_mc", "mem_mib", "scales", "cost_usd", "success")}
            decs = [r["dec_s"] for r in runs if r["dec_s"] is not None]
            agg["dec_s"] = np.mean(decs) if decs else None
            agg["model"] = model
            agg["variant"] = variant
            agg["cls"] = CLASS.get(model, "llm")
            agg["reps"] = f"{len(reps_done)}/3"
            agg["reps_n"] = len(reps_done)
            rows.append(agg)
    df = pd.DataFrame(rows)
    return df.sort_values("p99")


if __name__ == "__main__":
    for scen in ("cpu_bursty", "wiki_diurnal"):
        print(f"\n===== {scen} =====")
        df = summarize(scen)
        pd.set_option("display.width", 200)
        cols = ["model", "variant", "cls", "reps", "mean", "p50", "p90", "p99",
                "replicas", "cpu_mc", "mem_mib", "scales", "dec_s", "cost_usd", "success"]
        print(df[cols].round(1).to_string(index=False))
        df.to_csv(os.path.join(BASE, f"results_richer_{scen}_summary.csv"), index=False)
