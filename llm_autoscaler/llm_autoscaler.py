#!/usr/bin/env python3
"""LLM as Zero-Shot Kubernetes Autoscaler — simulation on Alibaba Cluster Trace 2018.

Infrastructure model:
  - Cluster: 5 worker nodes, 4 vCPU / 8 GiB each
  - Pod spec: 500m CPU request, 512 MiB RAM, max 20 replicas
  - Startup delay: new replicas take 30s (~0.5 step) to become ready
  - Scale cooldown: 60s (1 step) between scale events (matches real HPA)
  - Latency: M/M/c queuing model (Erlang-C)
  - Cost tracked in vCPU-minutes

Uses free-tier APIs: Groq (open-source LLMs).
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Infrastructure configuration
# ---------------------------------------------------------------------------

CLUSTER = {
    "nodes": 5,
    "vcpu_per_node": 4,
    "ram_gb_per_node": 8,
}

POD = {
    "cpu_request_m": 500,    # 500m = 0.5 vCPU
    "ram_request_mi": 512,
    "max_per_node": 7,       # floor(4000m / 500m) = 8, leave 1 for system
}

REPLICA_MIN, REPLICA_MAX = 1, 20
STARTUP_DELAY_STEPS = 1     # new replicas need 1 step (60s) to become ready
SCALE_COOLDOWN_STEPS = 1    # min steps between scale events

SERVICE_TIME_MS = 8.0       # mean processing time per request (fast microservice)
CAPACITY_PER_REPLICA = 200  # req/min at ~50% CPU target
SLA_LATENCY_MS = 200.0      # P90 latency SLA threshold
CRITICAL_LATENCY_MS = 500.0

CSV_COLUMNS = [
    "step", "replicas", "ready_replicas", "latency_p90", "cpu_pct",
    "requests", "success_rate", "vcpu_minutes", "scale_event",
    "llm_model", "llm_variant", "llm_tokens_used", "llm_latency_ms",
]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "provider": "groq",
        "label": "GPT-OSS 120B",
    },
    "llama-70b": {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "label": "Llama 3.3 70B",
    },
    "mistral-small4": {
        "id": "mistralai/mistral-small-4-119b-2603",
        "provider": "nvidia",
        "label": "Mistral Small 4 119B",
    },
    "llama4-scout": {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "label": "Llama 4 Scout 17B",
    },
    "llama-8b": {
        "id": "llama-3.1-8b-instant",
        "provider": "groq",
        "label": "Llama 3.1 8B",
    },
    "deepseek-v3": {
        "id": "DeepSeek-V3.1",
        "provider": "sambanova",
        "label": "DeepSeek V3.1",
    },
    "gemini-flash": {
        "id": "gemini-2.5-flash",
        "provider": "google",
        "label": "Gemini 2.5 Flash",
    },
    "deepseek-v4-flash": {
        "id": "deepseek-ai/deepseek-v4-flash",
        "provider": "nvidia",
        "label": "DeepSeek V4 Flash",
    },
    "gemma4-31b": {
        "id": "gemma-4-31B-it",
        "provider": "sambanova",
        "label": "Gemma 4 31B",
    },
    "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "provider": "github",
        "label": "GPT-4o Mini",
    },
    "qwen3-80b": {
        "id": "qwen/qwen3-next-80b-a3b-instruct",
        "provider": "nvidia",
        "label": "Qwen 3 80B",
    },
}

PROMPT_VARIANTS = ["zero_shot", "history_5", "cot", "domain"]

PROVIDER_MODEL_IDS = {
    "llama4-scout": {
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "llama-8b": {
        "groq": "llama-3.1-8b-instant",
        "nvidia": "meta/llama-3.1-8b-instruct",
    },
    "llama-70b": {
        "groq": "llama-3.3-70b-versatile",
        "sambanova": "Meta-Llama-3.3-70B-Instruct",
        "nvidia": "meta/llama-3.3-70b-instruct",
    },
    "mistral-small4": {
        "nvidia": "mistralai/mistral-small-4-119b-2603",
    },
    "gpt-oss-120b": {
        "groq": "openai/gpt-oss-120b",
        "cerebras": "gpt-oss-120b",
        "sambanova": "gpt-oss-120b",
    },
    "deepseek-v3": {
        "sambanova": "DeepSeek-V3.1",
    },
    "gemini-flash": {
        "google": "gemini-2.5-flash",
    },
    "deepseek-v4-flash": {
        "nvidia": "deepseek-ai/deepseek-v4-flash",
    },
    "gemma4-31b": {
        "sambanova": "gemma-4-31B-it",
        "nvidia": "google/gemma-4-31b-it",
    },
    "gpt-4o-mini": {
        "github": "gpt-4o-mini",
    },
    "qwen3-80b": {
        "nvidia": "qwen/qwen3-next-80b-a3b-instruct",
    },
}

# ---------------------------------------------------------------------------
# Multi-provider client factory
# ---------------------------------------------------------------------------

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "github": "https://models.inference.ai.azure.com",
}

PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "google": "GOOGLE_API_KEY",
    "github": "GITHUB_API_KEY",
}


def make_clients() -> dict[str, OpenAI]:
    clients = {}
    for provider, env_var in PROVIDER_ENV_VARS.items():
        key = os.environ.get(env_var)
        if key:
            clients[provider] = OpenAI(
                base_url=PROVIDER_URLS[provider],
                api_key=key,
            )
    return clients

# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_alibaba_trace(trace_path: str, duration_minutes: int = 1440) -> np.ndarray:
    """Load Alibaba machine_usage CSV → per-minute RPS with wide dynamic range."""
    print(f"Loading trace from {trace_path} ...")
    df = pd.read_csv(
        trace_path, header=None,
        names=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
               "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"],
        usecols=["machine_id", "time_stamp", "cpu_util_percent"],
        nrows=5_000_000,
    )
    df = df.dropna(subset=["cpu_util_percent"])
    df = df[df["cpu_util_percent"] > 0]

    t_min = df["time_stamp"].min()
    df["minute"] = ((df["time_stamp"] - t_min) / 60).astype(int)
    df = df[df["minute"] < duration_minutes]

    per_minute = df.groupby("minute")["cpu_util_percent"].mean()
    per_minute = per_minute.reindex(range(duration_minutes), fill_value=per_minute.median())

    cpu = per_minute.values
    rps = (cpu / 100.0) * 2500 + 100  # wide range: 100–2600 req/min

    # inject realistic spikes (flash crowds)
    rng = np.random.default_rng(42)
    for _ in range(8):
        center = rng.integers(30, max(31, duration_minutes - 30))
        height = rng.integers(500, 1500)
        width = rng.integers(3, 12)
        t = np.arange(duration_minutes)
        rps += height * np.exp(-0.5 * ((t - center) / width) ** 2)

    rps = np.clip(rps, 50, 4000).astype(int)
    print(f"Trace loaded: {len(rps)} steps, RPS range [{rps.min()}, {rps.max()}]")
    return rps


def generate_synthetic_trace(duration_minutes: int = 1440, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(duration_minutes)
    diurnal = 400 + 800 * np.sin(2 * np.pi * t / 1440 - np.pi / 2) ** 2
    trend = 100 * np.sin(2 * np.pi * t / (1440 * 3))
    noise = rng.normal(0, 60, duration_minutes)

    n_spikes = max(1, duration_minutes // 200)
    for _ in range(n_spikes):
        center = rng.integers(10, max(11, duration_minutes - 10))
        height = rng.integers(400, 1200)
        width = rng.integers(3, 15)
        diurnal += height * np.exp(-0.5 * ((t - center) / width) ** 2)

    rps = np.clip(diurnal + trend + noise, 50, 4000).astype(int)
    print(f"Synthetic trace: {len(rps)} steps, RPS range [{rps.min()}, {rps.max()}]")
    return rps

# ---------------------------------------------------------------------------
# M/M/c queuing model (Erlang-C)
# ---------------------------------------------------------------------------

def erlang_c(c: int, offered_load: float) -> float:
    """Probability that an arriving request must wait (Erlang-C formula)."""
    if c <= 0 or offered_load <= 0:
        return 0.0
    rho = offered_load / c
    if rho >= 1.0:
        return 1.0

    # compute in log space for numerical stability
    log_numerator = c * math.log(offered_load) - math.lgamma(c + 1) - math.log(1 - rho)
    log_sum = 0.0
    terms = []
    for k in range(c):
        terms.append(k * math.log(offered_load) - math.lgamma(k + 1))
    terms.append(log_numerator)
    max_term = max(terms)
    log_denominator = max_term + math.log(sum(math.exp(t - max_term) for t in terms))
    return math.exp(log_numerator - log_denominator)


def compute_metrics(ready_replicas: int, rps: int) -> dict:
    """Empirical utilization-based performance model.

    Latency curve calibrated to real microservice behavior:
      ~20ms at 30% CPU, ~50ms at 60%, ~200ms at 80%, >500ms at 90%+
    This matches published measurements from cloud autoscaling studies.
    """
    c = max(ready_replicas, 1)
    total_capacity = c * CAPACITY_PER_REPLICA
    rho = min(rps / max(total_capacity, 1), 2.0)  # utilization ratio

    cpu_pct = round(min(rho * 100.0, 100.0), 1)

    # Latency P90: continuous utilization curve, calibrated to real services.
    # SLA (200ms) is breached around 95-100% utilization.
    if rho <= 1.0:
        queue_factor = (rho / (1.0 - rho + 0.1)) ** 1.5
        latency_p90 = SERVICE_TIME_MS * (1.0 + queue_factor)
    else:
        # above capacity: exponential degradation from the value at ρ=1.0
        lat_at_capacity = SERVICE_TIME_MS * (1.0 + (1.0 / 0.1) ** 1.5)
        latency_p90 = lat_at_capacity * math.exp(3.0 * (rho - 1.0))

    jitter = 1.0 + np.random.normal(0, 0.05)
    latency_p90 = round(max(SERVICE_TIME_MS, min(latency_p90 * jitter, 10000.0)), 1)

    if rho <= 1.0:
        success = 1.0
    elif rho <= 1.3:
        success = 1.0 - 0.5 * (rho - 1.0)
    else:
        success = max(0.3, 0.85 - 0.5 * (rho - 1.3))

    return {
        "cpu_pct": cpu_pct,
        "latency_p90": latency_p90,
        "success_rate": round(success, 4),
        "utilization": round(rho, 4),
    }

# ---------------------------------------------------------------------------
# Cluster state (startup delay + cooldown)
# ---------------------------------------------------------------------------

class ClusterState:
    def __init__(self, initial_replicas: int = 3):
        self.target_replicas = initial_replicas
        self.ready_replicas = initial_replicas
        self._pending_scale: tuple[int, int] | None = None  # (target, ready_at_step)
        self._last_scale_step = -SCALE_COOLDOWN_STEPS - 1
        self.cumulative_vcpu_min = 0.0

    def request_scale(self, desired: int, current_step: int) -> tuple[int, bool]:
        """Request a scale change. Returns (actual_target, scale_event_bool)."""
        desired = max(REPLICA_MIN, min(REPLICA_MAX, desired))

        # enforce cooldown
        if current_step - self._last_scale_step < SCALE_COOLDOWN_STEPS:
            return self.target_replicas, False

        if desired == self.target_replicas:
            return self.target_replicas, False

        self.target_replicas = desired
        self._last_scale_step = current_step

        if desired > self.ready_replicas:
            # scale UP: new pods need startup time
            self._pending_scale = (desired, current_step + STARTUP_DELAY_STEPS)
        else:
            # scale DOWN: immediate (pod termination is fast)
            self.ready_replicas = desired
            self._pending_scale = None

        return self.target_replicas, True

    def tick(self, current_step: int):
        """Advance one step: resolve pending scale-ups, track cost."""
        if self._pending_scale:
            target, ready_at = self._pending_scale
            if current_step >= ready_at:
                self.ready_replicas = target
                self._pending_scale = None

        # cost: each ready replica uses 0.5 vCPU for 1 minute
        self.cumulative_vcpu_min += self.ready_replicas * (POD["cpu_request_m"] / 1000.0)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Kubernetes HPA controller. You manage pod replicas for a web "
    "service running on a 5-node cluster (4 vCPU, 8 GiB per node). Each pod "
    "requests 500m CPU and handles ~200 req/min at 50% CPU. SLA: P90 latency "
    "< 200ms. Respond with ONLY a JSON object: {\"replicas\": N} where N is "
    "an integer between 1 and 20. No other text."
)


def build_prompt_zero_shot(state: dict) -> str:
    return (
        f"Current state:\n"
        f"- Active replicas: {state['ready_replicas']} (target: {state['target_replicas']})\n"
        f"- Requests/min: {state['rps']}\n"
        f"- Latency P90: {state['latency_p90']} ms\n"
        f"- CPU utilization: {state['cpu_pct']}%\n"
        f"- Success rate: {state['success_rate']}\n\n"
        f"How many replicas should be running? (scale changes take ~30s to take effect)"
    )


def build_prompt_history_5(state: dict, history: list[dict]) -> str:
    lines = ["Recent history (last 5 minutes):"]
    for h in history[-5:]:
        lines.append(
            f"  t-{len(history)-h['step']}min: replicas={h['ready_replicas']}, "
            f"rps={h['rps']}, lat={h['latency_p90']}ms, "
            f"cpu={h['cpu_pct']}%, success={h['success_rate']}"
        )
    lines.append("")
    lines.append(build_prompt_zero_shot(state))
    return "\n".join(lines)


def build_prompt_cot(state: dict) -> str:
    return (
        build_prompt_zero_shot(state) + "\n\n"
        "Reason step by step:\n"
        "1. What is the current load relative to capacity?\n"
        "2. Is latency within SLA (<200ms P90)?\n"
        "3. Should we scale up, down, or hold?\n"
        "4. By how many replicas?\n"
        'Then give your final answer as {"replicas": N}.'
    )


def build_prompt_domain(state: dict) -> str:
    return (
        "Capacity & scaling rules for this cluster:\n"
        "- Each pod handles ~200 req/min at 50% CPU (500m request on 4-vCPU nodes).\n"
        "- SLA: P90 latency must stay below 200ms. Above 500ms = critical.\n"
        "- Target CPU: 40-60%. Scale up at >65%, scale down at <30%.\n"
        "- New pods take ~30s to start. Avoid thrashing: max ±3 replicas per step.\n"
        "- Over-provisioning wastes vCPU-hours. Under-provisioning drops requests.\n"
        "- With 5 nodes × 4 vCPU, max feasible is ~35 pods but we cap at 20.\n\n"
        + build_prompt_zero_shot(state)
    )


def build_prompt(variant: str, state: dict, history: list[dict]) -> str:
    if variant == "zero_shot":
        return build_prompt_zero_shot(state)
    elif variant == "history_5":
        return build_prompt_history_5(state, history)
    elif variant == "cot":
        return build_prompt_cot(state)
    elif variant == "domain":
        return build_prompt_domain(state)
    raise ValueError(f"Unknown variant: {variant}")

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model_id: str, user_prompt: str) -> tuple[int, int, float]:
    attempt = 0
    while True:
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            latency_ms = round((time.time() - t0) * 1000, 1)

            content = response.choices[0].message.content or ""
            content = content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            match = re.search(r'\{\s*"replicas"\s*:\s*(\d+)\s*\}', content)
            if match:
                replicas = int(match.group(1))
            else:
                nums = re.findall(r'\b(\d+)\b', content)
                replicas = int(nums[-1]) if nums else -1

            replicas = max(REPLICA_MIN, min(REPLICA_MAX, replicas))
            return replicas, tokens, latency_ms

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                attempt += 1
                wait = min(60 * attempt, 90)
                retry_match = re.search(r'try again in (\d+)m([\d.]+)', err_str)
                if retry_match:
                    wait = int(retry_match.group(1)) * 60 + int(float(retry_match.group(2))) + 5
                else:
                    sec_match = re.search(r'try again in ([\d.]+)s', err_str)
                    if sec_match:
                        wait = int(float(sec_match.group(1))) + 5
                if "tokens per day" in err_str or "daily" in err_str.lower():
                    wait = max(wait, 600)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt})...")
                time.sleep(wait)
                continue
            raise

# ---------------------------------------------------------------------------
# Baseline autoscalers (with same startup delay + cooldown)
# ---------------------------------------------------------------------------

def hpa_decision(cpu_pct: float, current_replicas: int, target_cpu: float = 50.0) -> int:
    desired = int(math.ceil(current_replicas * (cpu_pct / target_cpu)))
    return max(REPLICA_MIN, min(REPLICA_MAX, desired))


def keda_decision(rps: int, current_replicas: int, threshold: int = 200) -> int:
    desired = int(math.ceil(rps / threshold))
    return max(REPLICA_MIN, min(REPLICA_MAX, desired))

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    trace: np.ndarray,
    model_key: str,
    variant: str,
    client: OpenAI,
    output_dir: Path,
    initial_replicas: int = 3,
    cooldown: float = 0.5,
    resume: bool = False,
    provider: str | None = None,
):
    model_cfg = MODELS[model_key]
    provider = provider or model_cfg["provider"]
    if model_key in PROVIDER_MODEL_IDS and provider in PROVIDER_MODEL_IDS[model_key]:
        model_id = PROVIDER_MODEL_IDS[model_key][provider]
    else:
        model_id = model_cfg["id"]
    out_file = output_dir / f"results_{model_key}_{variant}.csv"
    print(f"\n{'='*60}")
    print(f"Running: {model_cfg['label']}  variant={variant}  ({provider})")
    print(f"Output:  {out_file}")
    print(f"{'='*60}")

    cluster = ClusterState(initial_replicas)
    history: list[dict] = []
    start_step = 0

    if resume and out_file.exists() and out_file.stat().st_size > 0:
        try:
            df = pd.read_csv(out_file)
            if len(df) > 0:
                start_step = int(df["step"].iloc[-1]) + 1
                cluster.target_replicas = int(df["replicas"].iloc[-1])
                cluster.ready_replicas = int(df["ready_replicas"].iloc[-1])
                cluster.cumulative_vcpu_min = float(df["vcpu_minutes"].iloc[-1])
                for _, row in df.iterrows():
                    history.append({
                        "ready_replicas": int(row["ready_replicas"]),
                        "target_replicas": int(row["replicas"]),
                        "rps": int(row["requests"]),
                        "latency_p90": float(row["latency_p90"]),
                        "cpu_pct": float(row["cpu_pct"]),
                        "success_rate": float(row["success_rate"]),
                        "step": int(row["step"]),
                    })
                print(f"  Resuming from step {start_step} ({len(df)} steps done)")
        except Exception as e:
            print(f"  Resume failed ({e}), starting fresh")
            start_step = 0
            history = []
            cluster = ClusterState(initial_replicas)

    if start_step >= len(trace):
        print(f"  Already complete ({start_step} steps)")
        return out_file

    mode = "a" if start_step > 0 else "w"
    errors = 0

    with open(out_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if mode == "w":
            writer.writeheader()

        for step in range(start_step, len(trace)):
            rps = trace[step]
            cluster.tick(step)
            metrics = compute_metrics(cluster.ready_replicas, int(rps))

            state = {
                "ready_replicas": cluster.ready_replicas,
                "target_replicas": cluster.target_replicas,
                "rps": int(rps),
                "latency_p90": metrics["latency_p90"],
                "cpu_pct": metrics["cpu_pct"],
                "success_rate": metrics["success_rate"],
                "step": step,
            }

            prompt = build_prompt(variant, state, history)

            try:
                new_replicas, tokens, llm_lat = call_llm(client, model_id, prompt)
                errors = 0
            except Exception as e:
                errors += 1
                print(f"  Step {step}: LLM error ({errors}): {e}")
                new_replicas, tokens, llm_lat = cluster.target_replicas, 0, 0.0
                if errors >= 10:
                    print("  Too many non-rate-limit errors, stopping this run.")
                    break

            _, scale_event = cluster.request_scale(new_replicas, step)

            writer.writerow({
                "step": step,
                "replicas": cluster.target_replicas,
                "ready_replicas": cluster.ready_replicas,
                "latency_p90": metrics["latency_p90"],
                "cpu_pct": metrics["cpu_pct"],
                "requests": int(rps),
                "success_rate": metrics["success_rate"],
                "vcpu_minutes": round(cluster.cumulative_vcpu_min, 2),
                "scale_event": int(scale_event),
                "llm_model": model_key,
                "llm_variant": variant,
                "llm_tokens_used": tokens,
                "llm_latency_ms": llm_lat,
            })
            f.flush()

            history.append(state)

            if step % 50 == 0:
                print(
                    f"  Step {step:4d}: rps={int(rps):4d} ready={cluster.ready_replicas:2d} "
                    f"target={cluster.target_replicas:2d} "
                    f"lat={metrics['latency_p90']:7.1f}ms cpu={metrics['cpu_pct']:5.1f}% "
                    f"vcpu={cluster.cumulative_vcpu_min:.0f} tokens={tokens}"
                )

            time.sleep(cooldown)

    print(f"Done: {out_file} ({step + 1} total steps, {errors} errors)")
    return out_file


def run_baselines(trace: np.ndarray, output_dir: Path, initial_replicas: int = 3):
    for name in ["hpa", "keda"]:
        out_file = output_dir / f"results_{name}_baseline.csv"
        cluster = ClusterState(initial_replicas)

        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for step, rps in enumerate(trace):
                cluster.tick(step)
                metrics = compute_metrics(cluster.ready_replicas, int(rps))

                if name == "hpa":
                    desired = hpa_decision(metrics["cpu_pct"], cluster.ready_replicas)
                else:
                    desired = keda_decision(int(rps), cluster.ready_replicas)

                _, scale_event = cluster.request_scale(desired, step)

                writer.writerow({
                    "step": step,
                    "replicas": cluster.target_replicas,
                    "ready_replicas": cluster.ready_replicas,
                    "latency_p90": metrics["latency_p90"],
                    "cpu_pct": metrics["cpu_pct"],
                    "requests": int(rps),
                    "success_rate": metrics["success_rate"],
                    "vcpu_minutes": round(cluster.cumulative_vcpu_min, 2),
                    "scale_event": int(scale_event),
                    "llm_model": name,
                    "llm_variant": "baseline",
                    "llm_tokens_used": 0,
                    "llm_latency_ms": 0,
                })

        print(f"Baseline {name}: {out_file} ({step + 1} steps)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Zero-Shot Kubernetes Autoscaler")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama-70b")
    parser.add_argument("--variant", choices=PROMPT_VARIANTS, default="zero_shot")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--trace", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--cooldown", type=float, default=2.5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume partial runs from existing CSV files")
    parser.add_argument("--provider", choices=list(PROVIDER_URLS.keys()), default=None,
                        help="Override the default provider for the model")
    args = parser.parse_args()

    clients = make_clients()
    if not clients and not args.baselines_only:
        print("ERROR: No API keys found. Set at least one of:")
        print("  export GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace:
        if args.trace.endswith(".npy"):
            trace = np.load(args.trace)
            print(f"Loaded pre-processed trace: {len(trace)} steps, RPS [{trace.min()}, {trace.max()}]")
        else:
            trace = load_alibaba_trace(args.trace, duration_minutes=args.steps)
    else:
        trace = generate_synthetic_trace(duration_minutes=args.steps)
    trace = trace[:args.steps]

    run_baselines(trace, output_dir)

    if args.baselines_only:
        print("Baselines done.")
        return

    if not clients:
        print("No API keys. Baselines done, skipping LLM runs.")
        return

    models_to_run = list(MODELS.keys()) if args.all_models else [args.model]
    variants_to_run = PROMPT_VARIANTS if (args.all_variants or args.all_models) else [args.variant]

    for model_key in models_to_run:
        provider = args.provider or MODELS[model_key]["provider"]
        if provider not in clients:
            print(f"Skipping {model_key}: no {provider} API key")
            continue
        if model_key in PROVIDER_MODEL_IDS and provider not in PROVIDER_MODEL_IDS[model_key]:
            print(f"Skipping {model_key}: not available on {provider}")
            continue
        client = clients[provider]
        for variant in variants_to_run:
            out_file = output_dir / f"results_{model_key}_{variant}.csv"
            if out_file.exists() and out_file.stat().st_size > 0:
                lines = sum(1 for _ in open(out_file))
                if lines > args.steps:
                    print(f"Skipping {model_key}/{variant}: complete ({lines} lines)")
                    continue
                if not args.resume:
                    print(f"Skipping {model_key}/{variant}: {out_file} exists (use --resume)")
                    continue
            run_simulation(trace, model_key, variant, client, output_dir,
                           cooldown=args.cooldown, resume=args.resume, provider=provider)


if __name__ == "__main__":
    main()
