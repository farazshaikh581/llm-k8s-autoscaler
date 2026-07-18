#!/usr/bin/env python3
"""LLM as Zero-Shot Kubernetes Autoscaler — simulation on Alibaba Cluster Trace 2018.

Infrastructure model:
  - Cluster: 5 worker nodes, 4 vCPU / 8 GiB each, per-node system reserve
  - Pod spec: 250m CPU request / 500m limit (mirrors k8s/deployment.yaml),
    max 20 replicas, node scheduling enforced by CPU request
  - Startup delay: new replicas take 30s (~0.5 step) to become ready
  - Scale cooldown: 60s (1 step) between scale events (matches real HPA)
  - Queueing: c parallel per-pod M/M/1/K queues (no shared queue, matching
    k8s Service load-balancing), finite TCP listen backlogs for real losses,
    and requests/limits-aware service rates under CFS throttling; latency P90
    from the exact waiting-time mixture, calibrated on the real testbed
    (see the queueing-model section for the measured constants)
  - Cost tracked in vCPU-minutes of reserved capacity (requests)

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
    "system_reserved_m": 500,   # kubelet + system daemons per node
}

# Pod resources mirror k8s/deployment.yaml exactly (guaranteed vs burstable).
POD = {
    "cpu_request_m": 250,    # guaranteed share (CFS weight)
    "cpu_limit_m": 500,      # hard CFS quota for sustained usage
    "ram_request_mi": 128,
    "ram_limit_mi": 256,
}

REPLICA_MIN, REPLICA_MAX = 1, 20
STARTUP_DELAY_STEPS = 1     # new replicas need 1 step (60s) to become ready
SCALE_COOLDOWN_STEPS = 1    # min steps between scale events

# Service model calibrated on the real testbed (results/k8s_v2, CPU workload,
# 2414 steps): cpu_millicores/rps slope gives the CPU demand per request; the
# observed low-load P90 floor of ~47 ms splits into that demand (unthrottled
# burst within one CFS period) plus network/HTTP overhead.
CPU_DEMAND_MS = 35.0        # ms·core of CPU work per request (measured)
BASE_OVERHEAD_MS = 12.0     # network + HTTP handling floor (measured)
SERVICE_CV2 = 0.1           # sha256 loop is near-deterministic
ARRIVAL_CV2 = 4.0           # within-minute burstiness of arrivals (trace-driven
                            # load is far from Poisson; fitted to testbed)
QUEUE_PER_POD = 5           # socketserver.TCPServer request_queue_size default
CFS_BURST_SHARE = 0.25      # sustained fraction of node headroom usable above
                            # the request (CFS throttling; fitted to testbed)

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
# Queueing model: per-pod M/M/1/K with requests/limits-aware service rates
#
# Structure (each piece traceable to the real testbed):
#   - Node scheduling: pods spread evenly across nodes; per-node allocatable
#     CPU = vcpu_per_node - system_reserved; scheduling is by CPU request.
#   - Effective sustained CPU per pod: the request (guaranteed) plus a CFS
#     throttled share of the node headroom, capped at the limit. This is what
#     the peak-vs-default rate distinction requires: a pod's default rate
#     comes from its request, its peak rate from its limit.
#   - Service time: CPU_DEMAND_MS at full speed when the CFS quota is not
#     exhausted (short bursts run unthrottled), stretching toward
#     CPU_DEMAND_MS / (cpu_m/1000) as sustained utilization rises.
#   - Queue placement: a k8s Service load-balances connections across pods;
#     there is NO shared queue, so the system is c parallel M/M/1/K queues
#     each fed lambda/c, not a pooled M/M/c. K = 1 + QUEUE_PER_POD is the
#     pod's TCP listen backlog. Arrivals that find the backlog full are
#     lost: genuine SLA-loss events, impossible in an infinite-queue M/M/c.
#   - Latency: the exact M/M/1/K stationary distribution gives the
#     waiting-time mixture for accepted requests (Erlang stages); the P90 is
#     found by bisection on the mixture survival function, scaled by the
#     Sakasegawa/Allen-Cunneen factor (ca^2 + cs^2)/2 for bursty arrivals
#     over near-deterministic service.
# ---------------------------------------------------------------------------

def pods_per_node_max() -> int:
    """Max pods a node can host, by CPU request (bin-packing constraint)."""
    allocatable_m = CLUSTER["vcpu_per_node"] * 1000 - CLUSTER["system_reserved_m"]
    return allocatable_m // POD["cpu_request_m"]


def schedulable_max() -> int:
    """Hard replica ceiling: the deployment cap or node capacity, whichever
    binds first. Keeps the sim from ever exceeding what the cluster can run."""
    return min(REPLICA_MAX, CLUSTER["nodes"] * pods_per_node_max())


def effective_cpu_m(replicas: int) -> float:
    """Sustained CPU (millicores) available to each pod.

    Pods are spread evenly (k8s default topology spread); the most loaded
    node determines the contended share. Each pod is guaranteed its request;
    on top of that it can sustain only a CFS_BURST_SHARE fraction of its fair
    share of the node headroom, never exceeding its limit.
    """
    pods_on_node = max(1, math.ceil(replicas / CLUSTER["nodes"]))
    allocatable_m = CLUSTER["vcpu_per_node"] * 1000 - CLUSTER["system_reserved_m"]
    headroom_m = max(0.0, allocatable_m - pods_on_node * POD["cpu_request_m"])
    burst_m = CFS_BURST_SHARE * headroom_m / pods_on_node
    return min(POD["cpu_limit_m"], POD["cpu_request_m"] + burst_m)


def service_time_s(lambda_pod: float, cpu_m: float) -> float:
    """Mean service time per request (seconds) for a pod sustaining cpu_m.

    Interpolates between the unthrottled burst regime (an isolated request's
    CPU burst fits inside one CFS period and runs at core speed) and the
    fully throttled regime (sustained demand pinned to cpu_m). The throttled
    fraction grows with the pod's own utilization; solved as a fixed point
    since utilization depends on the service time itself.
    """
    s_burst = CPU_DEMAND_MS / 1000.0
    s_sustained = (CPU_DEMAND_MS / 1000.0) * (1000.0 / cpu_m)
    s = s_burst
    for _ in range(30):
        rho = min(lambda_pod * s, 1.0)
        s_new = s_burst + (s_sustained - s_burst) * rho
        if abs(s_new - s) < 1e-9:
            break
        s = s_new
    return s


def _erlang_sf(k: int, rate: float, t: float) -> float:
    """P(Erlang(k, rate) > t) via the Poisson series (exact, k integer)."""
    if t <= 0:
        return 1.0
    x = rate * t
    term = math.exp(-x)
    total = term
    for i in range(1, k):
        term *= x / i
        total += term
    return min(1.0, total)


def mmck_distribution(c: int, K: int, a: float) -> list[float]:
    """Stationary distribution p_0..p_K of M/M/c/K, offered load a = λ/μ.

    Computed in log space for numerical stability at large c and load.
    """
    logs = []
    for n in range(0, min(c, K) + 1):
        logs.append(n * math.log(a) - math.lgamma(n + 1) if a > 0 else (0.0 if n == 0 else -math.inf))
    log_rho = math.log(a / c) if a > 0 else -math.inf
    for n in range(c + 1, K + 1):
        logs.append(logs[c] + (n - c) * log_rho)
    m = max(logs)
    weights = [math.exp(x - m) for x in logs]
    z = sum(weights)
    return [w / z for w in weights]


def compute_metrics(ready_replicas: int, rps: int) -> dict:
    """Performance model for one step: c parallel M/M/1/K pod queues.

    rps is the request rate for the step in requests/minute (trace units).
    Returns latency P90 (ms), success rate (1 - blocking), and CPU
    utilization measured HPA-style against the pod request.
    """
    c = max(ready_replicas, 1)
    lam = rps / 60.0                    # requests/second, system-wide
    lam_pod = lam / c                   # Service LB splits load across pods
    cpu_m = effective_cpu_m(c)

    s = service_time_s(lam_pod, cpu_m)  # mean service time per request
    mu = 1.0 / s
    K = 1 + QUEUE_PER_POD               # 1 in service + TCP listen backlog

    p = mmck_distribution(1, K, lam_pod / mu)
    p_block = p[K]
    success = 1.0 - p_block

    # Waiting time of *accepted* requests at one pod: an arrival finding
    # n >= 1 in the system (n < K) waits Erlang(n, mu). P90 by bisection on
    # the mixture survival function, scaled by the Allen-Cunneen burstiness
    # factor (ca^2 + cs^2)/2 since arrivals are far from Poisson-smooth
    # within a step.
    accept = 1.0 - p_block
    q_wait = [(n, p[n] / accept) for n in range(1, K) if p[n] > 0.0]
    p_wait_total = sum(q for _, q in q_wait)
    correction = (ARRIVAL_CV2 + SERVICE_CV2) / 2.0

    if p_wait_total <= 0.10:
        wq_p90 = 0.0
    else:
        def sf(t: float) -> float:
            return sum(q * _erlang_sf(n, mu, t) for n, q in q_wait)
        lo, hi = 0.0, 1.0
        while sf(hi) > 0.10 and hi < 600.0:
            hi *= 2.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if sf(mid) > 0.10:
                lo = mid
            else:
                hi = mid
        wq_p90 = ((lo + hi) / 2.0) * correction

    # Service is near-deterministic (fixed sha256 loop), so the P90 of the
    # sojourn composes the wait quantile with the mean service time plus the
    # measured constant overhead. At idle this reproduces the observed
    # ~47 ms floor (35 ms demand + 12 ms overhead).
    latency_p90 = (wq_p90 + s) * 1000.0 + BASE_OVERHEAD_MS
    jitter = 1.0 + np.random.normal(0, 0.03)
    latency_p90 = round(min(latency_p90 * jitter, 10000.0), 1)

    # HPA-style CPU%: actual usage of served traffic against the request.
    used_m_per_pod = lam_pod * success * CPU_DEMAND_MS
    cpu_pct = round(min(used_m_per_pod / POD["cpu_request_m"] * 100.0,
                        POD["cpu_limit_m"] / POD["cpu_request_m"] * 100.0), 1)

    rho = lam_pod * s
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
        """Request a scale change. Returns (actual_target, scale_event_bool).

        The target is clamped to what the cluster can actually schedule
        (deployment cap AND per-node CPU-request capacity), so the sim can
        never run more pods than the real testbed could."""
        desired = max(REPLICA_MIN, min(schedulable_max(), desired))

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

        # cost: reserved capacity, i.e. each ready replica's CPU request
        self.cumulative_vcpu_min += self.ready_replicas * (POD["cpu_request_m"] / 1000.0)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Per-pod throughput derived from the calibrated service model, so the
# numbers quoted to the LLM always match what the simulator implements.
POD_RATE_GUARANTEED = int(round(60000.0 / (CPU_DEMAND_MS * 1000.0 / POD["cpu_request_m"]), -1))
POD_RATE_PEAK = int(round(60000.0 / (CPU_DEMAND_MS * 1000.0 / POD["cpu_limit_m"]), -1))

SYSTEM_PROMPT = (
    "You are a Kubernetes HPA controller. You manage pod replicas for a web "
    f"service running on a {CLUSTER['nodes']}-node cluster "
    f"({CLUSTER['vcpu_per_node']} vCPU, {CLUSTER['ram_gb_per_node']} GiB per node). "
    f"Each pod requests {POD['cpu_request_m']}m CPU (limit {POD['cpu_limit_m']}m) and "
    f"sustains ~{POD_RATE_GUARANTEED} req/min guaranteed, up to ~{POD_RATE_PEAK} req/min "
    "when it can burst. SLA: P90 latency < 200ms. Respond with ONLY a JSON "
    "object: {\"replicas\": N} where N is an integer between 1 and "
    f"{REPLICA_MAX}. No other text."
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
        f"- Each pod sustains ~{POD_RATE_GUARANTEED} req/min guaranteed "
        f"({POD['cpu_request_m']}m CPU request), bursting to ~{POD_RATE_PEAK} req/min "
        f"({POD['cpu_limit_m']}m limit) only when its node has spare CPU.\n"
        "- SLA: P90 latency must stay below 200ms. Above 500ms = critical.\n"
        "- CPU% is measured against the request; >100% means the pod is bursting.\n"
        "- Target CPU: 40-60%. Scale up at >65%, scale down at <30%.\n"
        "- New pods take ~30s to start. Avoid thrashing: max ±3 replicas per step.\n"
        "- Over-provisioning wastes vCPU-hours. Under-provisioning drops requests.\n"
        f"- Replicas are capped at {schedulable_max()} "
        f"({CLUSTER['nodes']} nodes x {CLUSTER['vcpu_per_node']} vCPU, "
        "minus system reserve).\n\n"
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


def keda_decision(rps: int, current_replicas: int, threshold: int | None = None) -> int:
    if threshold is None:
        # target req/min per replica: 50% of the guaranteed (request-level) rate
        threshold = int(0.5 * 60000.0 / (CPU_DEMAND_MS * 1000.0 / POD["cpu_request_m"]))
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
