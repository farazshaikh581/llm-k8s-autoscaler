#!/usr/bin/env python3
"""LLM autoscaler for real Kubernetes — reads metrics, calls LLM, scales deployments.

Multi-provider support: Groq, NVIDIA, Cerebras, SambaNova, Google.
RL baseline support: loads trained DQN/PPO models for real cluster decisions.

Usage:
  python k8s_autoscaler.py --workload cpu --model llama4-scout --variant zero_shot \
      --provider groq --steps 120 --trace traces/trace_cpu.npy

Requires: kubectl configured, metrics-server running, load generator active.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKLOADS = {
    "cpu": {"deployment": "workload-cpu", "service": "workload-cpu", "type": "CPU-intensive"},
    "io":  {"deployment": "workload-io",  "service": "workload-io",  "type": "I/O-bound"},
}

MODELS = {
    "llama4-scout":     {"label": "Llama 4 Scout 17B"},
    "llama-8b":         {"label": "Llama 3.1 8B"},
    "llama-70b":        {"label": "Llama 3.3 70B"},
    "mistral-small4":   {"label": "Mistral Small 4 119B"},
    "gpt-oss-120b":     {"label": "GPT-OSS 120B"},
    "deepseek-v4-flash":{"label": "DeepSeek V4 Flash"},
    "qwen3-80b":        {"label": "Qwen 3 80B"},
}

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
        "nvidia": "openai/gpt-oss-120b",
    },
    "deepseek-v4-flash": {
        "nvidia": "deepseek-ai/deepseek-v4-flash",
    },
    "qwen3-80b": {
        "nvidia": "qwen/qwen3-next-80b-a3b-instruct",
    },
}

PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "google": "GOOGLE_API_KEY",
}

PROMPT_VARIANTS = ["zero_shot", "history_5", "cot", "domain", "baseline"]

REPLICA_MIN, REPLICA_MAX = 1, 20
SCALE_COOLDOWN_S = 60
SLA_LATENCY_MS = 200

CSV_COLUMNS = [
    "step", "timestamp", "replicas", "ready_replicas",
    "cpu_millicores", "memory_mib", "rps_target",
    "latency_p90_ms", "success_rate",
    "llm_decision", "scale_event",
    "llm_model", "llm_variant", "llm_tokens", "llm_latency_ms",
    "workload_type",
]

# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------

def make_client(provider: str) -> OpenAI:
    env_var = PROVIDER_ENV_VARS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"ERROR: Set {env_var} for provider {provider}")
        sys.exit(1)
    return OpenAI(base_url=PROVIDER_URLS[provider], api_key=api_key)


def resolve_model_id(model_key: str, provider: str) -> str:
    if model_key in PROVIDER_MODEL_IDS and provider in PROVIDER_MODEL_IDS[model_key]:
        return PROVIDER_MODEL_IDS[model_key][provider]
    raise ValueError(f"Model {model_key} not available on provider {provider}. "
                     f"Available: {list(PROVIDER_MODEL_IDS.get(model_key, {}).keys())}")

# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------

def kubectl(*args) -> str:
    result = subprocess.run(
        ["kubectl"] + list(args),
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def get_deployment_replicas(deployment: str) -> tuple[int, int]:
    out = kubectl("get", "deployment", deployment,
                  "-o", "jsonpath={.spec.replicas} {.status.readyReplicas}")
    parts = out.split()
    desired = int(parts[0]) if parts[0] else 0
    ready = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return desired, ready


def get_pod_metrics(deployment: str) -> dict:
    label = f"app={deployment}"
    out = kubectl("top", "pods", "-l", label, "--no-headers")

    total_cpu_m = 0
    total_mem_mi = 0
    count = 0

    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cpu_str = parts[1]
        mem_str = parts[2]

        if cpu_str.endswith("m"):
            total_cpu_m += int(cpu_str[:-1])
        else:
            total_cpu_m += int(cpu_str) * 1000

        if mem_str.endswith("Mi"):
            total_mem_mi += int(mem_str[:-2])
        elif mem_str.endswith("Gi"):
            total_mem_mi += int(float(mem_str[:-2]) * 1024)

        count += 1

    return {
        "total_cpu_m": total_cpu_m,
        "total_mem_mi": total_mem_mi,
        "pod_count": count,
        "avg_cpu_m": total_cpu_m // max(count, 1),
        "avg_mem_mi": total_mem_mi // max(count, 1),
    }


def scale_deployment(deployment: str, replicas: int) -> bool:
    replicas = max(REPLICA_MIN, min(REPLICA_MAX, replicas))
    kubectl("scale", "deployment", deployment, f"--replicas={replicas}")
    return True

# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(service_ip: str, n_requests: int = 20) -> tuple[float, float]:
    latencies = []
    successes = 0

    for _ in range(n_requests):
        try:
            t0 = time.monotonic()
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", "5", f"http://{service_ip}:80/"],
                capture_output=True, text=True, timeout=10,
            )
            elapsed = (time.monotonic() - t0) * 1000
            latencies.append(elapsed)
            if result.stdout.strip() == "200":
                successes += 1
        except Exception:
            latencies.append(5000)

    if not latencies:
        return 0, 0

    latencies.sort()
    p90_idx = int(len(latencies) * 0.9)
    p90 = latencies[p90_idx] if p90_idx < len(latencies) else latencies[-1]
    success_rate = successes / len(latencies)
    return round(p90, 1), round(success_rate, 4)

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Kubernetes HPA controller managing a real cluster. "
    "Two worker nodes: 3 vCPU, 16 GiB RAM each. Each pod requests 250m CPU (limit 500m). "
    "Max replicas: 20. SLA: P90 latency < 200ms. "
    'Respond with ONLY: {"replicas": N} where N is 1-20.'
)


def build_prompt(variant: str, state: dict, history: list, workload_type: str) -> str:
    base = (
        f"Workload type: {workload_type}\n"
        f"Current state:\n"
        f"- Replicas: {state['ready']} ready / {state['desired']} desired\n"
        f"- Avg CPU per pod: {state['avg_cpu_m']}m / 500m limit\n"
        f"- Avg Memory per pod: {state['avg_mem_mi']}Mi\n"
        f"- Target RPS: {state['rps_target']}\n"
        f"- Measured Latency P90: {state['latency_p90']}ms\n"
        f"- Success rate: {state['success_rate']}\n\n"
        f"How many replicas should be running?"
    )

    if variant == "zero_shot":
        return base
    elif variant == "history_5":
        lines = ["Recent history (last 5 minutes):"]
        for h in history[-5:]:
            lines.append(
                f"  t-{len(history)-h['step']}min: replicas={h['ready']}, "
                f"cpu={h['avg_cpu_m']}m, lat={h['latency_p90']}ms, "
                f"rps={h['rps_target']}, success={h['success_rate']}"
            )
        return "\n".join(lines) + "\n\n" + base
    elif variant == "cot":
        return (
            base + "\n\n"
            "Reason step by step:\n"
            "1. Current CPU utilization vs capacity?\n"
            "2. Is latency within SLA (<200ms)?\n"
            "3. Scale up, down, or hold?\n"
            '4. Final answer as {"replicas": N}.'
        )
    elif variant == "domain":
        return (
            "Scaling rules:\n"
            "- Each pod handles load proportional to its CPU usage.\n"
            "- Target CPU: 40-60% of limit (200-300m per pod).\n"
            "- Scale up if avg CPU > 350m or latency > 150ms.\n"
            "- Scale down if avg CPU < 150m and latency < 50ms.\n"
            "- Max +/- 3 replicas per step to avoid thrashing.\n\n"
            + base
        )
    raise ValueError(f"Unknown variant: {variant}")

# ---------------------------------------------------------------------------
# LLM call with robust retry
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model_id: str, prompt: str) -> tuple[int, int, float]:
    attempt = 0
    while True:
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            latency_ms = round((time.time() - t0) * 1000, 1)

            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            match = re.search(r'\{\s*"replicas"\s*:\s*(\d+)\s*\}', content)
            if match:
                replicas = int(match.group(1))
            else:
                nums = re.findall(r'\b(\d+)\b', content)
                replicas = int(nums[-1]) if nums else -1

            return max(REPLICA_MIN, min(REPLICA_MAX, replicas)), tokens, latency_ms

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                attempt += 1
                wait = min(60 * attempt, 900)
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
# Baseline autoscalers
# ---------------------------------------------------------------------------

def hpa_decision(avg_cpu_m: int, current_replicas: int, target_cpu_m: int = 250) -> int:
    desired = math.ceil(current_replicas * (avg_cpu_m / target_cpu_m))
    return max(REPLICA_MIN, min(REPLICA_MAX, desired))

def keda_decision(rps: int, current_replicas: int, threshold: int = 50) -> int:
    desired = math.ceil(rps / threshold)
    return max(REPLICA_MIN, min(REPLICA_MAX, desired))

# ---------------------------------------------------------------------------
# RL baseline (loads trained model, maps real metrics to observation space)
# ---------------------------------------------------------------------------

def load_rl_model(algo_name: str, version: str = "v2"):
    from stable_baselines3 import DQN, PPO
    base = Path(__file__).parent
    if version == "v2":
        model_dir = base / "models_v2"
        suffix = "_autoscaler_v2"
    else:
        model_dir = base / "models"
        suffix = "_autoscaler"
    if algo_name == "dqn":
        return DQN.load(model_dir / f"DQN{suffix}")
    elif algo_name == "ppo":
        return PPO.load(model_dir / f"PPO{suffix}")
    raise ValueError(f"Unknown RL algo: {algo_name}")


def rl_decision(model, state: dict, ready_replicas: int, version: str = "v2") -> int:
    max_replicas = 20
    rps = state["rps_target"]
    cpu_raw = state["avg_cpu_m"] / 500.0
    lat_norm = min(state["latency_p90"] / 1000.0, 1.0)

    if version == "v2":
        mem_pct = state.get("memory_pct", 30.0) / 100.0
        net_pct = state.get("network_pct", 5.0) / 100.0
        pending = state.get("pending_pods", 0) / max_replicas
        obs = np.array([
            min(cpu_raw, 1.0),
            min(state["latency_p90"] / 2000.0, 1.0),
            state["success_rate"],
            min(rps / 4000.0, 1.0),
            ready_replicas / max_replicas,
            min(mem_pct, 1.0),
            min(net_pct, 1.0),
            min(pending, 1.0),
        ], dtype=np.float32)
    else:
        obs = np.array([
            min(cpu_raw, 1.0),
            lat_norm,
            state["success_rate"],
            min(rps / 4000.0, 1.0),
            ready_replicas / max_replicas,
        ], dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    return int(action) + 1

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    wl = WORKLOADS[args.workload]
    deployment = wl["deployment"]
    workload_type = wl["type"]

    svc_ip = kubectl("get", "svc", wl["service"],
                     "-o", "jsonpath={.spec.clusterIP}")

    trace = np.load(args.trace)[:args.steps]
    print(f"Trace: {len(trace)} steps, RPS [{trace.min()}, {trace.max()}]")

    is_baseline = args.model in ("hpa", "keda")
    is_rl = args.model in ("rl-dqn", "rl-ppo")
    is_llm = not is_baseline and not is_rl

    client = None
    model_id = None
    rl_model = None

    if is_llm:
        provider = args.provider
        if not provider:
            avail = list(PROVIDER_MODEL_IDS.get(args.model, {}).keys())
            if not avail:
                print(f"ERROR: No providers configured for {args.model}")
                sys.exit(1)
            provider = avail[0]
        model_id = resolve_model_id(args.model, provider)
        client = make_client(provider)
        print(f"Provider: {provider}, Model ID: {model_id}")
    elif is_rl:
        algo = args.model.split("-")[1]
        rl_model = load_rl_model(algo)
        print(f"Loaded RL model: {args.model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"k8s_{args.workload}_{args.model}_{args.variant}.csv"

    start_step = 0
    history = []

    if args.resume and out_file.exists() and out_file.stat().st_size > 0:
        import pandas as pd
        try:
            df = pd.read_csv(out_file)
            if len(df) > 0:
                start_step = len(df)
                for _, row in df.iterrows():
                    history.append({
                        "desired": int(row["replicas"]),
                        "ready": int(row["ready_replicas"]),
                        "avg_cpu_m": int(row["cpu_millicores"]) // max(int(row["ready_replicas"]), 1),
                        "avg_mem_mi": int(row["memory_mib"]) // max(int(row["ready_replicas"]), 1),
                        "rps_target": int(row["rps_target"]),
                        "latency_p90": float(row["latency_p90_ms"]),
                        "success_rate": float(row["success_rate"]),
                        "step": int(row["step"]),
                    })
                print(f"Resuming from step {start_step} ({len(df)} steps done)")
        except Exception as e:
            print(f"Resume failed ({e}), starting fresh")
            start_step = 0
            history = []

    if start_step >= len(trace):
        print(f"Already complete ({start_step} steps)")
        return out_file

    if start_step == 0:
        scale_deployment(deployment, 3)
        time.sleep(10)

    mode = "a" if start_step > 0 else "w"
    errors = 0

    with open(out_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if mode == "w":
            writer.writeheader()

        for step in range(start_step, len(trace)):
            rps_target = int(trace[step])
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            desired, ready = get_deployment_replicas(deployment)
            try:
                metrics = get_pod_metrics(deployment)
            except Exception as e:
                print(f"  Step {step}: metrics error: {e}")
                metrics = {"avg_cpu_m": 0, "avg_mem_mi": 0, "total_cpu_m": 0,
                           "total_mem_mi": 0, "pod_count": ready}

            latency_p90, success_rate = measure_latency(svc_ip, n_requests=10)

            state = {
                "desired": desired, "ready": ready,
                "avg_cpu_m": metrics["avg_cpu_m"],
                "avg_mem_mi": metrics["avg_mem_mi"],
                "rps_target": rps_target,
                "latency_p90": latency_p90,
                "success_rate": success_rate,
                "step": step,
            }

            tokens, llm_lat = 0, 0.0
            if args.model == "hpa":
                new_replicas = hpa_decision(metrics["avg_cpu_m"], ready)
            elif args.model == "keda":
                new_replicas = keda_decision(rps_target, ready)
            elif is_rl:
                new_replicas = rl_decision(rl_model, state, ready)
            else:
                prompt = build_prompt(args.variant, state, history, workload_type)
                try:
                    new_replicas, tokens, llm_lat = call_llm(client, model_id, prompt)
                    errors = 0
                except Exception as e:
                    errors += 1
                    print(f"  Step {step}: LLM error ({errors}): {e}")
                    new_replicas = ready
                    if errors >= 10:
                        print("  Too many non-rate-limit errors, stopping.")
                        break

            scale_event = False
            now = time.time()
            if new_replicas != desired and (now - (history[-1].get("_scale_time", 0) if history else 0)) >= SCALE_COOLDOWN_S:
                scale_deployment(deployment, new_replicas)
                scale_event = True
                state["_scale_time"] = now

            writer.writerow({
                "step": step,
                "timestamp": ts,
                "replicas": new_replicas if scale_event else desired,
                "ready_replicas": ready,
                "cpu_millicores": metrics["total_cpu_m"],
                "memory_mib": metrics["total_mem_mi"],
                "rps_target": rps_target,
                "latency_p90_ms": latency_p90,
                "success_rate": success_rate,
                "llm_decision": new_replicas,
                "scale_event": int(scale_event),
                "llm_model": args.model,
                "llm_variant": args.variant,
                "llm_tokens": tokens,
                "llm_latency_ms": llm_lat,
                "workload_type": workload_type,
            })
            f.flush()

            history.append(state)

            if step % 5 == 0:
                print(
                    f"  [{step:3d}] {ts} rps={rps_target:4d} ready={ready:2d} "
                    f"cpu={metrics['avg_cpu_m']:3d}m lat={latency_p90:6.1f}ms "
                    f"decision={new_replicas:2d} {'SCALED' if scale_event else ''}"
                )

            time.sleep(max(1, args.interval))

    print(f"\nDone: {out_file}")
    return out_file


def main():
    all_models = list(MODELS.keys()) + ["hpa", "keda", "rl-dqn", "rl-ppo"]
    all_providers = list(PROVIDER_URLS.keys())

    parser = argparse.ArgumentParser(description="LLM K8s Autoscaler (real cluster)")
    parser.add_argument("--workload", choices=list(WORKLOADS.keys()), required=True)
    parser.add_argument("--model", choices=all_models, required=True)
    parser.add_argument("--variant", choices=PROMPT_VARIANTS, default="zero_shot")
    parser.add_argument("--provider", choices=all_providers, default=None,
                        help="LLM provider (auto-detected if not set)")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between steps")
    parser.add_argument("--output-dir", type=str, default="results_k8s")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
