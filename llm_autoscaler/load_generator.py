#!/usr/bin/env python3
"""Trace-driven HTTP load generator for K8s autoscaling experiments.

Replays an Alibaba-derived RPS trace against a Kubernetes service,
adjusting concurrency each step to match target request rate.

Usage:
  python load_generator.py --service-ip <cluster-ip> --trace traces/trace_cpu.npy \
      --steps 60 --interval 60 --output load_results/cpu_load.csv

Each step (default 60s):
  1. Read target RPS from trace
  2. Fire requests at that rate for the step duration
  3. Record actual RPS achieved, latency percentiles, success rate
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CSV_COLUMNS = [
    "step", "timestamp", "target_rps", "actual_rps",
    "latency_mean_ms", "latency_p50_ms", "latency_p90_ms", "latency_p99_ms",
    "success_count", "error_count", "success_rate",
]


def send_request(url: str, timeout: float = 5.0) -> tuple[bool, float]:
    """Send one HTTP request. Returns (success, latency_ms)."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 2,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        success = result.stdout.strip() == "200"
        return success, latency_ms
    except Exception:
        latency_ms = (time.monotonic() - t0) * 1000
        return False, latency_ms


def run_step(url: str, target_rps: int, duration_s: int,
             max_workers: int = 50) -> dict:
    """Send requests at target_rps for duration_s seconds."""
    total_requests = target_rps * duration_s // 60  # convert req/min to total for this step
    total_requests = max(1, total_requests)

    interval = duration_s / total_requests  # time between request launches

    latencies = []
    successes = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=min(max_workers, total_requests)) as pool:
        futures = []
        t_start = time.monotonic()

        for i in range(total_requests):
            target_time = t_start + i * interval
            now = time.monotonic()
            if target_time > now:
                time.sleep(target_time - now)

            futures.append(pool.submit(send_request, url))

        for f in as_completed(futures):
            success, lat = f.result()
            latencies.append(lat)
            if success:
                successes += 1
            else:
                errors += 1

    elapsed = time.monotonic() - t_start
    latencies.sort()

    if not latencies:
        return {
            "actual_rps": 0, "latency_mean_ms": 0,
            "latency_p50_ms": 0, "latency_p90_ms": 0, "latency_p99_ms": 0,
            "success_count": 0, "error_count": 0, "success_rate": 0,
        }

    n = len(latencies)
    return {
        "actual_rps": round(n / elapsed * 60),  # actual req/min achieved
        "latency_mean_ms": round(np.mean(latencies), 1),
        "latency_p50_ms": round(latencies[int(n * 0.50)], 1),
        "latency_p90_ms": round(latencies[int(n * 0.90)], 1),
        "latency_p99_ms": round(latencies[min(int(n * 0.99), n - 1)], 1),
        "success_count": successes,
        "error_count": errors,
        "success_rate": round(successes / n, 4) if n > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Trace-driven HTTP load generator")
    parser.add_argument("--service-ip", required=True, help="K8s ClusterIP of the target service")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--trace", required=True, help="Path to .npy trace file")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--interval", type=int, default=60, help="Seconds per step")
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="Multiply trace RPS by this factor (for resource-constrained setups)")
    args = parser.parse_args()

    trace = np.load(args.trace)[:args.steps]
    trace = (trace * args.scale_factor).astype(int)
    trace = np.clip(trace, 1, 10000)

    url = f"http://{args.service_ip}:{args.port}/"
    output = Path(args.output) if args.output else Path(f"load_{int(time.time())}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Load generator starting")
    print(f"  Target: {url}")
    print(f"  Trace: {len(trace)} steps, RPS [{trace.min()}, {trace.max()}]")
    print(f"  Interval: {args.interval}s per step")
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Output: {output}")
    print()

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for step, target_rps in enumerate(trace):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            result = run_step(url, int(target_rps), args.interval,
                              max_workers=args.max_workers)

            row = {
                "step": step,
                "timestamp": ts,
                "target_rps": int(target_rps),
                **result,
            }
            writer.writerow(row)
            f.flush()

            sla_flag = " !! SLA" if result["latency_p90_ms"] > 200 else ""
            print(
                f"  Step {step:3d} | target={int(target_rps):4d} actual={result['actual_rps']:4d} "
                f"| p90={result['latency_p90_ms']:7.1f}ms "
                f"| ok={result['success_count']:3d} err={result['error_count']:2d}"
                f"{sla_flag}"
            )

    print(f"\nDone: {output}")


if __name__ == "__main__":
    main()
