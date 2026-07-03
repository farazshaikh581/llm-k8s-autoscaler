"""Sanity tests for the hardened simulator queueing model.

Each test maps to an acceptance criterion from the simulation-fidelity
issues: real losses under overload (#1), principled latency behavior (#2),
requests/limits distinction (#3), capacity consistency (#4).

Run: python -m pytest tests/ -q   (or python tests/test_sim_model.py)
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "llm_autoscaler_sim", Path(__file__).resolve().parent.parent / "llm_autoscaler.py"
)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def metrics(c, rpm, seed=0):
    np.random.seed(seed)
    return sim.compute_metrics(c, rpm)


# --- Issue #1: genuine losses -----------------------------------------------

def test_no_losses_with_ample_capacity():
    for c, rpm in [(5, 300), (10, 1000), (20, 3000)]:
        assert metrics(c, rpm)["success_rate"] >= 0.9999


def test_losses_under_overload():
    m = metrics(2, 3000)  # 1500 req/min per pod, far beyond peak rate
    assert m["success_rate"] < 0.95, "overload must produce real blocking losses"


def test_losses_increase_with_overload():
    s = [metrics(2, rpm)["success_rate"] for rpm in (1000, 2000, 4000, 8000)]
    assert all(a >= b for a, b in zip(s, s[1:])), f"success must not rise with load: {s}"


def test_blocking_probability_is_proper():
    for c, rpm in [(1, 100), (1, 5000), (10, 50000)]:
        m = metrics(c, rpm)
        assert 0.0 <= m["success_rate"] <= 1.0


# --- Issue #2: latency model ------------------------------------------------

def test_latency_floor_matches_calibration():
    # idle latency = CPU demand at burst speed + measured overhead (~47 ms)
    lat = np.mean([metrics(10, 50, seed=s)["latency_p90"] for s in range(50)])
    floor = sim.CPU_DEMAND_MS + sim.BASE_OVERHEAD_MS
    assert abs(lat - floor) < 6, f"idle P90 {lat:.1f} should be near {floor}"


def test_latency_monotone_in_load():
    np.random.seed(0)
    lats = []
    for rpm in (100, 600, 1200, 1800, 2400):
        # average out jitter
        lats.append(np.mean([metrics(3, rpm, seed=s)["latency_p90"] for s in range(30)]))
    assert all(a < b for a, b in zip(lats, lats[1:])), f"not monotone: {lats}"


def test_latency_monotone_in_replicas():
    lat_few = np.mean([metrics(3, 2000, seed=s)["latency_p90"] for s in range(30)])
    lat_many = np.mean([metrics(15, 2000, seed=s)["latency_p90"] for s in range(30)])
    assert lat_many < lat_few


def test_mmck_distribution_normalized():
    for c, K, a in [(1, 6, 0.5), (1, 6, 3.0), (4, 24, 3.9), (20, 120, 25.0)]:
        p = sim.mmck_distribution(c, K, a)
        assert len(p) == K + 1
        assert abs(sum(p) - 1.0) < 1e-9
        assert all(x >= 0 for x in p)


# --- Issue #3: requests/limits ----------------------------------------------

def test_guaranteed_vs_peak_rate_distinct():
    assert sim.POD_RATE_PEAK > sim.POD_RATE_GUARANTEED * 1.5, (
        "peak (limit) rate must be clearly above guaranteed (request) rate"
    )


def test_effective_cpu_bounded_by_request_and_limit():
    for replicas in range(1, sim.schedulable_max() + 1):
        eff = sim.effective_cpu_m(replicas)
        assert sim.POD["cpu_request_m"] <= eff <= sim.POD["cpu_limit_m"]


def test_contention_reduces_effective_cpu():
    # a lone pod on a node can burst harder than pods on a packed node
    assert sim.effective_cpu_m(1) >= sim.effective_cpu_m(20)
    assert sim.effective_cpu_m(20) < sim.POD["cpu_limit_m"]


def test_service_time_stretches_under_sustained_load():
    idle = sim.service_time_s(0.1, sim.POD["cpu_request_m"])
    busy = sim.service_time_s(6.0, sim.POD["cpu_request_m"])
    assert busy > idle
    # sustained service time never exceeds the fully-throttled bound
    bound = (sim.CPU_DEMAND_MS / 1000.0) * (1000.0 / sim.POD["cpu_request_m"])
    assert busy <= bound + 1e-9


# --- Issue #4: capacity consistency -----------------------------------------

def test_schedulable_max_respects_node_capacity():
    per_node = sim.pods_per_node_max()
    alloc = sim.CLUSTER["vcpu_per_node"] * 1000 - sim.CLUSTER["system_reserved_m"]
    assert per_node == alloc // sim.POD["cpu_request_m"]
    assert sim.schedulable_max() <= min(sim.REPLICA_MAX, sim.CLUSTER["nodes"] * per_node)


def test_scale_request_clamped_to_schedulable():
    cs = sim.ClusterState(initial_replicas=3)
    target, changed = cs.request_scale(10_000, current_step=5)
    assert changed
    assert target == sim.schedulable_max()


def test_scale_up_has_startup_delay():
    cs = sim.ClusterState(initial_replicas=3)
    cs.request_scale(8, current_step=5)
    assert cs.ready_replicas == 3, "new pods must not be ready instantly"
    cs.tick(current_step=5 + sim.STARTUP_DELAY_STEPS)
    assert cs.ready_replicas == 8


def test_vcpu_cost_uses_requests():
    cs = sim.ClusterState(initial_replicas=4)
    cs.tick(current_step=0)
    assert math.isclose(
        cs.cumulative_vcpu_min, 4 * sim.POD["cpu_request_m"] / 1000.0
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
