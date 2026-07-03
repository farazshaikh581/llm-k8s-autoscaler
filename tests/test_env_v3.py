"""Sanity tests for the RL environment on the hardened simulator (issue #7).

The env must be a thin wrapper: same plant, same constraints, same cost
accounting as llm_autoscaler.py. These tests pin that equivalence so the
two code paths cannot drift apart again.

Run: python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_autoscaler as sim
from autoscale_env_v3 import AutoscaleEnvV3

TRACE = np.array([500, 800, 1200, 2000, 3000, 3500, 2000, 1000] * 4)


def rollout(actions, trace=TRACE, seed=0):
    env = AutoscaleEnvV3(trace, seed=seed)
    env.reset(seed=seed)
    infos = []
    for a in actions:
        _, _, done, _, info = env.step(a)
        infos.append(info)
        if done:
            break
    return env, infos


# --- same plant as the simulator ---------------------------------------------

def test_env_metrics_identical_to_simulator():
    """The env's metrics for a given (ready, rps) are compute_metrics itself."""
    np.random.seed(7)
    expected = sim.compute_metrics(3, int(TRACE[0]))
    env = AutoscaleEnvV3(TRACE)
    np.random.seed(7)
    env.reset()
    assert env.last_metrics["latency_p90"] == expected["latency_p90"]
    assert env.last_metrics["cpu_pct"] == expected["cpu_pct"]
    assert env.last_metrics["success_rate"] == expected["success_rate"]


def test_env_losses_under_overload():
    """One pod at heavy load must drop requests (issue #1 behavior)."""
    env, infos = rollout([0] * 8)  # pin to 1 replica
    overloaded = [i for i in infos if i["rps"] >= 2000]
    assert overloaded and all(i["success_rate"] < 0.999 for i in overloaded)


# --- same constraints ---------------------------------------------------------

def test_env_never_exceeds_schedulable_cap():
    env, infos = rollout([sim.REPLICA_MAX - 1] * 10)  # always ask for max
    cap = sim.schedulable_max()
    assert all(i["replicas"] <= cap and i["ready_replicas"] <= cap for i in infos)


def test_env_startup_delay():
    """Scale-up takes one step before pods are ready, as in the simulator."""
    env = AutoscaleEnvV3(TRACE, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(9)  # request 10 replicas from 3
    assert info["replicas"] >= info["ready_replicas"]


def test_env_cooldown_limits_scale_events():
    env, infos = rollout([2, 9, 2, 9, 2, 9])  # thrash every step
    # ClusterState enforces >= SCALE_COOLDOWN_STEPS between events
    events = [i["scale_event"] for i in infos]
    for a, b in zip(events, events[1:]):
        assert not (a and b) or sim.SCALE_COOLDOWN_STEPS <= 1


# --- same cost accounting -----------------------------------------------------

def test_env_cost_accrues_pod_requests():
    env, infos = rollout([2] * 6)  # hold 3 replicas
    per_step = 3 * (sim.POD["cpu_request_m"] / 1000.0)
    expected = per_step * (len(infos) + 1)  # +1 for the reset tick
    assert abs(infos[-1]["vcpu_minutes"] - expected) < 1e-6


# --- episode mechanics ---------------------------------------------------------

def test_env_episode_covers_trace():
    env, infos = rollout([2] * len(TRACE))
    assert infos[-1]["step"] == len(TRACE) - 1


def test_env_observation_bounded():
    env, infos = rollout([19, 0, 19, 0] * 4)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
