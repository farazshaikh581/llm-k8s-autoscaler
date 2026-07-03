"""Gym environment on the hardened simulator (issue #7).

Unlike autoscale_env_v2, this env implements no physics of its own: state
transitions, queueing, capacity, and cost all come from llm_autoscaler.py
(per-pod M/M/1/K with real losses, requests/limits with CFS throttling, node
scheduling caps, startup delay, cooldown). The RL agent therefore controls
exactly the same plant as the LLM controllers and the HPA/KEDA baselines,
and the two code paths cannot drift apart.

Step semantics mirror the simulator's control loop: the agent observes the
metrics of step t, its action is requested at step t (startup delay + cooldown
enforced by ClusterState), and the reward reflects the metrics at step t+1.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from llm_autoscaler import (
    POD_RATE_GUARANTEED,
    REPLICA_MAX,
    SLA_LATENCY_MS,
    ClusterState,
    compute_metrics,
)


class AutoscaleEnvV3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trace: np.ndarray, initial_replicas: int = 3,
                 seed: int | None = None):
        super().__init__()
        self.trace = trace
        self.initial_replicas = initial_replicas

        # obs: cpu (vs limit ceiling), latency, success, rps, ready, target, pending
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(REPLICA_MAX)  # action a -> a+1 replicas

        if seed is not None:
            np.random.seed(seed)  # compute_metrics jitter uses the global RNG

    def _obs(self, m: dict) -> np.ndarray:
        c = self.cluster
        return np.array([
            min(m["cpu_pct"] / 200.0, 1.0),
            min(m["latency_p90"] / 2000.0, 1.0),
            m["success_rate"],
            min(m["rps"] / 4000.0, 1.0),
            c.ready_replicas / REPLICA_MAX,
            c.target_replicas / REPLICA_MAX,
            max(0, c.target_replicas - c.ready_replicas) / REPLICA_MAX,
        ], dtype=np.float32)

    def _metrics(self, step: int) -> dict:
        rps = int(self.trace[min(step, len(self.trace) - 1)])
        m = compute_metrics(self.cluster.ready_replicas, rps)
        m["rps"] = rps
        return m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.cluster = ClusterState(self.initial_replicas)
        self.t = 0
        self.cluster.tick(0)
        self.last_metrics = self._metrics(0)
        return self._obs(self.last_metrics), {}

    def step(self, action):
        desired = int(action) + 1
        _, scale_event = self.cluster.request_scale(desired, self.t)

        self.t += 1
        self.cluster.tick(self.t)
        m = self._metrics(self.t)
        self.last_metrics = m

        lat, succ, rps = m["latency_p90"], m["success_rate"], m["rps"]

        optimal = max(1, math.ceil(rps / POD_RATE_GUARANTEED))
        over_prov = max(0, self.cluster.ready_replicas - optimal) / REPLICA_MAX

        if lat <= SLA_LATENCY_MS and succ >= 0.95:
            reward = 1.0 - 0.3 * over_prov
        elif lat <= SLA_LATENCY_MS * 1.5:
            severity = (lat - SLA_LATENCY_MS) / (SLA_LATENCY_MS * 0.5)
            reward = 0.5 * (1.0 - severity)
        elif lat <= SLA_LATENCY_MS * 3:
            reward = -0.5 * min(lat / SLA_LATENCY_MS, 3.0) / 3.0
        else:
            reward = -1.0

        if succ < 0.95:
            reward -= min(0.5, 0.95 - succ)

        reward -= 0.02 * int(scale_event)
        reward = float(max(-2.0, min(1.0, reward)))

        done = self.t >= len(self.trace) - 1

        info = {
            "step": self.t,
            "replicas": self.cluster.target_replicas,
            "ready_replicas": self.cluster.ready_replicas,
            "latency_p90": lat,
            "cpu_pct": m["cpu_pct"],
            "success_rate": succ,
            "rps": rps,
            "vcpu_minutes": round(self.cluster.cumulative_vcpu_min, 2),
            "scale_event": int(scale_event),
        }
        return self._obs(m), reward, done, False, info
