"""Gym environment for Kubernetes autoscaling — same simulation model as llm_autoscaler.py.

Observation: [cpu_pct, latency_p90_norm, success_rate, rps_norm, current_replicas_norm]
Action: discrete 0-19 → set replicas to action+1
Reward: +1 if SLA met, -1 if violated, -0.1 * over_provisioning penalty
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AutoscaleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: np.ndarray,
        capacity_per_replica: int = 200,
        service_time_ms: float = 8.0,
        sla_latency_ms: float = 200.0,
        max_replicas: int = 20,
        startup_delay_steps: int = 1,
        cooldown_steps: int = 1,
    ):
        super().__init__()
        self.trace = trace
        self.capacity = capacity_per_replica
        self.service_time = service_time_ms
        self.sla = sla_latency_ms
        self.max_replicas = max_replicas
        self.startup_delay = startup_delay_steps
        self.cooldown_steps = cooldown_steps

        # obs: cpu_pct, latency_norm, success_rate, rps_norm, replicas_norm
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(max_replicas)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.replicas = 3
        self.ready_replicas = 3
        self._pending_scale = None
        self._last_scale_step = -self.cooldown_steps - 1
        self.cumulative_vcpu = 0.0
        return self._get_obs(), {}

    def _compute_metrics(self, ready: int, rps: int):
        c = max(ready, 1)
        total = c * self.capacity
        rho = min(rps / max(total, 1), 2.0)
        cpu_pct = min(rho * 100, 100)

        if rho <= 1.0:
            qf = (rho / (1.0 - rho + 0.1)) ** 1.5
            lat = self.service_time * (1.0 + qf)
        else:
            lat_cap = self.service_time * (1.0 + (1.0 / 0.1) ** 1.5)
            lat = lat_cap * math.exp(3.0 * (rho - 1.0))
        lat = max(self.service_time, min(lat, 10000))

        if rho <= 1.0:
            success = 1.0
        elif rho <= 1.3:
            success = 1.0 - 0.5 * (rho - 1.0)
        else:
            success = max(0.3, 0.85 - 0.5 * (rho - 1.3))

        return cpu_pct, lat, success

    def _get_obs(self):
        rps = int(self.trace[min(self.step_idx, len(self.trace) - 1)])
        cpu, lat, succ = self._compute_metrics(self.ready_replicas, rps)
        return np.array([
            cpu / 100.0,
            min(lat / 1000.0, 1.0),
            succ,
            min(rps / 4000.0, 1.0),
            self.ready_replicas / self.max_replicas,
        ], dtype=np.float32)

    def step(self, action):
        desired = int(action) + 1  # action 0 → 1 replica, action 19 → 20

        # apply scaling with cooldown + startup delay
        if desired != self.replicas and (self.step_idx - self._last_scale_step) >= self.cooldown_steps:
            self.replicas = desired
            self._last_scale_step = self.step_idx
            if desired > self.ready_replicas:
                self._pending_scale = (desired, self.step_idx + self.startup_delay)
            else:
                self.ready_replicas = desired

        if self._pending_scale:
            target, ready_at = self._pending_scale
            if self.step_idx >= ready_at:
                self.ready_replicas = target
                self._pending_scale = None

        rps = int(self.trace[self.step_idx])
        cpu, lat, succ = self._compute_metrics(self.ready_replicas, rps)

        # reward
        optimal = max(1, math.ceil(rps / self.capacity))
        over_prov = max(0, self.ready_replicas - optimal) / self.max_replicas

        if lat <= self.sla:
            reward = 1.0 - 0.3 * over_prov  # SLA met, penalize waste
        else:
            reward = -1.0 - 0.5 * (lat / self.sla)  # SLA violated, proportional penalty

        self.cumulative_vcpu += self.ready_replicas * 0.5  # 500m per pod per step
        self.step_idx += 1
        done = self.step_idx >= len(self.trace)

        return self._get_obs(), reward, done, False, {
            "cpu_pct": cpu, "latency_p90": lat, "success_rate": succ,
            "replicas": self.ready_replicas, "rps": rps,
            "vcpu_minutes": self.cumulative_vcpu,
        }
