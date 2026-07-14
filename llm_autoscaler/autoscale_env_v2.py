"""Realistic Kubernetes autoscaling Gym environment with proper queuing theory.

Models a multi-node K8s cluster with:
  - M/M/c queuing (Erlang-C) for latency estimation
  - Node-level bin-packing with CPU/memory/network contention
  - Stochastic pod lifecycle: log-normal startup, OOM kills, evictions
  - Memory pressure: working set grows with RPS, GC pauses spike latency
  - Network I/O saturation on shared node NICs
  - Metrics observation lag (agent sees 15-30s old data)
  - Connection pool limits with TCP backpressure
  - Cold start penalty for newly ready pods
  - Non-linear horizontal scaling overhead
  - Cascading failures under extreme load
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class PodState:
    node_id: int
    created_step: int
    ready_step: int
    cpu_millicores: int = 500
    memory_mib: int = 512
    is_ready: bool = False
    is_warm: bool = False
    warmup_remaining: int = 0
    rss_mib: float = 128.0
    oom_killed: bool = False
    requests_served: int = 0


@dataclass
class NodeState:
    node_id: int
    total_cpu_m: int
    total_memory_mib: int
    total_bandwidth_mbps: float
    pods: list = field(default_factory=list)

    @property
    def used_cpu_m(self):
        return sum(p.cpu_millicores for p in self.pods if not p.oom_killed)

    @property
    def used_memory_mib(self):
        return sum(p.rss_mib for p in self.pods if not p.oom_killed)

    @property
    def available_cpu_m(self):
        return self.total_cpu_m - self.used_cpu_m

    @property
    def available_memory_mib(self):
        return self.total_memory_mib - self.used_memory_mib

    @property
    def pod_count(self):
        return sum(1 for p in self.pods if not p.oom_killed)

    def can_schedule(self, cpu_m: int, mem_mib: int) -> bool:
        return self.available_cpu_m >= cpu_m and self.available_memory_mib >= mem_mib


def erlang_c(c: int, offered_load: float) -> float:
    """Erlang-C: probability an arriving customer waits in an M/M/c queue."""
    if c <= 0 or offered_load <= 0:
        return 0.0
    rho = offered_load / c
    if rho >= 1.0:
        return 1.0

    log_num = c * math.log(offered_load) - math.lgamma(c + 1) - math.log(1 - rho)
    terms = [k * math.log(offered_load) - math.lgamma(k + 1) for k in range(c)]
    terms.append(log_num)
    mx = max(terms)
    log_den = mx + math.log(sum(math.exp(t - mx) for t in terms))
    return math.exp(log_num - log_den)


class AutoscaleEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: np.ndarray,
        num_nodes: int = 2,
        node_cpu_m: int = 3000,
        node_memory_mib: int = 16384,
        node_bandwidth_mbps: float = 1000.0,
        pod_cpu_request_m: int = 500,
        pod_memory_request_mib: int = 512,
        pod_memory_limit_mib: int = 1024,
        capacity_per_replica: int = 200,
        service_time_ms: float = 8.0,
        sla_latency_ms: float = 200.0,
        max_replicas: int = 20,
        cooldown_steps: int = 1,
        metrics_lag_steps: int = 1,
        connection_pool_per_pod: int = 100,
        request_timeout_ms: float = 5000.0,
        bytes_per_request: int = 4096,
        warmup_steps: int = 3,
        gc_threshold_mib: float = 768.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.trace = trace
        self.num_nodes = num_nodes
        self.node_cpu_m = node_cpu_m
        self.node_memory_mib = node_memory_mib
        self.node_bandwidth_mbps = node_bandwidth_mbps
        self.pod_cpu_req = pod_cpu_request_m
        self.pod_mem_req = pod_memory_request_mib
        self.pod_mem_limit = pod_memory_limit_mib
        self.capacity = capacity_per_replica
        self.service_time = service_time_ms
        self.sla = sla_latency_ms
        self.max_replicas = max_replicas
        self.cooldown_steps = cooldown_steps
        self.metrics_lag = metrics_lag_steps
        self.conn_pool = connection_pool_per_pod
        self.req_timeout = request_timeout_ms
        self.bytes_per_req = bytes_per_request
        self.warmup_steps = warmup_steps
        self.gc_threshold = gc_threshold_mib

        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(max_replicas)

        self._rng = np.random.default_rng(seed)
        self._metrics_buffer: deque = deque(maxlen=max(metrics_lag_steps + 1, 2))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_idx = 0
        self.cumulative_vcpu = 0.0
        self.cumulative_cost = 0.0
        self._last_scale_step = -self.cooldown_steps - 1
        self._pending_pods: list[PodState] = []

        self.nodes = [
            NodeState(i, self.node_cpu_m, self.node_memory_mib, self.node_bandwidth_mbps)
            for i in range(self.num_nodes)
        ]

        self._schedule_initial_pods(3)
        self._metrics_buffer.clear()
        initial = self._compute_full_metrics(int(self.trace[0]))
        for _ in range(self.metrics_lag + 1):
            self._metrics_buffer.append(initial)

        return self._obs_from_metrics(initial), {}

    def _schedule_initial_pods(self, count: int):
        for i in range(count):
            node = self.nodes[i % self.num_nodes]
            pod = PodState(
                node_id=node.node_id,
                created_step=0,
                ready_step=0,
                cpu_millicores=self.pod_cpu_req,
                memory_mib=self.pod_mem_req,
                is_ready=True,
                is_warm=True,
                warmup_remaining=0,
                rss_mib=self.pod_mem_req * 0.25,
            )
            node.pods.append(pod)

    @property
    def _all_pods(self) -> list[PodState]:
        pods = []
        for n in self.nodes:
            pods.extend(p for p in n.pods if not p.oom_killed)
        return pods

    @property
    def _ready_pods(self) -> list[PodState]:
        return [p for p in self._all_pods if p.is_ready]

    @property
    def _ready_count(self) -> int:
        return len(self._ready_pods)

    @property
    def _total_pods(self) -> int:
        return len(self._all_pods)

    def _schedule_pod(self, step: int) -> bool:
        startup_steps = max(1, int(self._rng.lognormal(mean=0.7, sigma=0.5)))
        startup_steps = min(startup_steps, 8)

        candidates = sorted(self.nodes, key=lambda n: n.pod_count)
        for node in candidates:
            if node.can_schedule(self.pod_cpu_req, self.pod_mem_req):
                pod = PodState(
                    node_id=node.node_id,
                    created_step=step,
                    ready_step=step + startup_steps,
                    cpu_millicores=self.pod_cpu_req,
                    memory_mib=self.pod_mem_req,
                    is_ready=False,
                    is_warm=False,
                    warmup_remaining=self.warmup_steps,
                    rss_mib=self.pod_mem_req * 0.1,
                )
                node.pods.append(pod)
                self._pending_pods.append(pod)
                return True
        return False

    def _remove_pod(self):
        ready = self._ready_pods
        if not ready:
            return
        victim = max(ready, key=lambda p: p.created_step)
        for node in self.nodes:
            if victim in node.pods:
                node.pods.remove(victim)
                return

    def _advance_pod_lifecycle(self, step: int, rps_per_pod: float):
        newly_ready = []
        for pod in self._pending_pods[:]:
            if step >= pod.ready_step:
                pod.is_ready = True
                self._pending_pods.remove(pod)
                newly_ready.append(pod)

        for pod in self._all_pods:
            if not pod.is_ready:
                continue

            base_rss = self.pod_mem_req * 0.25
            load_rss = rps_per_pod * 0.15
            leak = pod.requests_served * 0.0001
            pod.rss_mib = min(base_rss + load_rss + leak + self._rng.normal(0, 5),
                              self.pod_mem_limit * 1.1)

            if pod.rss_mib > self.pod_mem_limit:
                pod.oom_killed = True
                pod.is_ready = False

            if pod.warmup_remaining > 0:
                pod.warmup_remaining -= 1
                if pod.warmup_remaining == 0:
                    pod.is_warm = True

    def _compute_full_metrics(self, rps: int) -> dict:
        ready = self._ready_pods
        c = len(ready)
        if c == 0:
            return {
                "cpu_pct": 100.0, "latency_p90": self.req_timeout,
                "success_rate": 0.5,
                "rps": rps, "replicas": 0, "ready_replicas": 0,
                "memory_pct": 100.0, "network_pct": 100.0,
                "vcpu_minutes": self.cumulative_vcpu,
            }

        total_capacity = c * self.capacity
        warm_count = sum(1 for p in ready if p.is_warm)
        cold_count = c - warm_count
        effective_capacity = warm_count * self.capacity + cold_count * (self.capacity * 0.4)
        rho = rps / max(effective_capacity, 1)

        cpu_pct = min(rho * 100.0, 100.0)

        overhead_factor = 1.0 + 0.02 * max(0, c - 5)

        total_conn = c * self.conn_pool
        conn_utilization = rps / max(total_conn, 1)

        rps_per_pod = rps / c
        bandwidth_per_node = {}
        for node in self.nodes:
            node_pods = [p for p in node.pods if p.is_ready and not p.oom_killed]
            if node_pods:
                node_rps = rps_per_pod * len(node_pods)
                bw_mbps = (node_rps * self.bytes_per_req * 8) / 1e6
                bandwidth_per_node[node.node_id] = min(bw_mbps / node.total_bandwidth_mbps, 1.0)

        max_bw_util = max(bandwidth_per_node.values()) if bandwidth_per_node else 0.0

        avg_mem_pct = 0.0
        gc_pods = 0
        for pod in ready:
            mem_ratio = pod.rss_mib / self.pod_mem_limit
            avg_mem_pct += mem_ratio
            if pod.rss_mib > self.gc_threshold:
                gc_pods += 1
        avg_mem_pct = (avg_mem_pct / c) * 100.0

        mu = 1.0 / (self.service_time / 1000.0)
        offered_load_per_server = (rps / 60.0) / mu
        total_offered = offered_load_per_server * c if c > 0 else 0

        if rho < 1.0 and c > 0:
            total_arrival = rps / 60.0
            offered = total_arrival / mu
            ec = erlang_c(c, offered)
            mean_wait = ec / (c * mu * (1 - rho))
            p90_wait = mean_wait * math.log(10 * ec + 1) if ec > 0.01 else mean_wait * 0.5
            latency_base = self.service_time + p90_wait * 1000
        else:
            excess = min(rho - 1.0, 1.0)
            latency_base = self.service_time * (1.0 + 15.0 * excess + 20.0 * excess**2)

        latency = latency_base * overhead_factor

        if gc_pods > 0:
            gc_fraction = gc_pods / c
            gc_pause_ms = self._rng.exponential(15.0) * gc_fraction
            latency += gc_pause_ms

        if max_bw_util > 0.7:
            bw_penalty = 1.0 + 2.0 * (max_bw_util - 0.7) / 0.3
            latency *= bw_penalty

        if conn_utilization > 0.8:
            conn_penalty = 1.0 + 3.0 * (conn_utilization - 0.8) / 0.2
            latency *= conn_penalty

        cold_penalty = 1.0 + 0.3 * (cold_count / c) if c > 0 else 1.0
        latency *= cold_penalty

        jitter = max(0.85, self._rng.normal(1.0, 0.06))
        latency *= jitter
        latency = max(self.service_time, min(latency, self.req_timeout))

        success = 1.0
        if rho > 1.0:
            excess = min(rho - 1.0, 1.5)
            overload_drop = 0.15 * excess + 0.1 * excess**2
            success -= overload_drop

        if conn_utilization > 1.0:
            conn_excess = min(conn_utilization - 1.0, 1.0)
            success -= 0.1 * conn_excess

        if latency > self.req_timeout * 0.6:
            timeout_frac = (latency - self.req_timeout * 0.6) / (self.req_timeout * 0.4)
            timeout_frac = min(timeout_frac, 1.0)
            success -= 0.15 * timeout_frac

        success = max(0.5, min(1.0, success))

        for pod in ready:
            pod.requests_served += int(rps_per_pod)

        self.cumulative_vcpu += c * (self.pod_cpu_req / 1000.0)

        return {
            "cpu_pct": round(cpu_pct, 1),
            "latency_p90": round(latency, 1),
            "success_rate": round(success, 4),
            "rps": rps,
            "replicas": self._total_pods,
            "ready_replicas": c,
            "memory_pct": round(avg_mem_pct, 1),
            "network_pct": round(max_bw_util * 100, 1),
            "vcpu_minutes": round(self.cumulative_vcpu, 2),
        }

    def _obs_from_metrics(self, metrics: dict) -> np.ndarray:
        lagged = self._metrics_buffer[0] if self._metrics_buffer else metrics
        return np.array([
            min(lagged["cpu_pct"] / 100.0, 1.0),
            min(lagged["latency_p90"] / 2000.0, 1.0),
            lagged["success_rate"],
            min(lagged["rps"] / 4000.0, 1.0),
            lagged["ready_replicas"] / self.max_replicas,
            min(lagged["memory_pct"] / 100.0, 1.0),
            min(lagged["network_pct"] / 100.0, 1.0),
            len(self._pending_pods) / max(self.max_replicas, 1),
        ], dtype=np.float32)

    def step(self, action):
        desired = int(action) + 1
        current = self._total_pods
        scale_event = 0

        if desired != current and (self.step_idx - self._last_scale_step) >= self.cooldown_steps:
            if desired > current:
                scheduled = 0
                for _ in range(desired - current):
                    if self._schedule_pod(self.step_idx):
                        scheduled += 1
                if scheduled > 0:
                    scale_event = 1
                    self._last_scale_step = self.step_idx
            elif desired < current:
                for _ in range(current - desired):
                    self._remove_pod()
                scale_event = 1
                self._last_scale_step = self.step_idx

        rps = int(self.trace[min(self.step_idx, len(self.trace) - 1)])
        rps_per_pod = rps / max(self._ready_count, 1)

        self._advance_pod_lifecycle(self.step_idx, rps_per_pod)

        oom_count = sum(1 for n in self.nodes for p in n.pods if p.oom_killed)
        for node in self.nodes:
            node.pods = [p for p in node.pods if not p.oom_killed]

        metrics = self._compute_full_metrics(rps)
        self._metrics_buffer.append(metrics)

        optimal = max(1, math.ceil(rps / self.capacity))
        ready = self._ready_count
        over_prov = max(0, ready - optimal) / self.max_replicas

        lat = metrics["latency_p90"]
        succ = metrics["success_rate"]

        if lat <= self.sla and succ >= 0.95:
            reward = 1.0 - 0.3 * over_prov
        elif lat <= self.sla * 1.5:
            severity = (lat - self.sla) / (self.sla * 0.5)
            reward = 0.5 * (1.0 - severity)
        elif lat <= self.sla * 3:
            reward = -0.5 * min(lat / self.sla, 3.0) / 3.0
        else:
            reward = -1.0

        if succ < 0.95:
            reward -= min(0.5, (0.95 - succ))

        reward -= 0.02 * scale_event

        if oom_count > 0:
            reward -= min(0.3, 0.15 * oom_count)

        reward = max(-2.0, min(1.0, reward))

        self.step_idx += 1
        done = self.step_idx >= len(self.trace)

        obs = self._obs_from_metrics(metrics)

        return obs, reward, done, False, {
            "cpu_pct": metrics["cpu_pct"],
            "latency_p90": metrics["latency_p90"],
            "success_rate": metrics["success_rate"],
            "replicas": metrics["replicas"],
            "ready_replicas": metrics["ready_replicas"],
            "rps": rps,
            "vcpu_minutes": metrics["vcpu_minutes"],
            "memory_pct": metrics["memory_pct"],
            "network_pct": metrics["network_pct"],
            "oom_kills": oom_count,
            "pending_pods": len(self._pending_pods),
            "scale_event": scale_event,
        }
