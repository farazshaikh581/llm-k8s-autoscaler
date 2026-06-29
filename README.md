# From Training to Reasoning: LLMs as Closed-Loop Controllers for Cloud Autoscaling

A multi-model empirical study evaluating large language models as closed-loop autoscalers for Kubernetes workloads, benchmarked against traditional autoscalers (HPA, KEDA) and reinforcement learning baselines (DQN, PPO).

## Overview

This repository contains the code, data, and results for the paper:

> **From Training to Reasoning: LLMs as Closed-Loop Controllers for Cloud Autoscaling**

We evaluate whether LLMs can make real-time scaling decisions for containerized workloads without any task-specific training, using only natural language prompts describing the current system state. The study covers:

- **4 core LLMs**: Llama 3.1 8B, Llama 3.3 70B, Mistral Small 4 (119B), Qwen 3 80B
- **2 supplementary LLMs**: GPT-OSS 120B, Llama 4 Scout (17B)
- **4 prompt strategies**: Zero-Shot, Domain-Enriched, History-Augmented (5-step), Chain-of-Thought
- **4 baselines**: HPA (CPU-based), KEDA (RPS-based), DQN, PPO
- **2 experiment types**: High-fidelity simulation (1440 steps) and real Kubernetes cluster (120 steps x 60s)
- **2 workload types**: CPU-intensive and I/O-intensive (Alibaba Cluster Trace 2018)

## Key Findings

- **LLMs match or exceed traditional autoscalers** — Llama 70B with domain prompting achieves 100% SLA compliance at lower cost than HPA
- **Model size does not predict quality** — 8B parameter models outperform 80B and 120B models
- **Domain-enriched prompts are consistently best** — averaging 100% SLA across all core models
- **History augmentation causes instability** — leading to thrashing and SLA degradation, especially on larger models
- **Chain-of-thought provides no benefit** — overthinking hurts reactive scaling decisions
- **Simulation results transfer to real clusters** — with CPU workloads being the real differentiator (I/O workloads are trivially easy)

## Repository Structure

```
.
├── llm_autoscaler.py          # LLM-based autoscaler (simulation)
├── k8s_autoscaler.py          # LLM-based autoscaler (real Kubernetes cluster)
├── load_generator.py          # Trace-driven HTTP load generator for K8s
├── autoscale_env.py           # Gymnasium environment for RL training (v1)
├── autoscale_env_v2.py        # Gymnasium environment for RL training (v2)
├── train_rl.py                # DQN + PPO training and evaluation (v1)
├── train_rl_v2.py             # DQN + PPO training and evaluation (v2)
├── extract_workload_traces.py # Alibaba trace extraction (CPU/IO split)
├── analyze_results.py         # Short simulation analysis
├── plot_all.py                # Comprehensive plotting (all experiments)
├── api_keys.conf.example      # API key configuration template
├── requirements.txt           # Python dependencies
│
├── k8s/                       # Kubernetes deployment manifests
│   ├── app.py                 # Sample workload application
│   ├── Dockerfile             # Container image for workload
│   ├── deployment.yaml        # Base deployment manifest
│   └── workloads.yaml         # CPU + I/O workload definitions
│
├── scripts/                   # Experiment orchestration
│   ├── run_all.sh             # Short simulation (150 steps, all models)
│   ├── run_long.sh            # Long simulation (1440 steps, multi-provider)
│   ├── run_k8s_experiment.sh  # K8s v1 experiment runner
│   └── run_k8s_v2.sh         # K8s v2 experiment runner (4 models x 4 variants x 2 workloads)
│
├── traces/                    # Workload traces (Alibaba Cluster Trace 2018)
│   ├── trace_alibaba_v2.npy   # Combined trace (1440 steps, RPS 499-3866)
│   ├── trace_alibaba_1440.npy # Original 1440-step trace
│   ├── trace_cpu.npy          # CPU-intensive machines (RPS 200-2199)
│   └── trace_io.npy           # I/O-intensive machines (RPS 100-3351)
│
├── models/                    # Trained RL models
│   ├── v1/                    # Models for simulation experiments
│   └── v2/                    # Models for K8s experiments
│
├── results/                   # Experiment results (CSV)
│   ├── short_sim/             # 150-step simulation results
│   ├── long_sim/              # 1440-step simulation results
│   ├── k8s_v1/               # Real K8s experiment v1 (60 steps)
│   └── k8s_v2/               # Real K8s experiment v2 (120 steps, CPU + IO)
│
└── plots/                     # Generated figures and tables
    ├── *.png                  # Plot images
    ├── *.tex                  # LaTeX tables
    └── *.csv                  # Summary CSVs
```

## Setup

### Requirements

- Python 3.10+
- Free API keys from one or more LLM providers (see `api_keys.conf.example`)
- For real cluster experiments: Kubernetes cluster (tested on k3s v1.35.5)

### Installation

```bash
git clone https://github.com/<your-username>/llm-k8s-autoscaler.git
cd llm-k8s-autoscaler
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API Key Configuration

```bash
cp api_keys.conf.example api_keys.conf
# Edit api_keys.conf with your API keys
```

## Usage

### Running Simulations

**Short simulation** (150 steps, quick validation):
```bash
export GROQ_API_KEY='your-key'
python llm_autoscaler.py --model llama-8b --variant zero_shot \
    --provider nvidia --trace traces/trace_alibaba_v2.npy \
    --steps 150 --output-dir results/short_sim
```

**Long simulation** (1440 steps, full evaluation):
```bash
bash scripts/run_long.sh
```

### Running Real Kubernetes Experiments

1. Deploy the workload application:
```bash
kubectl apply -f k8s/workloads.yaml
```

2. Run the autoscaler:
```bash
export NVIDIA_API_KEY='your-key'
python k8s_autoscaler.py --workload cpu --model llama-8b --variant domain \
    --trace traces/trace_cpu.npy --steps 120 --interval 60 \
    --output-dir results/k8s_v2 --provider nvidia
```

### Training RL Baselines

```bash
python train_rl_v2.py --algo dqn --timesteps 500000
python train_rl_v2.py --algo ppo --timesteps 500000
```

### Generating Plots and Tables

```bash
python plot_all.py              # standard quality
python plot_all.py --dpi 300    # publication quality
```

This generates 15 plots, 3 LaTeX tables, and 2 summary CSVs in the `plots/` directory.

## Plots

| # | Plot | Description |
|---|------|-------------|
| 01 | Workload Traces | Alibaba trace characterization (combined, CPU, IO) |
| 02 | Summary Bars | SLA / Cost / Stability per model and variant |
| 03 | Time-Series | Replicas + latency + CPU over 1440 steps |
| 04 | Ablation Heatmap | 6 metrics across all models and prompt variants |
| 05 | Model Size Scaling | Performance vs parameter count (8B-120B) |
| 06 | Cost-SLA Pareto | Cost-efficiency frontier |
| 07 | K8s Summary | CPU vs IO workload bars |
| 08 | Workload Comparison | CPU vs IO scatter plot |
| 09 | K8s Time-Series | Real cluster replicas + latency + CPU |
| 10 | LLM Overhead | Token usage and inference latency |
| 11 | Sim vs Real | Simulation fidelity analysis |
| 12 | Radar Profiles | Multi-dimensional model comparison |
| 13 | Variant Time-Series | Prompt variant effect on scaling behavior |
| 14 | Workload Difficulty | CPU (hard) vs IO (easy) ranking |
| 15 | Latency Distributions | Box plots by model and variant |

## Infrastructure

### Simulation Environment
- Cluster: 5 worker nodes, 4 vCPU / 8 GiB each
- Pod spec: 500m CPU request, 512 MiB RAM, max 20 replicas
- Latency model: M/M/c queuing (Erlang-C)
- Startup delay: 30s for new replicas
- Scale cooldown: 60s between events

### Real Kubernetes Cluster
- k3s v1.35.5
- 1 master node (6 vCPU, 62 GiB, NoSchedule taint)
- 2 worker nodes (3 vCPU, 16 GiB each, KVM VMs)

### LLM Providers
| Provider | Models | Notes |
|----------|--------|-------|
| NVIDIA (build.nvidia.com) | Llama 8B, 70B, Mistral Small 4, Qwen 80B | Primary provider |
| Groq (console.groq.com) | Llama 4 Scout | Rate-limited |
| Cerebras (cloud.cerebras.ai) | GPT-OSS 120B | Simulation only |

## Workload Traces

The workload traces are derived from the [Alibaba Cluster Trace 2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018), specifically the `machine_usage` table (~1.3 million machines over 8 days). The original dataset is described in:

> Q. Liu, Z. Yu. **The Elasticity and Plasticity in Semi-Containerized Co-locating Cloud Workload: a View from Alibaba Trace.** *ACM SoCC 2018.* DOI: [10.1145/3267809.3267830](https://doi.org/10.1145/3267809.3267830)

**Extraction process** (`extract_workload_traces.py`):
1. Load 20M rows from `machine_usage.csv` (columns: `machine_id`, `time_stamp`, `cpu_util`, `mem_util`, `disk_io`)
2. Classify machines into CPU-intensive, memory-intensive, and I/O-intensive types using k-means clustering on per-machine resource usage statistics (mean and std of CPU, memory, and disk I/O)
3. For each machine type, aggregate utilization into 1440 one-minute bins (24 hours) and convert to RPS using a linear scaling factor
4. Output: `trace_cpu.npy` (RPS 200–2199), `trace_io.npy` (RPS 100–3351), `trace_alibaba_v2.npy` (combined, RPS 499–3866)

The raw `machine_usage.csv` is not included in this repository due to size (~30 GB compressed). Download it from the [Alibaba cluster-trace-v2018 repository](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018).

## Experiment Results

All experiment results are stored as CSV files in the `results/` directory. Each row represents one autoscaling decision step with columns for:

- `step` — time step index
- `replicas` / `ready_replicas` — target and actual replica count
- `latency_p90` — 90th percentile response latency (ms)
- `cpu_pct` — cluster CPU utilization
- `requests` — incoming requests per step
- `success_rate` — fraction of requests meeting SLA
- `vcpu_minutes` — cumulative infrastructure cost
- `scale_event` — whether a scaling action occurred
- `llm_model` / `llm_variant` — model and prompt strategy used
- `llm_tokens_used` — tokens consumed per decision
- `llm_latency_ms` — LLM inference latency

## License

MIT License. See [LICENSE](LICENSE) for details.
