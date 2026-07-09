# Cost-as-Objective: LLM vs RL/HPA/KEDA Autoscalers

Reframes the long-simulation results (hardened simulator, `results/long_sim/`, Alibaba
trace, 1440 steps = 24 h) so **cost is a first-class objective alongside the latency SLA**.
Regenerate with `python plot_business_case.py`.

## Cost model & pricing assumption

- Every controller is charged identically: **0.25 vCPU per ready replica per step**,
  1 step = 1 minute → 1440 steps = 24 h of operation. (Verified from the CSVs; RL,
  HPA/KEDA, and LLM runs all use the same basis, so cost is directly comparable.)
- Dollar figures assume **$0.04 / vCPU-hour** (AWS general-purpose on-demand, mid-range),
  annualized from the 24 h run. Per **single** service; multiply by fleet size for a
  company-wide figure. Change `VCPU_HOUR_USD` in the script to re-price.

## RL baselines are reported over 5 seeds

The DQN and PPO baselines are trained on the hardened simulator (500K steps each) and
evaluated over **5 seeds**; the tables below use the medoid seed (the run closest to the
per-algorithm mean), and `plots/rl_seed_variance.csv` carries the mean ± std. This matters:

| Algo | SLA | Annual $/service | Scaling actions |
|---|---|---|---|
| DQN | 99.9 ± 0.1% | $381 ± 15 | **183 ± 25** |
| PPO | 99.1 ± 0.4% | $468 ± 35 | 12 ± 5 |

**DQN's stability is seed-dependent** — scale count ranges 145–207 across seeds. An earlier
single-seed run reported 65 scales at $695; that was an unrepresentative draw. The honest
central estimate is ~183 scales at ~$381.

## Figures

| File | Shows |
|---|---|
| `plots/20_business_latency_cost.png` | **The headline two-panel:** latency-SLA (left) and annualized cost (right), one representative config per model + baselines. |
| `plots/21_business_cost_stability.png` | Cost vs. scaling-churn at matched SLA (≥99%) — where the differentiation lives. |
| `plots/22_business_stability.png` | Scaling actions over 24 h. |
| `plots/summary_business_case.csv` | Full per-config table (SLA, latency, cost vCPU + $, scale events). |
| `plots/rl_seed_variance.csv` | Per-algorithm mean ± std over 5 seeds (DQN, PPO). |

## The two business regimes

**A) Performance-critical (latency-sensitive product)** — minimize cost s.t. SLA ≥ 99.5%.
→ **DQN wins: $382/yr, 99.9% SLA, 19% cheaper than HPA.** Best LLM at matched SLA:
Qwen-80B/domain at $445.

**B) Cost-&-stability-sensitive (large fleet / SRE-constrained)** — minimize scaling churn
s.t. SLA ≥ 99% and cost ≤ HPA. → **PPO wins: 16 scaling actions, 99.4% SLA, $457/yr.**
Best LLM: Mistral/zero_shot at 166 scales.

**The LLMs win neither regime.** Read the honest framing below before quoting these.

## Honest framing (state this in the paper)

On the (cost, SLA, stability) axes, the **trained RL baselines win both regimes** — DQN is
cheapest at matched SLA, PPO is the most stable — and HPA/KEDA sit on the (cost, SLA)
frontier at 100% SLA. The LLMs do **not** dominate on cost or stability. Their defensible
value is different:

1. **Competitive zero-shot.** The LLM controllers match trained RL and HPA on SLA and land
   within ~15–20% on cost **with no workload-specific training and no reward tuning**. DQN
   and PPO reach their numbers only after **500K training steps on this exact trace**; move
   the workload and they must be retrained, while the LLM controller does not.
2. **Lower operational variance.** DQN's cost/stability edge is **seed-unstable** (145–207
   scaling actions depending on the training seed; the earlier single-seed 65 was an
   outlier). The zero-shot LLM controllers have no training-seed dependence.

The claim to make is *competitive-without-training + low-variance*, **not** cost or
stability dominance. Config choice still matters: `history_5` thrashes (Qwen/history_5 =
911 actions) and Llama-8B over-provisions (2.5–3.6× HPA cost) — present the frontier, not
a cherry-pick.

## Caveats

- Cost is comparable **within** the hardened-simulator results (same 0.25 vCPU basis); do
  not compare $ against pre-rework runs (0.5 vCPU basis).
- RL baselines are 5-seed (medoid for the trajectory plots, mean ± std in the tables);
  LLM and HPA/KEDA runs are deterministic given the trace.
- All 28 LLM/baseline configs are included (4 core models × 4 variants + gpt-oss-120b × 4 +
  llama4-scout × 4 + HPA/KEDA/DQN/PPO). `plot_business_case.py` auto-includes only complete
  1440-step runs.
