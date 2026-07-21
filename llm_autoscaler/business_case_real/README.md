# business_case_real/

Figures for the real k8s cluster runs. Data comes from `results_richer/`
(3 reps, two traces: `cpu_bursty` and `wiki_diurnal`). Made by
`plot_business_case_real.py` and `plot_percentiles_real.py`.

Files ending in `_cpu_bursty` are for the cpu_bursty trace. Files ending
in `_wiki_diurnal` are for the wiki_diurnal trace.

## Data status

Both traces have full data now. All 4 core LLMs (Llama-8B, Llama-70B,
Mistral-Small, GPT-OSS-120B), all 4 prompt styles (zero_shot, domain,
history_5, cot), and all 4 baselines (HPA, KEDA, RL-PPO, RL-DQN), 3 reps
each.

Qwen3-80B is not in this set. NVIDIA dropped the hosted model, so it
never got a full run. Its old partial data is archived, not deleted, in
`results_richer/_dropped_qwen3-80b_nvidia_deprecation/`.

## Known issue: GPT-OSS-120B / cot

GPT-OSS-120B with the `cot` prompt is the worst config in both traces.
Mean latency is over 400ms (next worst is under 110ms). Only 9-11% of
steps clear the P99 SLA. It causes 88-98 scale events per run, out of
120 steps.

Cause: the `cot` prompt has no rule to limit how much it can scale per
step. Compare it to the `domain` prompt, which says "Max +/- 3 replicas
per step to avoid thrashing." The `cot` prompt just asks the model to
reason and give a final number, with no cap.

The result is the replica count swings wildly. It jumps from 1 to 14
and back, over and over. Other models (Llama-8B, Mistral) also jump to
14 under `cot`, but they mostly settle there once they land on a value.
Only GPT-OSS-120B keeps swinging back down and up again for the whole
run.

Part of the reason: k8s reports the ready replica count with a one-step
lag. Every decision is based on the result of the previous decision,
not the current state. The `cot` prompt has no memory of past decisions
and no per-step cap, so each step gets re-reasoned from scratch using
only that lagged number. When the lagged number shows overload,
GPT-OSS-120B swings hard to the max. Once it sees the max already
applied and looking fine, it swings hard back down. This repeats the
whole run.

This is separate from the max_tokens truncation bug already fixed in
`k8s_autoscaler.py` (raised success from about 55% to about 90%). The
remaining gap here is the oscillation above, a prompt design gap, not a
token limit problem.

## Ablation: LLM decision latency

See `../ablation_llm_overhead.py` and `../ablation_llm_overhead.csv` at
the top level. It measures how much of each model's real API latency
is network and provider overhead versus real compute time, using each
model's own median call as a "nothing went wrong" estimate for what
local hosting would look like.
