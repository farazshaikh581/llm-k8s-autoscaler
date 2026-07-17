# business_case_richer/

Real-cluster business-case figures from `results_richer/` (the N=3-replicated,
two-trace sweep: `cpu_bursty` + `wiki_diurnal`). Generated via
`plot_business_case_real.py` and `plot_percentiles_real.py` against
`results_richer/{cpu_bursty,wiki_diurnal}`.

**Differs from `../business_case_real/`**, which holds the earlier
`results_k8s_v2` figures (single rep, no `gpt-oss-120b`, no wiki_diurnal
breakout). Do not conflate the two — file naming here always ends in
`_bursty` or `_wiki` to make the source trace explicit; the old directory's
files end in plain `_cpu`/`_io`.

Snapshot as of 2026-07-17 09:20 UTC: cpu_bursty is 8/9 controllers at full
N=3 (Qwen3-80B partial, being retired); wiki_diurnal is partial (see
`summary_wiki_diurnal.csv` / `percentiles_wiki_diurnal.csv` for exact reps
completed per config). Note: GPT-OSS-120B's `cot` variant on cpu_bursty
shows a reproducible scale-oscillation bug (P99 ~3.8s, 63% success) across
all 3 reps — visible as the extreme outlier in the ladder/tail-ratio charts.
