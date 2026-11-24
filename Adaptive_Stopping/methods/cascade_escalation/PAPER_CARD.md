# Paper implementation card: Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades

- **Paper / OpenReview:** Dylan Bouchard, ICML 2026 AdaptFM workshop. arXiv: <https://arxiv.org/abs/2605.06350> (implemented against v2, 2026-10-05). OpenReview forum: _TODO: copy the forum link from the [official accepted-paper list](https://adaptfm.gitlab.io/accepted-papers); it could not be fetched from the build sandbox._ Note: the upstream repository README uses the earlier title "Is Escalation Worth It? On the Depth of LLM Cascades"; both titles refer to the same arXiv ID.
- **Primary module:** `Adaptive_Stopping/` (a cascade is a per-query stopping rule over a sequence of models: stop at the first model whose confidence clears its threshold).
- **Secondary modules:** `Performance_Metrics/` (cost/quality frontiers, budget-constrained evaluation); `Backbones/` (model pool); `Dataloaders/` (benchmark prompts).
- **Owner / reviewer:** @Jigsaw442
- **Status:** implementing
- **Upstream code / licence:** <https://github.com/dylanbouchard/llm-cascade-frontiers> at commit `3872d8a`, Apache-2.0. Upstream ships scripts and cached numerical results only, not raw model responses. Its `src/data/prompts/*.parquet` files contain the paper's exact MMLU, TriviaQA and MATH prompts. Our code is a clean reimplementation; no upstream code is copied.
- **Backbone IDs:** `llm.qwen2.5-0.5b-instruct`, `llm.qwen2.5-1.5b-instruct`, `llm.qwen2.5-3b-instruct`, `llm.qwen2.5-7b-instruct`, `llm.phi-3.5-mini-instruct` (proposed IDs; Hugging Face repos `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` and `microsoft/Phi-3.5-mini-instruct`). _TODO: pin revisions once the GPU run records them._
- **Dataset IDs and split:** MMLU, TriviaQA and MATH (levels 3 to 5), using the paper's exact upstream prompt parquet files (SHA-256 checked against `llm-cascade-frontiers@3872d8a`). 2,000 queries per benchmark, subsampled with seed 2026. The paper's SimpleQA and LiveCodeBench are skipped because they need an LLM judge or code execution. Each benchmark is split 50 times into stratified calibration/test halves (seeds 42 to 91).
- **Hardware assumptions:** generation on 1x NVIDIA RTX 4090 (24 GB), bf16, greedy decoding, top-15 log-probs, one model resident at a time; the offline policy search is CPU-only NumPy and needs no GPU.

## Team

Team: [Jigsaw442](https://github.com/Jigsaw442) (fork owner), [Boxbob](https://github.com/Boxbob), [paraskatiyar](https://github.com/paraskatiyar), [Sumit-Singatia-2027](https://github.com/Sumit-Singatia-2027).

## Research summary

### Problem

LLM cascades answer each query with a cheap model first and escalate to a more expensive model only when the cheap model looks unsure. They are widely used to cut serving cost, but it is unclear when escalation actually beats simply picking the best single model for the budget, and whether deeper cascades (three or more models) are worth their extra calls. The paper gives a decision-theoretic answer and tests it empirically.

### Method

- **Policy.** A cascade is an ordered sequence of models `(m_1, ..., m_d)`, cheapest first, with thresholds `(tau_1, ..., tau_{d-1})`. Model `m_i` answers and emits a confidence score `s_i(x)` from the same generation call. The query stops at the first stage with `s_i(x) >= tau_i`, otherwise it escalates; the last model always answers. Cost is the sum of the costs of every model called.
- **Confidence scores.** Five white-box scores computed from the top-K token log-probabilities of the answer, with no extra model call: length-normalised probability, minimum token probability, probability margin, mean normalised negentropy and minimum normalised negentropy (the paper computes them through UQLM).
- **Theory.** Escalation beats a random router at the same cost exactly when the score ranks the cheap model's errors above its successes in a way that also correlates with the expensive model being right (the paper's Prop. 3.2 condition). Under concavity of the cost/accuracy frontier, extra depth adds little once the best depth-2 cascade is chosen.
- **Search protocol.** Policy classes `S_1 ⊂ S_2 ⊂ ... ⊂ S_5`, where `S_d` is every ordered model subsequence of length at most `d` with every threshold vector. Thresholds are searched exhaustively on calibration score partitions. For each of 500 cost budgets the best calibration policy within budget is frozen and evaluated on the held-out test split, over 50 calibration/test splits.

### Expected behaviour

Paper-reported (Table 1, mean-negentropy score; not yet re-measured here): the best depth-2 cascade beats the best single model (`S_2 - S_1`) by +3.93 pp on MMLU, +5.72 pp on TriviaQA, +2.80 pp on MATH, +8.22 pp on SimpleQA and +7.01 pp on LiveCodeBench. Going deeper rarely helps: `S_3 - S_2` is negative in 38 of 45 cases.

Locally we expect the same qualitative picture with open models: `S_2 > S_1` at mid-range budgets, little or no `S_3+` gain, and escalation beating the random router only where the score has useful ranking power (AUROC well above 0.5).

## Implementation

- **Input contract:** per-query arrays over `N` queries and `M` models: correctness `U[N, M]` in {0, 1}, cost `C[N, M]` (float, units recorded in `cost_unit`), and confidence scores `S[score_name][N, M]` (float, higher = more confident). Optional measured latency `L[N, M]` in ms.
- **Output contract:** per (benchmark, score, policy class, split seed, budget): the frozen policy (model sequence + thresholds), test accuracy, test mean cost, escalation rate, mean models called, effective depth. Rows follow the schema in [`README_adaptiveStopping.md`](../../README_adaptiveStopping.md#llm-cascade-escalation-bouchard-2026).
- **Configuration:** `Adaptive_Stopping/experiments/cascade_escalation/configs/qwen25_phi35_pool.yaml`
- **Reproduction command:**
  1. GPU generation (RTX 4090): `bash Adaptive_Stopping/experiments/cascade_escalation/run_gpu.sh` writes `results/cascade/generations/<dataset>/<slug>.jsonl.gz`, `results/cascade/meta/<slug>.json` and the live batch-size-1 run under `results/cascade/live/`.
  2. Offline sweep (CPU): `python Adaptive_Stopping/experiments/cascade_escalation/run_sweep.py --config Adaptive_Stopping/experiments/cascade_escalation/configs/qwen25_phi35_pool.yaml --results results/cascade` writes `results/cascade/sweep/*.csv`, plots, and the summary `Performance_Metrics/results/cascade_escalation.csv`.
- **Tests:** `pytest Adaptive_Stopping/tests -q`

Code layout (on branch `cascade-escalation`):

| Path | Contents |
|---|---|
| `methods/cascade_escalation/scores.py` | Five log-prob confidence scores. |
| `methods/cascade_escalation/policy.py` | `CascadePolicy`, vectorised `evaluate_policy`, and `CascadeStoppingPolicy` (the module's `StoppingPolicy` contract for live cascades). |
| `methods/cascade_escalation/search.py` | Threshold grids, sequence enumeration, Pareto frontier, 500-budget grid, select-and-evaluate. |
| `methods/cascade_escalation/theory.py` | AUROC, Prop. 3.2 check, frontier slopes, concavity and stage-wise dominance diagnostics. |
| `methods/cascade_escalation/learned_scores.py` | Optional learned (logistic / ridge) scores on top of the five base scores. |
| `baselines/cascade_baselines.py` | Always-model (always-small, always-large) and random-router baselines. |

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | `S_2 - S_1`: MMLU +3.93, TriviaQA +5.72, MATH +2.80 pp | not reported | not reported | not reported | not reported | Table 1, mean-negentropy; API models; cost in USD; mean over 50 splits |
| Local baseline (always-small, Qwen2.5-0.5B) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | not_measured | RTX 4090, bf16 |
| Local baseline (always-large, Qwen2.5-7B) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | not_measured | RTX 4090, bf16 |
| Local baseline (random router) | _TODO_ | _TODO_ | n/a | n/a | not_measured | expected-value line between small and large |
| Local method (best `S_2`) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | not_measured | best score per benchmark |
| Local method (best `S_5`) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | not_measured | |

## Differences and limitations

- **Models.** The paper uses a pool of commercial API models; we use five open models that fit on one 24 GB GPU (Qwen2.5-Instruct 0.5B/1.5B/3B/7B and Phi-3.5-mini). Absolute accuracies and the shape of the frontier will differ.
- **Cost.** The paper uses API token prices. We have no prices, so the primary cost is a FLOP proxy, `2 · N_params · (n_in + n_out)`, and the secondary cost is measured GPU seconds (batch wall time apportioned to queries by `n_out + 1`). Results are reported under both.
- **Scores.** Our five scores follow the paper's definitions directly rather than calling UQLM; small numeric differences are possible (for example empty responses).
- **Search.** Depth-2 search is exhaustive over calibration partitions, as in the paper. Depth 3 to 5 uses a quantile grid per stage plus the never/always sentinels instead of the paper's exhaustive search with pruning, so `S_{d-1} ⊂ S_d` still holds exactly but deep classes may be slightly under-optimised. The grid sizes are recorded with each result.
- **Benchmarks.** SimpleQA and LiveCodeBench are not run (they need an LLM judge or code execution), and each benchmark is subsampled to 2,000 queries.
- **Not implemented.** The paper's embedding-based scorers and its FOC local-search optimiser.
- **Responses.** Upstream does not release raw responses, so every response is regenerated locally; results depend on our decoding settings and grader.
- **Energy** is not measured.

## Reproduction checklist

- [ ] Exact paper link and version verified (arXiv v2 done; OpenReview link TODO)
- [ ] Base model and dataset revisions pinned
- [ ] Baseline runs on the same setup
- [ ] Tests and seed handling added (split seeds 42 to 91)
- [ ] Resource and quality metrics reported
- [x] Licence/data terms checked (upstream Apache-2.0; dataset records keep their original licences; Qwen2.5 Apache-2.0 except 3B under the Qwen Research licence; Phi-3.5-mini MIT)
