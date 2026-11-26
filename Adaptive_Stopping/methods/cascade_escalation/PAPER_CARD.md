# Paper implementation card: Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades

- **Paper / OpenReview:** Dylan Bouchard, ICML 2026 AdaptFM workshop. arXiv: <https://arxiv.org/abs/2605.06350> (implemented against v2, 2026-10-05). OpenReview forum: <https://openreview.net/forum?id=MfWD7bOKiF> (linked from the [official accepted-paper list](https://adaptfm.gitlab.io/accepted-papers), Poster Session 1). Note: the upstream repository README uses the earlier title "Is Escalation Worth It? On the Depth of LLM Cascades"; both titles refer to the same arXiv ID.
- **Primary module:** `Adaptive_Stopping/` (a cascade is a per-query stopping rule over a sequence of models: stop at the first model whose confidence clears its threshold).
- **Secondary modules:** `Performance_Metrics/` (cost/quality frontiers, budget-constrained evaluation); `Backbones/` (model pool); `Dataloaders/` (benchmark prompts).
- **Owner / reviewer:** @Jigsaw442
- **Status:** partial-reproduction (qualitative findings reproduced on an open-model pool; absolute numbers not comparable to the paper's API-model results)
- **Upstream code / licence:** <https://github.com/dylanbouchard/llm-cascade-frontiers> at commit `3872d8a`, Apache-2.0. Upstream ships scripts and cached numerical results only, not raw model responses. Its `src/data/prompts/*.parquet` files contain the paper's exact MMLU, TriviaQA and MATH prompts. Our code is a clean reimplementation; no upstream code is copied.
- **Backbone IDs:** `llm.qwen2.5-0.5b-instruct`, `llm.qwen2.5-1.5b-instruct`, `llm.qwen2.5-3b-instruct`, `llm.qwen2.5-7b-instruct`, `llm.phi-3.5-mini-instruct` (proposed IDs; Hugging Face repos `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` and `microsoft/Phi-3.5-mini-instruct`). Revisions used (Hub commit SHAs recorded by the run): Qwen2.5-0.5B `7ae5576`, 1.5B `989aa79`, 3B `aa8e725`, 7B `a09a354`, Phi-3.5-mini `2fe1924` (full SHAs in [`results_rtx4090/generations_meta/`](../../experiments/cascade_escalation/results_rtx4090/generations_meta/)).
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
- **Tests:** `pytest Adaptive_Stopping/tests -q` (needs numpy, pandas, pyyaml, pytest; the MATH symbolic-equivalence test also needs sympy and antlr4-python3-runtime==4.11.1, and the live-cascade tests need torch, otherwise they are skipped)

Code layout:

| Path | Contents |
|---|---|
| `methods/cascade_escalation/scores.py` | Five log-prob confidence scores. |
| `methods/cascade_escalation/policy.py` | `CascadePolicy`, vectorised `evaluate_policy`, and `CascadeStoppingPolicy` (the module's `StoppingPolicy` contract for live cascades). |
| `methods/cascade_escalation/search.py` | Threshold grids, sequence enumeration, Pareto frontier, 500-budget grid, select-and-evaluate. |
| `methods/cascade_escalation/theory.py` | AUROC, Prop. 3.2 check, frontier slopes, concavity and stage-wise dominance diagnostics. |
| `methods/cascade_escalation/learned_scores.py` | Optional learned (logistic / ridge) scores on top of the five base scores. |
| `baselines/cascade_baselines.py` | Always-model (always-small, always-large) and random-router baselines. |

## Results

Run on 1x RTX 4090 (24 GB), vLLM 0.34.0 for generation (greedy, bf16, top-15 log-probs), transformers 5.24.1 for the live batch-size-1 runs, 2026-11-24/25. The GPU run used commit `80d5fa0` (the merged code plus a teammate's `run_gpu.sh` fixes for the rented box); those changes are included in this repository's history. Offline protocol: 50 stratified calibration/test splits (seeds 42 to 91), 500 budgets per split, policy classes S1 to S5. All raw tables and plots: [`experiments/cascade_escalation/results_rtx4090/`](../../experiments/cascade_escalation/results_rtx4090/).

**Quality vs. compute (offline, cost = FLOP proxy, score = mean negentropy, test accuracy area-averaged over the budget grid, mean over 50 splits).**

| Benchmark | Always-small (Qwen2.5-0.5B) | Always-large (Qwen2.5-7B) | Random router | S1 best single model | S2 cascade | S2 − S1 (pp) | S3 − S2 (pp) | S5 − S2 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU | 40.1% | 71.0% | 54.5% | 59.7% | 62.3% | +2.54 ± 0.46 | −0.04 | −0.04 |
| TriviaQA | 0.1% | 51.2% | 23.6% | 36.8% | 40.6% | +3.75 ± 0.37 | −0.01 | −0.01 |
| MATH (L3-5) | 25.1% | 62.7% | 42.6% | 47.6% | 50.9% | +3.34 ± 0.38 | +0.002 | +0.002 |
| Paper (Table 1, API models) | | | | | | MMLU +3.93, TriviaQA +5.72, MATH +2.80 | MMLU −0.05, TriviaQA +0.05, MATH −0.08 | |

Across all 7 informative scores × 3 benchmarks × 2 cost measures (42 cases per depth), S2 − S1 ranges from +1.73 to +6.06 pp (an uninformative random score gives +1.07 to +2.92 pp, i.e. part of the gain is just mixing two models). Going deeper almost never helps: S3/S4/S5 − S2 is ≤ 0 in 122 of 126 comparisons, and the 4 positive cases are each ≤ +0.002 pp. This matches the paper's headline ("depth beyond two rarely pays").

**Measured live cascade (batch size 1, Qwen2.5-0.5B → Qwen2.5-7B, both resident, thresholds frozen from split-0 calibration at 25/50/75% of the cost range).**

| Source | Quality | Latency (median / p95) | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | S2 − S1 = +3.93 / +5.72 / +2.80 pp (MMLU / TriviaQA / MATH) | not reported | not reported | not reported | not reported | API models, USD cost |
| Local baseline: always-small, MMLU (n=200) | 38.0% | 42 / 46 ms | 19.8 q/s | 15,560 MB* | not_measured | 0.22 TFLOPs/query |
| Local baseline: always-large, MMLU | 57.5% | 54 / 77 ms | 16.9 q/s | 15,656 MB* | not_measured | 3.36 TFLOPs/query |
| Local method: cascade@0.75, MMLU | 58.5% (79% escalated) | 96 / 128 ms | 10.2 q/s | 15,656 MB | not_measured | 3.02 TFLOPs/query |
| Local baseline: always-large, TriviaQA | 49.5% | 93 / 187 ms | 9.6 q/s | 15,550 MB* | not_measured | |
| Local method: cascade@0.75, TriviaQA | 27.5% (69% escalated) | 466 / 959 ms | 2.0 q/s | 15,546 MB | not_measured | small model is verbose and nearly always wrong |
| Local baseline: always-large, MATH (n=20) | 65.0% | 10.4 / 22.9 s | 0.08 q/s | 15,613 MB* | not_measured | |
| Local method: cascade@0.75, MATH | 60.0% (80% escalated) | 20.1 / 33.3 s | 0.05 q/s | 15,613 MB | not_measured | |

\* Both models stay loaded for all live configurations, so peak memory is that of an L+H deployment (weights: 0.95 GB small + 14.6 GB large). A single-model deployment needs only its own weights.

Key takeaways:

- **Escalation pays in compute, not in batch-1 latency.** The cascade saves FLOPs at matched accuracy, but on a single RTX 4090 the 7B model is memory-bandwidth-bound and answers short questions almost as fast as the 0.5B model. Running both models sequentially on escalated queries therefore *increases* end-to-end latency (MMLU median 54 → 96 ms at 79% escalation). The paper's cost model (API price per token) has no such effect.
- **Depth > 2 does not help**, under both cost measures and all scores (see above).
- **Prop. 3.2 holds on real data:** the directly integrated gain over random escalation equals p·[Γ(U_L) − Γ(U_H)] to numerical precision for every score with continuous values ([`theory.csv`](../../experiments/cascade_escalation/results_rtx4090/sweep/theory.csv)).
- **TriviaQA small model.** Qwen2.5-0.5B answers in full sentences, so the paper's strict normalised exact-match grader scores it at 0.1%. Under GPU-time cost the cheapest model is Qwen2.5-1.5B instead (31.5%), giving S2 − S1 = +2.3 to +3.8 pp.

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

- [x] Exact paper link and version verified (arXiv v2; OpenReview forum MfWD7bOKiF)
- [x] Base model and dataset revisions pinned (model SHAs above; prompts SHA-256 checked)
- [x] Baseline runs on the same setup (always-small, always-large, random router; same splits and budgets)
- [x] Tests and seed handling added (split seeds 42 to 91; 31 CPU tests)
- [x] Resource and quality metrics reported (energy not measured)
- [x] Licence/data terms checked (upstream Apache-2.0; dataset records keep their original licences; Qwen2.5 Apache-2.0 except 3B under the Qwen Research licence; Phi-3.5-mini MIT)
