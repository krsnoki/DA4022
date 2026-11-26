# Paper implementation card: Resource-Adaptive Foundation Model Reasoning via Semantic Coverage

- **Paper / OpenReview:** https://openreview.net/forum?id=suD1tqogTd (preliminary work under review at ICML; version read: the anonymized PDF with Appendices A to H)
- **Primary module:** `Adaptive_Stopping/` (budget policy)
- **Secondary modules:** `Backbones/Generative_Pretrained/` (generation wrappers), `Dataloaders/evaluation/` (MMLU-Pro), `Dataloaders/synthetic/` (smoke data), `Performance_Metrics/` (result rows)
- **Owner / reviewer:** <GitHub handles>
- **Status:** tested (controller and baselines tested on CPU; no MMLU-Pro run yet, see Results)
- **Upstream code / licence:** no public code found. The paper describes an anonymized supplement (`portfolio_search` package) that is not released. This implementation is written from the paper text.
- **Backbone IDs:** `text.qwen3_5_9b.post_trained` (paper setting), `text.qwen2_5_0_5b_instruct.post_trained` (CPU pipeline check), `mock.mcq.latent_dist` (tests). See `Backbones/Generative_Pretrained/registry.yaml`.
- **Dataset IDs and split:** `text.mmlu_pro.test` (256 questions, seed 47), `synthetic.mcq` (smoke). See `Dataloaders/evaluation/DATASET_CARD_mmlu_pro.md`.
- **Hardware assumptions:** paper: one CUDA GPU, vLLM, bfloat16, max model length 16384 (GPU model not named in the paper). Here: CPU for tests and the mock run; any CUDA GPU supported by vLLM for the paper setting.

## Research summary

### Problem

Test-time reasoning methods give every input the same number of samples or tree expansions. On exact-answer tasks many rollouts collapse to the same answer class, so trace count overstates progress and decode tokens are spent on duplicates.

### Method

SCPS treats test-time reasoning as building a portfolio of completed artifacts under a budget.

- **Semantic coverage** values a portfolio by the importance-weighted task support it adds beyond a baseline. For exact-answer tasks the targets are answer classes, weighted by the posterior that each class is correct. The paper proves this objective is normalized, monotone and submodular (Theorem 1), and that the expected gain of a batch of bounded rollout actions keeps those properties under batch separability (Theorem 2).
- **Overcount identity** (Proposition P3): scoring actions additively overcounts the true gain by exactly the expected number of duplicate residual hits. The second-order Bonferroni bound (Equation 5) subtracts pairwise overlaps.
- **Controller** (Algorithm 1): each round, build candidate actions from the frontier, estimate residual support `v`, pairwise overlap `omega` and cost `c`, select a feasible batch maximizing `L = sum v - beta * sum omega - lambda * sum c` (Equation 7), stop if the best batch has `L <= 0`, otherwise execute and replan. SCPS returns the portfolio; a separate selector picks the final answer.

### Expected behaviour

Paper-reported (Table 10 and Table 13; 256 held-out MMLU-Pro test questions, Qwen3.5-9B, vLLM, temperature 0.6, max 16 rollouts, 192 new tokens per completion, decode limit 3072, total limit 30000):

| Method | Pass@16 (portfolio) | Mean total tokens | Unique answers | Dup. token frac. |
|---|---:|---:|---:|---:|
| SCPS (selected configuration) | 91.0% (233/256) | 1120.25 | 3.469 | 55.4% |
| Self-consistency r16 | 78.1% (200/256) | 316.13 | 2.555 | 84.0% |
| Tree-prefix | 75.0% (192/256) | 3546.10 | 1.766 | 81.5% |
| Semantic pruning | 74.2% (190/256) | 3548.08 | 1.770 | 81.3% |

Paired Pass@16 deltas: +16.0 pp vs tree-prefix (95% CI [10.6, 21.5]), +16.8 pp vs semantic pruning ([11.3, 22.7]). Selector diagnostics on SCPS portfolios: common-selector accuracy 41.8%, majority accuracy 65.2%. Paper-reported mean wall clock per question for SCPS: 56.05 s. The paper reports no memory or energy figures.

## Implementation

| File | Role |
|---|---|
| `coverage.py` | Exact finite-target versions of Equations 1 to 6 (used by property tests, not on the hot path) |
| `surrogate.py` | Training-free plug-in surrogate for `v`, `omega` and cost |
| `controller.py` | `SCPSConfig`, `SCPSController` (Algorithm 1), `SCPSStoppingPolicy` (shared contract) |
| `../../../common/` | Budget ledger, MCQ prompt and canonicalizer, rollout execution, portfolio metrics, `StoppingPolicy` |
| `../../../baselines/` | Self-consistency (full-compute baseline), tree-prefix, semantic pruning (threshold baseline) |

- **Input contract:** `item = {"question_id", "question", "options": list[str] (2 to 26), "answer": letter}`; a backbone implementing `GenerativeBackbone` (and optionally `OptionScoringBackbone` for terminal-option log-probabilities); a `Budget(max_rollouts, max_new_tokens, decode_token_limit, total_token_limit)`.
- **Output contract:** `PortfolioResult` with completions (`answer` label or `None`, attributed decode tokens, source key, text, round), the ledger (rollouts, decode, auxiliary, total and prompt tokens), rounds, stop reason, per-round trace (selected actions, `v`, `omega`, cost, score, covered classes, surrogate state) and metadata (telemetry mode, config, budget violations).
- **Stop signal:** evaluated once per planning round before any generation. `SCPSStoppingPolicy.should_stop` reads the round index, the number of covered answer classes, budget feasibility and the best batch score. All of these come from the controller's memory, so the stop check needs no extra forward pass. Option scoring for the root prompt (and for new frontier prefixes, if enabled) is charged as auxiliary tokens.
- **Extra trained weights:** none. The paper's learned bundle (value, answer distribution, length and collision MLPs over pooled Qwen features, trained on a disjoint 512-question split) is replaced by the plug-in surrogate, so no calibration data is used (`calibration_dataset: null`).
- **Determinism:** deterministic for a fixed `run_seed` with deterministic backbones (the mock and HF on CPU with seeded sampling). vLLM serving can be nondeterministic across hardware and batch composition.
- **Token accounting:** total tokens = decode + auxiliary scoring tokens, matching the paper's split (Appendix H.6: 1120.25 total, 836.12 decode). Prompt prefill is reported separately. Scoring is charged as suffix tokens + 1, assuming prefix caching.
- **Configuration:** `Adaptive_Stopping/configs/scps_mmlupro_qwen35_9b.yaml` (paper setting), `scps_mmlupro_cpu_small.yaml` (CPU pipeline check), `scps_synthetic_smoke.yaml` (mock).
- **Reproduction command:**

  ```bash
  python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
      --config Adaptive_Stopping/configs/scps_mmlupro_qwen35_9b.yaml \
      --out-dir results/scps_mmlupro_qwen35_9b
  ```

- **Smoke command (CPU, about 10 s):**

  ```bash
  python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
      --config Adaptive_Stopping/configs/scps_synthetic_smoke.yaml \
      --out-dir results/scps_synthetic_smoke
  ```

- **Tests:** `python -m pytest -q Adaptive_Stopping/tests` (128 passed and 1 skipped on CPU with torch 2.15 and transformers 5.24; the vLLM test runs only when `SCPS_VLLM_MODEL` is set).

### Mapping from paper settings (Table 8) to config

| Paper setting | Config field | Value | Note |
|---|---|---|---|
| 10 planning rounds | `max_rounds` | 10 | |
| batch size 1 | `actions_per_round` | 1 | |
| root proposal batch 16 | `root_proposal_batch` | 16 | our reading: candidate sources scored per round |
| distinct-source-prefix target 12 | `distinct_source_target` | 12 | cap on distinct executed sources |
| terminal answer-class quota 8 | `answer_class_quota` | 8 | stop once 8 classes are covered |
| discovery weight 1.0 | `discovery_weight` | 1.0 | |
| overlap penalty 0.5 | `overlap_penalty` | 0.5 | |
| seen-answer penalty 1.0 | `seen_answer_penalty` | 1.0 | formula not given in the paper; ours is in `controller.score` |
| repeat penalty 0.25 | `repeat_penalty` | 0.25 | formula not given in the paper; ours is in `controller.score` |
| certification weight 0.0 | not implemented | | zero in the paper, so no effect |
| (not given) | `sample_sizes`, `cost_weight`, `posterior_mix`, `prior_strength` | (1, 2, 4), 1.0, 0.05, 4.0 | our choices |

## Results

| Source | Quality (portfolio Pass@16) | Latency (median / p95 per question) | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported, SCPS | 91.0% | 56.05 s mean (no median/p95 reported) | 145.84 decode tok/s | not reported | not reported | Qwen3.5-9B, vLLM, bf16, one unnamed CUDA GPU |
| Paper-reported, tree-prefix | 75.0% | not reported | not reported | not reported | not reported | same setup |
| Local baseline (self-consistency), MMLU-Pro | not measured | not measured | not measured | not measured | not measured | blocked: no model or dataset access where this was implemented |
| Local method (SCPS), MMLU-Pro | not measured | not measured | not measured | not measured | not measured | blocked, as above |
| Local baseline (self-consistency), synthetic mock | 81.25% (208/256) | 0.269 / 0.309 ms | mock, not meaningful | 516.5 MB process RSS | not measured | controller-logic smoke test only |
| Local method (SCPS), synthetic mock | 93.36% (239/256) | 1.99 / 3.008 ms | mock, not meaningful | 516.5 MB process RSS | not measured | controller-logic smoke test only |

The synthetic rows use a mock backbone whose option scores equal its own sampling distribution and whose targeted prompts hit their target 30% to 80% of the time. They show that the controller runs, respects every hard limit (0 violations in all methods) and spends fewer tokens (276.9 vs 400.5 mean total) on fewer duplicates (47.5% vs 74.0% duplicate-token fraction). They say nothing about MMLU-Pro quality. Machine-readable rows are in `Performance_Metrics/README_metrics.md`.

## Differences and limitations

- **Surrogate.** The learned bundle is replaced by a training-free estimator: option log-probabilities sharpened by the sampling temperature, Dirichlet updates from observed answers, a Beta-estimated steering rate for targeted actions and a Beta-estimated null rate. Collisions assume conditional independence of actions. Results with this surrogate are `approximate` at best.
- **Answer-diverse action generation.** The paper's targeted action generator is not specified beyond "answer-diverse root proposals". Ours appends "Before answering, carefully check whether option X is correct." to the root prompt. The paper's ablation shows this component matters a lot (91.0% vs 82.4% without it), so prompt choice will move results.
- **Pass@N on MCQ can be inflated by enumeration.** With 10 options and 16 rollouts, a controller that steers each rollout to a new option would cover the gold answer trivially. The answer-class quota (8) limits this, as in the paper. Always read Pass@16 together with unique answers, majority accuracy and the selector diagnostics.
- **Question IDs.** The paper's 256 held-out IDs and its 512-question controller-training split are not public. Our seed-47 sample is a different slice of the same test split, so paired comparisons with paper numbers are not possible.
- **Baselines.** Tree-prefix ranks frontier nodes by maximum option probability and re-expands the root when the frontier is empty; semantic pruning skips nodes whose predicted answer is already covered or whose cosine overlap with a chosen node exceeds 0.75. The paper does not give more detail, so these are our readings of Table 7. With completion-level horizons and answer-only prompting both behave close to repeated root sampling.
- **vLLM wrapper** is untested here (no GPU). Run `test_backbones_optional.py` with `SCPS_VLLM_MODEL` set before trusting it.
- **Scope of the paper's evidence** (from its own limitations section): one model, one MMLU-Pro slice, one serving setup, tokens as the only resource readout, a single operating point and no equal-token comparison.

## Reproduction checklist

- [x] Exact paper link and version verified (OpenReview forum ID from `PAPER_INDEX.md`, anonymized ICML submission PDF)
- [ ] Base model and dataset revisions pinned (configs carry `revision: null`; pin full commit hashes before reporting)
- [x] Baseline runs on the same setup (same items, prompt, canonicalizer, seeds and hard budgets for all four methods)
- [x] Tests and seed handling added
- [ ] Resource and quality metrics reported (synthetic only; MMLU-Pro run pending)
- [x] Licence/data terms checked (MMLU-Pro: MIT; Qwen3.5-9B: Apache-2.0; Qwen2.5-0.5B-Instruct: Apache-2.0)
