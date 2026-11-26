# Paper implementation card: Resource-Adaptive Foundation Model Reasoning via Semantic Coverage

- **Paper / OpenReview:** https://openreview.net/forum?id=suD1tqogTd (preliminary work under review at ICML; version read: the anonymized PDF with Appendices A to H)
- **Primary module:** `Adaptive_Stopping/` (budget policy)
- **Secondary modules:** `Backbones/Generative_Pretrained/` (generation wrappers), `Dataloaders/evaluation/` (MMLU-Pro), `Dataloaders/synthetic/` (smoke data), `Performance_Metrics/` (result rows)
- **Owner / reviewer:** <GitHub handles>
- **Status:** partial-reproduction (full 256-question MMLU-Pro run with Qwen3.5-4B in 4-bit NF4 on a 4 GB laptop GPU, all four methods plus a top-K control; the paper's Qwen3.5-9B bf16 vLLM setting has not been run)
- **Upstream code / licence:** no public code found. The paper describes an anonymized supplement (`portfolio_search` package) that is not released. This implementation is written from the paper text.
- **Backbone IDs:** `text.qwen3_5_4b.post_trained` (reported run, `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4), `text.qwen3_5_9b.post_trained` (paper setting, not run), `text.qwen3_5_2b.post_trained` (fallback), `text.qwen2_5_0_5b_instruct.post_trained` (CPU pipeline check), `mock.mcq.latent_dist` (tests). See `Backbones/Generative_Pretrained/registry.yaml`.
- **Dataset IDs and split:** `text.mmlu_pro.test` (`TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`, 256 of 12,032 test questions, seed 47), `synthetic.mcq` (smoke). See `Dataloaders/evaluation/DATASET_CARD_mmlu_pro.md`.
- **Hardware assumptions:** paper: one CUDA GPU, vLLM, bfloat16, max model length 16384 (GPU model not named in the paper). Reported run: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, driver 610.91), Windows 10.0.26200, Python 3.11.18, torch 2.11.0+cu128, transformers 5.24.1, bitsandbytes 0.50.2. Tests: CPU only.

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
| `../../../baselines/` | Self-consistency (full-compute baseline), tree-prefix, semantic pruning (threshold baseline), option top-K control |
| `../../../experiments/scps_mmlupro/` | `run.py` (experiment), `report.py` (tables in this card), `run_windows.ps1`, `reported_runs/` (outputs of the reported run) |

- **Input contract:** `item = {"question_id", "question", "options": list[str] (2 to 26), "answer": letter}`; a backbone implementing `GenerativeBackbone` (and optionally `OptionScoringBackbone` for terminal-option log-probabilities); a `Budget(max_rollouts, max_new_tokens, decode_token_limit, total_token_limit)`.
- **Output contract:** `PortfolioResult` with completions (`answer` label or `None`, attributed decode tokens, source key, text, round), the ledger (rollouts, decode, auxiliary, total and prompt tokens), rounds, stop reason, per-round trace (selected actions, `v`, `omega`, cost, score, covered classes, surrogate state) and metadata (telemetry mode, config, budget violations).
- **Stop signal:** evaluated once per planning round before any generation. `SCPSStoppingPolicy.should_stop` reads the round index, the number of covered answer classes, budget feasibility and the best batch score. All of these come from the controller's memory, so the stop check needs no extra forward pass. Option scoring for the root prompt (and for new frontier prefixes, if enabled) is charged as auxiliary tokens.
- **Extra trained weights:** none. The paper's learned bundle (value, answer distribution, length and collision MLPs over pooled Qwen features, trained on a disjoint 512-question split) is replaced by the plug-in surrogate, so no calibration data is used (`calibration_dataset: null`).
- **Determinism:** deterministic for a fixed `run_seed` with deterministic backbones. Checked on the reported setup: a separate SCPS rerun of the first 32 questions in a new process reproduced all 32 portfolios exactly. vLLM serving can be nondeterministic across hardware and batch composition.
- **Token accounting:** total tokens = decode + auxiliary scoring tokens, matching the paper's split (Appendix H.6: 1120.25 total, 836.12 decode). Prompt prefill is reported separately. Scoring is charged as suffix tokens + 1, assuming prefix caching.
- **Configuration:** `Adaptive_Stopping/configs/scps_mmlupro_qwen35_4b_bnb4.yaml` (reported run, revisions pinned), `scps_mmlupro_qwen35_9b.yaml` (paper setting, vLLM), `scps_mmlupro_qwen35_9b_bnb4.yaml` and `scps_mmlupro_qwen35_2b_bnb4.yaml` (other GPU sizes), `scps_mmlupro_cpu_small.yaml` (CPU pipeline check), `scps_synthetic_smoke.yaml` (mock).
- **Reproduction command (reported run, about 6 hours on an RTX 3050 Laptop 4 GB):**

  ```bash
  python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
      --config Adaptive_Stopping/configs/scps_mmlupro_qwen35_4b_bnb4.yaml \
      --out-dir results/scps_mmlupro_qwen35_4b_bnb4 --resume
  python -m Adaptive_Stopping.experiments.scps_mmlupro.report results/scps_mmlupro_qwen35_4b_bnb4
  ```

  On Windows, `run_windows.ps1` does environment setup, tests, an 8-question pilot and the full run. For the paper setting use `scps_mmlupro_qwen35_9b.yaml` on a vLLM-capable GPU.

- **Smoke command (CPU, about 10 s):**

  ```bash
  python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
      --config Adaptive_Stopping/configs/scps_synthetic_smoke.yaml \
      --out-dir results/scps_synthetic_smoke
  ```

- **Tests:** `python -m pytest -q Adaptive_Stopping/tests` (132 passed and 1 skipped on Linux CPU with Python 3.13, torch 2.15 and transformers 5.24; the vLLM test runs only when `SCPS_VLLM_MODEL` is set). Includes a check that the committed `reported_runs` summary is reproducible from its per-question files.

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

### MMLU-Pro, Qwen3.5-4B NF4, 256 questions (local run)

Setup: `scps_mmlupro_qwen35_4b_bnb4.yaml`; answer-only prompt; temperature 0.6; hard budget of 16 rollouts, 192 new tokens per completion, 3,072 decode tokens and 30,000 total tokens per question; same questions, prompt, canonicalizer and seeds for every method. Self-consistency draws its 16 samples 4 per call (16 in one batch ran out of memory). Status of every row: `approximate`. Outputs: `Adaptive_Stopping/experiments/scps_mmlupro/reported_runs/qwen35_4b_bnb4_rtx3050/`.

| Method | Pass@16 (gold in portfolio) | Majority-vote accuracy | Distinct answers | Rollouts | Total tokens (decode + aux) | Dup. token frac. | Null completions | Latency median / p95 per question | Peak CUDA memory | Budget violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCPS | 87.1% (223/256) | 37.1% | 4.96 | 14.82 | 38.4 | 64.9% | 7 | 15.6 s / 65.6 s | 4,339 MB | 0 |
| Self-consistency (16 samples) | 63.3% (162/256) | 37.5% | 2.39 | 16.00 | 32.0 | 85.1% | 0 | 7.2 s / 37.0 s | 4,770 MB | 0 |
| Tree-prefix | 64.5% (165/256) | 38.7% | 2.36 | 16.00 | 38.8 | 85.0% | 4 | 11.1 s / 16.5 s | 4,080 MB | 0 |
| Semantic pruning (0.75) | 61.3% (157/256) | 38.7% | 2.31 | 16.00 | 37.0 | 85.5% | 0 | 10.1 s / 21.6 s | 4,080 MB | 0 |
| Option top-8 (no sampling) | 98.0% (251/256) | 38.7% | 7.79 | 0.00 | 5.0 | n/a | 0 | 0.5 s / 0.9 s | 3,975 MB | 0 |

"Option top-8" is a control, not a sampler: it lists the model's 8 most probable options after the root prompt (one forward pass). Its quality column is gold-in-top-8, and its majority-vote column is the model's single most probable option (greedy accuracy).

Paired bootstrap, SCPS minus baseline (10,000 resamples of questions, 95% percentile interval):

| Baseline | Pass@16 delta | Majority-vote accuracy delta |
|---|---|---|
| Self-consistency (16 samples) | +23.8 pp [+18.4, +29.7] (65 wins / 4 losses) | -0.4 pp [-5.1, +4.3] (17 wins / 18 losses) |
| Tree-prefix | +22.7 pp [+17.2, +28.1] (61 wins / 3 losses) | -1.6 pp [-6.2, +3.1] (16 wins / 20 losses) |
| Semantic pruning (0.75) | +25.8 pp [+19.9, +31.6] (70 wins / 4 losses) | -1.6 pp [-5.9, +3.1] (15 wins / 19 losses) |

**Matched-cardinality control.** For each question, take K = the number of distinct answers SCPS found on that question and check whether the gold option is in the model's top K options. This reaches 87.5% against SCPS's 87.1%: SCPS minus matched top-K is -0.4 pp [-2.3, +1.6] (3 wins, 4 losses). Gold-in-top-K for fixed K:

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gold in top-K | 38.7% | 58.6% | 73.0% | 81.6% | 87.1% | 91.0% | 96.1% | 97.7% | 98.4% | 100.0% |

Per category (Pass@16 counts; the top-8 control counts gold-in-top-8):

| Category | n | SCPS | Self-consistency (16 samples) | Tree-prefix | Semantic pruning (0.75) | Option top-8 (no sampling) |
|---|---:|---:|---:|---:|---:|---:|
| biology | 11 | 11 | 10 | 10 | 9 | 11 |
| business | 20 | 20 | 12 | 11 | 10 | 20 |
| chemistry | 26 | 23 | 12 | 16 | 14 | 26 |
| computer science | 8 | 7 | 6 | 5 | 5 | 8 |
| economics | 16 | 14 | 12 | 11 | 12 | 16 |
| engineering | 19 | 16 | 15 | 15 | 13 | 17 |
| health | 14 | 13 | 10 | 9 | 9 | 13 |
| history | 3 | 2 | 2 | 2 | 2 | 3 |
| law | 24 | 18 | 14 | 13 | 13 | 24 |
| math | 34 | 29 | 20 | 24 | 21 | 33 |
| other | 20 | 17 | 14 | 15 | 12 | 20 |
| philosophy | 7 | 6 | 6 | 5 | 5 | 7 |
| physics | 43 | 36 | 22 | 21 | 24 | 42 |
| psychology | 11 | 11 | 7 | 8 | 8 | 11 |

Latency comparison on the same 32 questions. SCPS's 256-question run happened in an earlier session than the baselines and was slower on identical work, so the full-run SCPS latency above is not directly comparable:

| Method | Session | Median | p95 |
|---|---|---:|---:|
| SCPS | main run (session 1) | 12.8 | 26.6 |
| SCPS | rerun (session 3) | 8.6 | 24.4 |
| Self-consistency (16 samples) | session 2 | 7.8 | 21.4 |
| Tree-prefix | session 2 | 11.5 | 14.8 |
| Semantic pruning (0.75) | session 2 | 10.9 | 20.5 |
| Option top-8 (no sampling) | session 2 | 0.6 | 1.0 |

### What the run shows

1. **Pass@16 direction reproduces.** SCPS covers the gold answer far more often than all three baselines (+22.7 to +25.8 pp, intervals well above zero), as in the paper (+16.0 pp over tree-prefix there).
2. **The gain is coverage of more options, not better search.** SCPS finds 4.96 distinct answers per question against about 2.3 for the baselines. Ranking options by the model's own root probability and taking the same number per question gives the same Pass@16 (-0.4 pp [-2.3, +1.6]) from one forward pass. On 205 of 256 questions the set of answers SCPS found is exactly the model's top-K options. On this multiple-choice setup, steering rollouts toward unseen options adds nothing beyond what the model's option probabilities already contain.
3. **No accuracy change.** Majority-vote accuracy is 37.1% to 38.7% for every method, with paired differences of -1.6 to -0.4 pp whose intervals include zero. The targeted prompts that raise coverage do not hurt or help the final answer under majority voting.
4. **No compute saving here.** With answer-only outputs of about 2 tokens, the cost term is negligible and SCPS stops on the hard budget in 188 of 256 questions (14.82 rollouts on average). Its sequential planning rounds make it slower per question than batched self-consistency (8.6 s vs 7.8 s median on the same 32 questions in comparable sessions; 15.6 s vs 7.2 s over the full runs).
5. **Budgets hold.** 0 budget violations in 1,280 question-method runs; 11 null completions in total.

### Paper-reported vs local

| Source | SCPS Pass@16 | Tree-prefix Pass@16 | SCPS tokens / question | SCPS latency | Peak memory | Energy |
|---|---:|---:|---:|---:|---:|---:|
| Paper (Qwen3.5-9B bf16, vLLM, unnamed GPU) | 91.0% | 75.0% | 1120.25 | 56.05 s mean | not reported | not reported |
| Local (Qwen3.5-4B NF4, Transformers, RTX 3050 Laptop 4 GB) | 87.1% | 64.5% | 38.4 | 15.6 s median (8.6 s on the 32-question rerun) | 4,339 MB CUDA peak allocated | not measured |

The question samples differ (the paper's IDs are not public), so these rows are not a paired comparison.

### Synthetic smoke run (mock backbone, controller logic only)

| Source | Quality (portfolio Pass@16) | Latency (median / p95 per question) | Peak memory | Notes |
|---|---:|---:|---:|---|
| Self-consistency, synthetic mock | 81.25% (208/256) | 0.269 / 0.309 ms | 516.5 MB process RSS | controller-logic test only |
| SCPS, synthetic mock | 93.36% (239/256) | 1.99 / 3.008 ms | 516.5 MB process RSS | controller-logic test only |

Machine-readable rows for both runs are in `Performance_Metrics/README_metrics.md`.

## Differences and limitations

- **Surrogate.** The learned bundle is replaced by a training-free estimator: option log-probabilities sharpened by the sampling temperature, Dirichlet updates from observed answers, a Beta-estimated steering rate for targeted actions and a Beta-estimated null rate. Collisions assume conditional independence of actions. Results with this surrogate are `approximate` at best.
- **Answer-diverse action generation.** The paper's targeted action generator is not specified beyond "answer-diverse root proposals". Ours appends "Before answering, carefully check whether option X is correct." to the root prompt. The paper's ablation shows this component matters a lot (91.0% vs 82.4% without it), so prompt choice will move results.
- **Pass@N on MCQ rewards enumeration.** The reported run confirms this concern: the matched top-K control explains the whole SCPS Pass@16 gain. Any future SCPS claim on multiple choice should report the matched top-K control, unique answers and a selector metric next to Pass@N.
- **Question IDs.** The paper's 256 held-out IDs and its 512-question controller-training split are not public. Our seed-47 sample is a different slice of the same test split, so paired comparisons with paper numbers are not possible.
- **Baselines.** Tree-prefix ranks frontier nodes by maximum option probability and re-expands the root when the frontier is empty; semantic pruning skips nodes whose predicted answer is already covered or whose cosine overlap with a chosen node exceeds 0.75. The paper does not give more detail, so these are our readings of Table 7. With completion-level horizons and answer-only prompting both behave close to repeated root sampling.
- **Model and serving.** The reported run uses Qwen3.5-4B with 4-bit NF4 weights through Transformers, not the paper's Qwen3.5-9B bf16 through vLLM. On Windows, Qwen3.5's linear-attention layers run on Transformers' reference PyTorch kernels because `flash-linear-attention` and `causal-conv1d` need Triton, which is not available there; this makes absolute latency much higher than an optimized server.
- **Memory.** Peak CUDA allocations (MB = 2^20 bytes) for SCPS (4,339) and self-consistency (4,770) exceed the card's 4,096, so the Windows driver spilled into shared system memory at peaks; the other methods peaked just below the limit (3,975 to 4,080). Spilling slows those steps and adds to latency variance.
- **Latency noise.** Per-question wall clock includes outliers from the laptop: one SCPS question took 2,902 s (the machine most likely slept) and sits in the reported maximum. Medians are the reliable figure; p95 values are affected.
- **Accuracy level.** Greedy accuracy with the answer-only prompt and thinking disabled is 38.7%, far below the 79.1 MMLU-Pro score reported for Qwen3.5-4B with reasoning ([summary of Qwen's release benchmarks](https://rits.shanghai.nyu.edu/ai/qwen-3-5-small-models-9b-parameters-that-beat-120b/)). All methods share this prompt, so comparisons between them are fair, but absolute accuracy is not comparable to leaderboard numbers.
- **vLLM wrapper** is untested here (no suitable GPU). Run `test_backbones_optional.py` with `SCPS_VLLM_MODEL` set before trusting it.
- **Scope of the paper's evidence** (from its own limitations section): one model, one MMLU-Pro slice, one serving setup, tokens as the only resource readout, a single operating point and no equal-token comparison.

## Reproduction checklist

- [x] Exact paper link and version verified (OpenReview forum ID from `PAPER_INDEX.md`, anonymized ICML submission PDF)
- [x] Base model and dataset revisions pinned for the reported run (`scps_mmlupro_qwen35_4b_bnb4.yaml`); the other configs still carry `revision: null`
- [x] Baseline runs on the same setup (same items, prompt, canonicalizer, seeds and hard budgets for all methods), plus a matched top-K control
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported (MMLU-Pro with Qwen3.5-4B NF4: quality, tokens, latency, peak memory; energy not measured)
- [ ] Paper setting (Qwen3.5-9B bf16, vLLM) run
- [x] Licence/data terms checked (MMLU-Pro: MIT; Qwen3.5-4B and Qwen3.5-9B: Apache-2.0; Qwen2.5-0.5B-Instruct: Apache-2.0)
