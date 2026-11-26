# Paper implementation card: Step-Tagging Early-Stopping (ST-ES)

**Full title:** *Step-Tagging Early-Stopping: Toward controlling the generation of Language Reasoning Models through black-box step monitoring*

- **Paper / OpenReview:**
  - https://openreview.net/forum?id=bNn5OusO22 (AdaptFM @ ICML 2026, poster session 1). The title was checked against the official accepted-paper list.
  - arXiv [2512.14332](https://arxiv.org/abs/2512.14332), v2 of 16 Jul 2026.
  - The team downloaded the latest version of the paper: the 22-page proceedings-format PDF (`75_Step_Tagging_Early_Stopping.pdf`, with author names, built 2026-06-08). We compared its text with arXiv v2 line by line. It is identical except for Algorithm 1, which adds a *minimal step size* k ("Generate step s_i … where |s_i| > k"), and no value for k is given. Eq. 1, Eq. 3, the 100-token answer budget and all reported numbers are the same in both.
  - Authors: Y. Belkhiter, S. Tirupathi, G. Zizzo, J. D. Kelleher.
- **Primary module:** `Adaptive_Stopping/`
- **Secondary modules:** `Backbones/`, `Dataloaders/`, `Performance_Metrics/`
- **Owner / reviewer:**
  - Implementation: @devanshjangid2005, the author of every commit.
  - Team: Sanidhya Srivastava, Arnav Tiku, Aman Gupta, Karan Malik.
  - No separate reviewer is assigned; the reproduction checks are in [D](docs/D_reproduction_and_review.md).
  - Tracking issue: krsnoki/DA4022#24, which lists the five PRs of this submission.
- **Status:** `partial-reproduction`
- **Upstream code / licence:**
  - No code release for ST-ES was found (paper, arXiv page, GitHub search, 2026-11-25).
  - The first author's MIT-licensed [TRACES](https://github.com/yannisbelkhiter/TRACES) repository belongs to a later paper whose stopping rule (a constructive/evaluative ratio with a sliding window) is different. It was **not** implemented. It was consulted only to confirm details the ST-ES paper also states: the `"\n\n"` split, the 14 annotation label names, and the prompt-guided baseline prompts.
  - The 99 GPT-4o-mini-labelled validation steps come from the authors' MIT-licensed [ReasonType_Human_validation](https://github.com/yannisbelkhiter/ReasonType_Human_validation) (commit `efb7676`). They are downloaded at run time and not committed.
- **Backbone IDs:** `text.deepseek-r1-distill-qwen-1.5b` (revision `ad9f0ae`)
- **Dataset IDs and split:**
  - GSM8K: calibration on `text.gsm8k.train` (200 problems), evaluation on `text.gsm8k.test` (200).
  - MATH: calibration on `text.math.train` (150), evaluation on `text.math500.test` (150).
  - All subsets use seed 42.
- **Hardware assumptions:**
  - One CUDA GPU with at least 8 GB (tested: RTX 4060 Laptop GPU, 8 GB, Windows 11, driver 610.88).
  - The policy itself is CPU-only.
  - Tests run on CPU.

## Research summary

### Problem
Reasoning models over-generate. After they have effectively found the answer, they keep verifying, re-checking and reflecting, which costs tokens, latency and energy. Existing dynamic early-exit criteria (confidence or entropy) need access to model internals (white-box or grey-box). Static budgets ignore what the model is actually doing.

### Method
1. **Segment the generated reasoning into steps** at every `"\n\n"` (Eq. 1).
2. **Tag each step** with a reasoning type from *ReasonType*, a 14-class taxonomy (problem re-statement, formula substitution, verification, self-talk, exploration, ...). The paper derived the taxonomy by having GPT-4o-mini label steps. It labels steps with GPT-4o-mini and trains one binary `bert-base-uncased` classifier per step type on those labels.
3. **Stop rule (Eq. 3, Alg. 1):** for a chosen step type τ\* and threshold δ, continue while the running count of τ\* steps is ≤ δ. Stop at the first step that makes the count exceed δ.
4. **Answer forcing:** after stopping, prompt the model for its current best answer, with 100 extra tokens.
5. **Calibration:** sweep τ\* over the step types and δ ∈ [0, 20] on a training split. Keep the configurations that preserve 85%, 90% and 95% of standard accuracy. Apply them to the held-out test split.

The whole criterion is black-box: it reads only the generated text.

### Expected behaviour
- **Token savings:** the paper reports 20–50% fewer tokens than standard generation, with accuracy between +2% and −12% depending on the target.
- **Comparison with the token-count baseline:** on the training split, ST-ES curves for late-reasoning types (verification, self-talk) match or beat the token-count (step-count) baseline.
- **Latency and memory:** the paper reports token counts, not measured latency. Its runtime analysis (App. F) is an estimate that assumes runtime is linear in tokens.

Paper-reported test results (Table 6, mean of 5 seeds; "Avg@5" = mean accuracy):

| Model | Dataset | Config | Tokens | Saved | Avg@5 |
|---|---|---|---:|---:|---:|
| DS-R1-Distill-Llama-8B | GSM8K | standard | 958.3 | – | 0.829 |
| | | ST-ES 85% / 90% / 95% | 521.5 / 618.8 / 696.7 | 45.6 / 35.4 / 27.3% | 0.708 / 0.788 / 0.809 |
| DS-R1-Distill-Llama-8B | MATH500 | standard | 3655.0 | – | 0.878 |
| | | ST-ES 85% / 90% / 95% | 1801.4 / 2120.3 / 2614.5 | 50.7 / 42.0 / 28.5% | 0.761 / 0.793 / 0.845 |
| DS-R1-Distill-Qwen-14B | GSM8K | standard | 662.9 | – | 0.910 |
| | | ST-ES 85% / 90% / 95% | 377.2 / 415.9 / 464.6 | 43.1 / 37.3 / 29.9% | 0.527 / 0.621 / 0.865 |
| DS-R1-Distill-Qwen-14B | MATH500 | standard | 3388.8 | – | 0.923 |
| | | ST-ES 85% / 90% / 95% | 2158.4 / 2393.6 / 2851.9 | 36.3 / 29.4 / 15.8% | 0.827 / 0.853 / 0.909 |

## Implementation

- **Input contract:**
  - The policy (`StepTagEarlyStopping.should_stop(state, metrics, budget)`) receives a `StepState` for every completed step: step index, step text (`str`, or `None` for missing telemetry), running generated-token count, and whether the model is still inside `<think>`.
  - Optional `metrics={"tag": ...}` passes a precomputed tag.
  - Optional `budget={"max_steps", "max_tokens"}` adds hard caps; a zero budget stops at the first step.
  - The live loop runs at batch size 1 inside `model.generate` as a stopping criterion. The decision is made at the token that closes `"\n\n"`.
- **Output contract:** `generate_with_policy(...)` returns:
  - the generated token IDs, whether the policy stopped, the stop step and the cut (number of kept tokens);
  - the forced-answer token IDs;
  - timings (end-to-end, TTFT, reasoning, answer, policy time);
  - the policy report: tag counts, count of τ\*, stop reason.
  - `answer_text()` gives the scored text.
- **Configuration:**
  - [`configs/st_es/gsm8k.yaml`](../../configs/st_es/gsm8k.yaml) and [`configs/st_es/math500.yaml`](../../configs/st_es/math500.yaml).
  - Key settings: seed 42; sampling T=0.6, top-p 0.95, top-k 50; bf16; token cap 4096 (GSM8K) / 8192 (MATH).
  - Tagger `keyword_reasontype.v1`; exit prompt `exit_boxed.v1` = `</think>\n\n**Final Answer**\n\boxed{` followed by ≤100 greedy tokens.
  - Grid: 13 step types × δ ∈ [0, 20], plus step counts {1…10, 12, …, 500}. Targets 85/90/95%.
  - Every run writes a manifest with model revision, counted parameters, GPU, library versions and the trace sha256.
- **Reproduction command** (same for `math500.yaml`):
  ```bash
  python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
  python -m Adaptive_Stopping.experiments.st_es.validate_tagger
  python -m Adaptive_Stopping.experiments.st_es.make_report --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
  ```
  Additional analyses, each documented in [`docs/`](docs/):

  | Analysis | Doc page | Command |
  |---|---|---|
  | Prompt-guided baselines | [A](docs/A_prompt_guided_baselines.md) | `run_prompt_guided --config …` |
  | Minimal step size k | [B](docs/B_min_step_size.md) | `ablation_min_step --config …` |
  | Tagger analysis | [C](docs/C_tagger_analysis.md) | `tagger_analysis --configs …` |
  | Reproduction kit and review | [D](docs/D_reproduction_and_review.md) | `run_all --config Adaptive_Stopping/configs/st_es/gsm8k_smoke.yaml --run-dir <tmp>` (or `math500_smoke.yaml`) |

  The implementation is submitted as five PRs, in dependency order:
  1. shared backbone, dataset and scorer entries;
  2. the step-level stopping framework (`common/`, `NoStopping`, `StepCountStopping`);
  3. the ST-ES method and experiment pipeline;
  4. the extensions A–C;
  5. the main results, metrics rows and the reproduction kit D.
- **Tests:**
  - Command: `python -m pytest Adaptive_Stopping/tests Dataloaders/tests Performance_Metrics/tests -q`
  - Count: 188 tests, all on CPU, nothing downloaded. PRs 1–5 contribute 21, 30, 85, 44 and 8.
  - Two of them (`test_steps.py`, `test_prompt_guided.py`) use the real backbone tokenizer. They are skipped unless it is already in the local Hugging Face cache (`HF_HOME`). Any earlier run of the pipeline puts it there.
  - They cover segmentation versus `str.split` under arbitrary chunking, tagger cases, δ/budget edge cases, fast-sweep versus policy replay equivalence, step merging, prompt templates, the loaders, the scorer, and `run_all` wiring and config validation.
  - A tiny random Qwen2 test checks four equivalences:
    - batched continuation generation (static KV cache, chunked prefill) equals plain greedy decoding;
    - shared-prefix forced answers equal fresh prefill;
    - the live forced answer with KV-cache reuse equals fresh prefill;
    - the live stop equals the replay stop.

## Results

All local numbers use DeepSeek-R1-Distill-Qwen-1.5B with one seed (42). Replay numbers come from offline replay over cached generations, as in the paper. Latency and memory come only from the live run. Paper-reported numbers are in the table above and come from different, larger models.

- **Machine-readable rows:** [`Performance_Metrics/results/st_es.csv`](../../../Performance_Metrics/results/st_es.csv) (ST-ES, step-count and standard). The prompt-guided baseline rows are in `Performance_Metrics/results/st_es_prompt_guided.csv` ([A](docs/A_prompt_guided_baselines.md)).
- **Full outputs:** [`experiments/st_es/results/`](../../experiments/st_es/results/), with the calibration sweep, test table, paired comparisons, live summary, plots and provenance including trace sha256 values.

### GSM8K

**Calibration.** On 200 `text.gsm8k.train` problems, standard generation scores 0.875 with 1508 mean tokens. Calibration then selects:

| Target | ST-ES config | Step-count config |
|---|---|---|
| 85% | verification, δ=0 | k=14 |
| 90% | verification, δ=0 | k=16 |
| 95% | exploration, δ=0 | k=20 |

**Held-out test, replay** (200 `text.gsm8k.test` problems, token cap 4096; 28 of the standard traces hit the cap). Δ is the paired-bootstrap 95% interval against standard.

| Config | Accuracy | Δ vs standard [95% CI] | Mean generated tokens | Tokens saved | Stop rate |
|---|---:|---|---:|---:|---:|
| Standard | 0.830 | – | 1785.8 | – | 0 |
| **ST-ES 85%/90%: verification, δ=0** | 0.815 | −1.5 [−5.5, +2.0] | 779.3 | **56.4%** | 0.91 |
| **ST-ES 95%: exploration, δ=0** | 0.830 | 0.0 [−3.0, +3.5] | 921.8 | **48.4%** | 0.71 |
| Step-count 85%: k=14 | 0.750 | −8.0 [−13.0, −3.0] | 737.1 | 58.7% | 0.77 |
| Step-count 90%: k=16 | 0.795 | −3.5 [−7.5, +0.5] | 814.8 | 54.4% | 0.69 |
| Step-count 95%: k=20 | 0.840 | +1.0 [−3.0, +5.0] | 947.3 | 47.0% | 0.58 |

**ST-ES minus step-count at the same target** (paired bootstrap):

| Target | Difference [95% CI] | Tokens, ST-ES vs step-count |
|---|---|---|
| 85% | **+6.5 points [+2.0, +11.5]** | 779 vs 737 |
| 90% | +2.0 [−2.0, +6.5] | 779 vs 815 |
| 95% | −1.0 [−4.5, +2.5] | 922 vs 947 |

So, on held-out GSM8K, ST-ES halves the generated tokens with no statistically significant accuracy loss against standard generation. It beats the paper's step-count baseline at the most aggressive target, and matches it at the other two. This agrees in direction with the paper (20–50% savings; ST-ES at or above the token-count baseline), but the edge over the baseline is smaller here and only significant at 85%.

**Live, measured** (first 30 problems of the same test subset; batch 1; RTX 4060 Laptop GPU, bf16, on AC power; one warm-up request; prompt token IDs on the GPU to final answer, including tagging and the forced answer):

| Policy | Accuracy | Mean tokens | Latency median | Latency mean | Latency p95 | TTFT median | Tokens/s | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.900 | 1564 | 45.2 s | 56.7 s | 123.1 s | 44 ms | 27.6 | 3599 MB |
| ST-ES 90% (verification, δ=0) | 0.700 | 834 | **21.5 s** | **30.5 s** | 85.2 s | 43 ms | 27.4 | 3599 MB |

- **Latency.** Median latency falls by 52% and mean latency by 46%, roughly in proportion to the generated tokens; the decode speed is unchanged.
- **Tagger cost.** The tagger costs 0.2 ms per step (0.1 s over all 30 requests).
- **Peak memory.** Peak memory is the same, because the maximum comes from a request where no verification step occurred and generation ran to the 4096-token cap.
- **Accuracy (30 problems): diagnosed.** Live accuracy dropped 20 points (6 of 30 problems), while on the same 30 problems the replay traces show no drop. We re-ran those 6 problems with token logging (`live_benchmark --only-uids`). The live code path is correct:
  - each run is deterministic;
  - the ST-ES run follows the standard run token for token up to its stop;
  - every stop matched an offline replay of the same tokens.

  The losses are a property of the method on this model. The 1.5B model checks **intermediate** results ("let me check: 20% tip … is $3. Yeah, that checks out"; "let me confirm: … 18 by the administrators, that's 36"). With δ=0 the first such check stops generation, and the forced answer is that intermediate quantity: 3 instead of 18, 18 instead of 36, 24 instead of 48.
  - **In the 200-problem replay** the same mechanism loses 9 answers, but early stopping also *recovers* 6 problems where standard generation ran into the 4096-token cap without answering. The net change is −1.5 points.
  - **In the first 30 live problems** only one standard trace hit the cap, so there was almost nothing to recover.
  - **On MATH-500** the replay shows 16 losses and 4 recoveries.

  The replay over 200 problems remains the accuracy estimate. The live run shows the failure mode: δ=0 with a verification trigger is fragile on multi-step word problems.
- **Measurement caveat.** Two of the 30 live requests (problems 2–3) overlapped with CPU-heavy replay jobs on the same machine, so their latency may be slightly inflated.

### MATH-500

**Calibration.** On 150 `text.math.train` problems, standard generation scores 0.847 with 3461 mean tokens; 24 of the 150 traces hit the 8192 cap. Calibration then selects:

| Target | ST-ES config | Step-count config |
|---|---|---|
| 85% | self-talk, δ=3 | k=40 |
| 90% | self-talk, δ=4 | k=60 |
| 95% | self-talk, δ=6 | k=70 |

On this split the self-talk curve lies above the step-count curve (`results/text.math500.test/calibration_curves.png`), as in the paper's Fig. 3.

**Held-out test, replay** (150 `text.math500.test` problems; 26 of the standard traces hit the cap):

| Config | Accuracy | Δ vs standard [95% CI] | Mean generated tokens | Tokens saved | Stop rate |
|---|---:|---|---:|---:|---:|
| Standard | 0.807 | – | 3710 | – | 0 |
| ST-ES 85%: self-talk, δ=3 | 0.720 | −8.7 [−14.7, −3.3] | 1311 | **64.7%** | 0.76 |
| ST-ES 90%: self-talk, δ=4 | 0.727 | −8.0 [−14.0, −2.7] | 1491 | **59.8%** | 0.65 |
| ST-ES 95%: self-talk, δ=6 | 0.767 | −4.0 [−9.3, +0.7] | 1757 | **52.6%** | 0.55 |
| Step-count 85%: k=40 | 0.733 | −7.3 [−13.3, −1.3] | 1710 | 53.9% | 0.58 |
| Step-count 90%: k=60 | 0.780 | −2.7 [−7.3, +2.0] | 2118 | 42.9% | 0.49 |
| Step-count 95%: k=70 | 0.787 | −2.0 [−6.7, +2.0] | 2310 | 37.7% | 0.41 |

- **Cost against standard.** On MATH-500, early stopping costs accuracy: ST-ES loses 4–9 points while saving 53–65% of tokens.
  - The losses are of the same size as the paper's: DS-R1-Distill-Llama-8B on MATH500 lost 11.7 / 8.5 / 3.3 points with 51 / 42 / 28% saved.
  - Our token savings are larger.
- **Against the step-count baseline at the same target.** ST-ES scores −1.3 [−6.7, +4.7], −5.3 [−10.7, −0.7] and −2.0 [−6.7, +2.0] points, but uses 23–30% fewer tokens.
- **At matched token budgets:**
  - ST-ES 95% (1757 tokens) vs step-count k=40 (1710 tokens): **+3.3 points [−1.3, +8.7]**.
  - On GSM8K, ST-ES verification δ=0 (779 tokens) vs k=16 (815 tokens): +2.0 [−2.0, +6.5].
  - So, with our model and tagger, ST-ES is at least as good as the paper's baseline. Its advantage is significant only at the most aggressive GSM8K target (above).

**Live, measured** (first 10 problems of the test subset; batch 1; same setup as GSM8K):

| Policy | Accuracy | Mean tokens | Latency median | Latency mean | Latency p95 | Tokens/s | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.70 | 3822 | 71.3 s | 160.5 s | 375.4 s | 23.8 | 3799 MB |
| ST-ES 90% (self-talk, δ=4) | 0.80 | 1417 | **50.6 s** | **51.5 s** | **79.3 s** | 27.5 | **3548 MB** |

- **Tail latency.** It drops most: mean latency falls by 68% and p95 by 79%. Standard generation sometimes runs to the 8192-token cap (374 s for one problem), and early stopping cuts those runaway traces. This also lowers peak memory (−251 MB), because the KV cache stops growing.
- **Decode speed.** Standard decodes more slowly (23.8 versus 27.5 tok/s) because attention over 8k-token contexts costs more.
- **Stops and tagger cost.** ST-ES stopped on 5 of 10 problems, and all 10 live stop decisions matched an offline replay of the same tokens. The tagger costs 0.16 ms per step.
- **Sample size.** Live accuracy on 10 problems carries little weight; the 150-problem replay is the accuracy estimate.

### Tagger agreement with the paper's annotator

On 99 DS-14B MATH500 steps labelled by GPT-4o-mini (released by the authors), the rule-based tagger agrees on 40.4% of 14-way labels. One-vs-rest macro-F1 per step type:

| Step type | Macro-F1 |
|---|---:|
| exploration | 0.88 |
| final conclusion | 0.84 |
| formula substitution | 0.77 |
| self-talk | 0.69 |
| verification | 0.63 |
| problem re-statement | 0.63 |
| other types | 0.48–0.69 |

### Template summary

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | see table above (Avg@5) | not reported | not reported | not reported | not reported | 8B/14B/32B models, 5 seeds; runtime only estimated (App. F) |
| Local baseline (GSM8K) | 0.830 replay / 0.900 live | 45.2 s median | 27.6 tok/s | 3599 MB | not measured | replay n=200, live n=30 |
| Local method (GSM8K) | 0.815 replay / 0.700 live | 21.5 s median | 27.4 tok/s | 3599 MB | not measured | verification δ=0; `approximate` (rule-based tagger) |
| Local baseline (MATH-500) | 0.807 replay / 0.70 live | 71.3 s median (160.5 s mean) | 23.8 tok/s | 3799 MB | not measured | replay n=150, live n=10 |
| Local method (MATH-500) | 0.727 replay / 0.80 live | 50.6 s median (51.5 s mean) | 27.5 tok/s | 3548 MB | not measured | self-talk δ=4; `approximate` |

## Differences and limitations

- **Backbone.** We use DeepSeek-R1-Distill-Qwen-1.5B, not the 8B/14B/32B models. A 1.5B model reasons differently (more repetition, more formula steps), so the absolute numbers are not comparable with the paper's.
- **Step tagger (main deviation, `approximate`).** We use a rule-based keyword tagger instead of GPT-4o-mini labels and binary BERT taggers, because there was no API budget or labelled data.
  - On 99 GPT-4o-mini-labelled steps released by the authors (DS-14B, MATH500, not used to write the rules), it agrees on 40.4% of 14-way labels.
  - One-vs-rest macro-F1 is 0.48–0.88 depending on step type: verification 0.63 (recall 3/9), self-talk 0.69 (over-predicted), exploration 0.88. The paper's BERT taggers score about 0.8–0.9 on their own data.
  - The main failure: a leading "Wait," makes v1 say self-talk, whatever the step does.
  - A second rule set derived from our own traces did not improve agreement ([C](docs/C_tagger_analysis.md)).
  - The step types it finds are therefore only an approximation of the paper's.
- **Seeds and subsets.** One seed (42) instead of five, so we report pass@1, not Avg@5. Subsets have 150–200 problems; GPQA, MMLU-Pro and AIME were not run.
- **Token cap.** 4096 (GSM8K) and 8192 (MATH) generated tokens instead of 32768. The same cap applies to the baseline and to ST-ES. Traces that hit the cap count as wrong unless an early stop rescues them.
- **Prompt.** The paper's standard prompt is the raw question (as in the authors' code). We append the DeepSeek model card's "reason step by step … \boxed{}" directive, which helps a 1.5B model produce a parseable answer.
- **Exit prompt.** The paper does not give its exit prompt. Ours closes the thinking block and opens `\boxed{`. The answer is decoded greedily and stops when the box closes (≤100 tokens).
- **Monitoring scope.** Steps after `</think>` (the final-answer section) are never used to stop. The paper does not discuss this case.
- **Minimal step size.** The main results split at every `"\n\n"`, as in Eq. 1, arXiv v2 and the authors' code (k = 0). The proceedings version's minimal step size k (Algorithm 1) has no value in the paper. It is implemented and ablated separately in [B](docs/B_min_step_size.md).
- **Baselines.**
  - The paper's "token-count baseline" stops after k steps of any type (`StepCountStopping`). It is calibrated with the same 85/90/95% rule.
  - The four prompt-guided baselines were run on GSM8K test only, not on MATH-500 ([A](docs/A_prompt_guided_baselines.md)).
- **Offline replay versus live run.** Like the paper, accuracy and token counts come from offline replay over cached generations. Latency and peak memory come from one live run of the final configuration on a small subset (batch size 1). Replay-derived runtimes, where shown, are labelled *estimated*.
- **Cached-generation batching.** Traces are sampled in continuation rounds: per-row prefill into a static KV cache, rounds of 1024 tokens, re-queued until EOS or the cap. This changes the random stream, not the sampling distribution.
  - Runs that were interrupted (GPU-memory fixes, a session break) resumed from the finished traces. The GSM8K-train and MATH-train trace files were completed in more than one session (188 of 200 and 111 of 150 records come from an earlier session); each manifest records this.
  - Finished traces are never regenerated, so each trace is a complete sample from one session.
- **Correctness checking.** Math-Verify runs without timeouts on Windows, where its subprocess-based timeout fails.
- **Not measured.** Energy (`not_measured`).

## Reproduction checklist

- [x] Exact paper link and version verified. The official AdaptFM accepted-papers page (fetched 2026-11-25) links `openreview.net/forum?id=bNn5OusO22` to this exact title and author list. The latest version of the paper (the proceedings PDF) matches arXiv v2 except for Algorithm 1 and contains no code link (see [D](docs/D_reproduction_and_review.md)).
- [x] Base model and dataset revisions pinned (registry entries)
- [x] Baseline runs on the same setup (standard and step-count, same traces, prompts and cap)
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported (replay quality and tokens; live latency and peak memory)
- [x] Licence/data terms checked (model MIT; GSM8K and MATH MIT; MATH-500 card states no licence and is recorded as `unknown`)
