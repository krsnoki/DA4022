# D: Reproduction check and review

Part of the ST-ES implementation ([paper card](../PAPER_CARD.md)). This page records what was checked, how, and what was found.

## 1. Fresh-clone check (CPU)

On 2026-11-26 we cloned the branch from the public fork into an empty directory and ran the full test suite there. This catches dependence on files that exist only in the development checkout.

```bash
git clone --branch st-es-step-tagging --single-branch https://github.com/devanshjangid2005/DA4022.git DA4022_fresh
cd DA4022_fresh
python -m pytest Adaptive_Stopping/tests Dataloaders/tests Performance_Metrics/tests -q
```

| Item | Value |
|---|---|
| Commit | `6d37f36` |
| Result | **176 passed** in 8 s (CPU only, `CUDA_VISIBLE_DEVICES=""`). The fixes below added tests; the suite now has 188. |
| Platform | Windows 11 10.0.26200, Python 3.12.14, torch 2.11.0+cu128, transformers 5.24.1 |

**Final audit, 2026-11-26.** The check was repeated in a **fresh environment**: a new virtual environment holding PyTorch 2.11.0+cu128 and then only `requirements-st_es.txt`. Results:
- **188 passed** with the backbone tokenizer in the local Hugging Face cache.
- **186 passed, 2 skipped** with an empty cache. The skipped tests are the two in `test_steps.py` and `test_prompt_guided.py` that render through the real tokenizer.

## 2. Integration check (five PRs)

The submission is five PRs in dependency order. Every PR only adds or edits its own files, so no two PRs touch the same path.

| PR | Content | Needs |
|---|---|---|
| 1 | Shared backbone, dataset and scorer entries (`Backbones/`, `Dataloaders/`, `Performance_Metrics/quality`) | nothing |
| 2 | Step-level stopping framework (`common/`, `NoStopping`, `StepCountStopping`, requirements) | nothing |
| 3 | ST-ES method and experiment pipeline (taxonomy, tagger, policy, calibration, experiment scripts, main configs, paper card) | PRs 1 and 2 |
| 4 | Extensions: prompt-guided baselines (A), minimal step size k (B), tagger analysis (C) | PRs 1–3 |
| 5 | Main GSM8K and MATH-500 results, metrics rows, and the reproduction kit (D) | PRs 1–3 |

The check rebuilds the project from upstream exactly as the PRs will. Each step runs in a fresh worktree of upstream `krsnoki/DA4022@main` (`368db37`). Patches are made by `git format-patch` and applied with `git am`, or merged as GitHub does with merge commits:

| Step | Result |
|---|---|
| Each of the five patches on bare upstream `main` | applies; each PR's diff contains only its own files |
| PR 1 alone; PR 2 alone | 21 and 30 tests pass; neither needs the other |
| PRs 1 + 2 + 3 | 136 tests pass; the method and pipeline need neither PR 4 nor PR 5 |
| PRs 1–3 + 4; PRs 1–3 + 5 | 180 and 144 tests pass; PRs 4 and 5 are independent of each other |
| All five, merged in order with merge commits | 188 tests pass. The final tree is identical to the integrated branch (same git tree hash); 105 changed files. |
| Other merge orders, and a `core.autocrlf=true` checkout | identical tree |
| PR 3 without PRs 1 and 2 | its tests fail at import, as expected: it needs the registries and the framework |
| Commit metadata after `git am` by another person | author unchanged (the implementer), committer = the person applying it |

The script (`make_handover.py`) and its generated report live in the submission's hand-over folder, outside this repository.

Two problems that only show up when the work is split were found and fixed:
- `tests/test_run_all.py` used the smoke config of the reproduction kit, so the method PR alone failed one test. It now uses `gsm8k.yaml`.
- The prompt-guided rows were in the main metrics file, `results/st_es.csv`. They now live in their own `results/st_es_prompt_guided.csv`, next to the code that writes them.

## 3. End-to-end smoke run (GPU)

`configs/st_es/gsm8k_smoke.yaml` uses 20 calibration and 20 test problems and 3 live requests, with otherwise the same settings as the main run. It exercises every stage (generation, calibration replay, test replay, live benchmark, report) in about 10 minutes.

```bash
python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k_smoke.yaml --run-dir <tmp>
```

We ran it from the fresh clone with a separate run directory, in two attempts.

**First attempt (commit `6d37f36`): found a real bug.**
- With a non-default `--run-dir`, `run_all` passed the directory only to the generation stage. Calibration wrote its `selected.json` to the default run directory, and the evaluation stage could not find it.
- The main runs never hit this, because they use the default directory.
- Fixed in `a39cd1a`: `replay_sweep` and `live_benchmark` take `--run-dir` and `run_all` passes it to every stage. The new test `tests/test_run_all.py` checks the wiring for every stage and validates all configs.

**Second attempt (commit `a39cd1a`): all stages passed.** `run_all` exited 0, and `make_report` exited 0.

| Stage | Result |
|---|---|
| Generation, 20 calibration + 20 test problems | 149 s and 140 s; 273 and 198 tok/s, with small batches in the tail |
| Calibration replay (full grid; 736 forced answers in 21 s) | Standard 0.90 / 2030 tokens; selects formula substitution δ=5 (85%) and exploration δ=0 (90/95%) |
| Test replay | Standard 0.85 / 1387 tokens; ST-ES exploration δ=0 0.80 / 781 (−44%); step-count k=14 0.80 / 816 |
| Live, 3 requests | ST-ES stopped on 2 of 3, all answers correct, median latency 36.8 s → 23.4 s; every stop matched an offline replay of the same tokens |
| Wall time | About 10 minutes from scratch (about 6 minutes with cached traces) |

The numbers come from 20 problems and only show that the pipeline works; they are not results.

**Final audit (2026-11-26): both datasets, fresh environment, empty run directories.**
- **Environment and data.** The runs used the fresh environment of section 1. The model and dataset files came from the local Hugging Face cache at their pinned revisions, with `HF_HUB_OFFLINE=1`.
- **GSM8K.** It ran on a tree holding upstream `main` plus PRs 1–3 and the main result files. The extensions and the reproduction kit were absent, and the smoke config was passed by path. So the method and pipeline need neither to run.
- **MATH-500.** It ran on the fully integrated tree (all five PRs) with the new `math500_smoke.yaml` (10 calibration and 10 test problems, 2 live requests, otherwise the settings of `math500.yaml`).

| | GSM8K smoke, without extensions and kit | MATH-500 smoke, integrated |
|---|---|---|
| Exit codes | `run_all` 0 (11.1 min), `make_report` 0 | `run_all` 0 (20.3 min), `make_report` 0 |
| Calibration replay | Standard 0.90 / 2030 tokens; selects formula substitution δ=5 (85%) and exploration δ=0 (90/95%) | Standard 0.70 / 3534 tokens; 462 forced answers; selects self-talk δ=1 (85%) and self-talk δ=5 (90/95%) |
| Test replay | Standard 0.85 / 1387; ST-ES exploration δ=0 0.80 / 781 (−44%); step-count k=14 0.80 / 816 | Standard 0.80 / 3735; ST-ES self-talk δ=5 0.90 / 1930 (−48%); step-count k=70 0.90 / 2137 |
| Live | 3/3 correct for both policies; ST-ES stopped on 2; median 36.0 s → 22.4 s; replay agreement 1.0 | 1/2 correct for both policies; ST-ES stopped on 1; mean 214.9 s → 45.6 s (one standard request ran to the 8192-token cap in 372 s); replay agreement 1.0 |
| Tracked files changed by the run | 0 (only `results/smoke_gsm8k/` is new) | 0 (only `results/smoke_math500/` is new) |

The GSM8K numbers are identical to the second attempt above, which ran from a fresh clone in the old environment: generation with a fixed seed and batch schedule reproduces exactly.

**Another hazard found while reviewing the report step.** `make_report` named its output folder and metrics rows after the evaluation dataset. A smoke run made inside the main repository would therefore have overwritten the main GSM8K results. Fixed in `e8eb50a`: configs can set `report_name` and `write_metrics: false`, and the smoke config does.

## 4. Paper version and links

- **Official AdaptFM accepted-paper list:** checked 2026-11-25. It links `openreview.net/forum?id=bNn5OusO22` to this exact title and author list.
- **Latest version of the paper:** **verified.** The team downloaded the latest version, a 22-page proceedings-format PDF with author names, built 2026-06-08. We compared it line by line with arXiv v2. The only substantive difference is the minimal step size k in Algorithm 1 (see [B](B_min_step_size.md)).
- **Code link:** none. The PDF's only code link points to Math-Verify, a third-party tool, and no ST-ES code release was found.
- **OpenReview page:** our scripts could not fetch it (a timeout on 2026-11-25, then a bot challenge, `HTTP 403`, on 2026-11-26). Nothing depends on it, because the downloaded PDF and the official list settle the version. Reading the page's reviews is optional.

## 5. Review notes

These are points a reviewer should know, in rough order of how much they affect the results.

1. **Approximate tagger.**
   - The rule-based tagger agrees with GPT-4o-mini on only 40% of 14-way labels, and verification recall is 3/9 (see [C](C_tagger_analysis.md)).
   - ST-ES results measure "ST-ES with this tagger". They are labelled `approximate`.
2. **Live and replay accuracy differ on GSM8K. This was diagnosed and is not a bug.**
   - In the 30-problem live run, ST-ES scored 0.70 against 0.90 for standard generation; the 200-problem replay shows only −1.5 points.
   - A re-run of the 6 failing problems showed that every live stop equals an offline replay of the same tokens, and that ST-ES follows standard's tokens exactly up to the stop.
   - The losses come from δ=0 stopping at the model's first check of an *intermediate* result. In the replay, these losses are offset by early stops that recover traces which hit the token cap (see the paper card).
3. **Sampling and batching.** Cached traces are sampled in batched continuation rounds. The distribution is unchanged, but the exact token streams depend on the batch schedule. Live runs sample at batch size 1, so live and replay traces of the same problem differ.
4. **Token caps.** 4096 for GSM8K and 8192 for MATH, against 32768 in the paper. 14% (GSM8K) and 17% (MATH-500) of standard test traces hit the cap and are scored as generated.
5. **Math-Verify on Windows.** Its timeouts are disabled because the subprocess-based timeout fails there. A pathological expression could hang the scorer; none did in our runs.
6. **transformers 5.x API.** The code uses `generate(cache=...)`, `StaticCache` and `DynamicCache.layers`, and pins `transformers==5.24.1`.
7. **GPU memory on Windows.** Exceeding dedicated memory spills into shared memory instead of raising OOM. The scripts cap the allocator at 85% and use static KV caches with chunked prefill. Budgets are tuned for 8 GB and can be raised on larger GPUs (`--kv-token-budget`).
8. **Report provenance.** `make_report.py` stamps the commit at report time. The traces themselves are identified by their sha256 in `provenance.json`.
9. **Independent recomputation (final audit).** A separate script re-derived every reported number from the raw traces and the forced-answer cache. It reused only the segmenter, tagger and scorer, and re-implemented the stop rule, aggregation, calibration selection, bootstrap and live statistics. Covered:
   - trace checksums;
   - all 304 calibration-sweep rows per dataset;
   - the selected configurations;
   - held-out accuracy, tokens, tokens saved and stop rates;
   - every live latency and memory statistic;
   - the MATH-500 matched-token comparison.

   There were **0 mismatches**. Bootstrap intervals agree within resampling noise; the 10,000-resample check moves some bounds by 0.5 points. Re-scoring each forced answer that a reported configuration uses reproduced its stored correctness flag. That covers the whole calibration grid and the selected test configurations.
10. **Scope of the live/replay agreement claim.** The audit field `replay_agrees` was added after the 30-problem GSM8K live run. Live stops are therefore verified against replay on all 10 MATH-500 live requests, the 6-problem GSM8K diagnostic and the smoke runs, but not on that first GSM8K run.
11. **Other fixes made in the final audit:**
    - The dataset module was renamed from `math_reasoning.py` to `gsm8k_math.py`, because another team's open PR (#23) adds a different file at the old path.
    - The live summaries no longer record an absolute local path for `selected.json`; it is now relative to the run directory.
    - The `README_metrics.md` section now lists the ST-ES and baseline rows, as the repository README asks.
