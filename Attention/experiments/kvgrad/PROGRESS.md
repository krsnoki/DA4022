# KVgrad contribution: progress log

Running record of what has been done for krsnoki/DA4022#8, in order, including problems met and how they
were solved. The summary of results lives in [`PAPER_CARD.md`](PAPER_CARD.md); how to run things is in
[`README.md`](README.md).

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-18 | Implement KVgrad (AdaptFM / ICML 2026) in `Attention/`; issue #8 opened | KV-cache eviction belongs to the Attention module per `PAPER_INDEX.md` |
| 2026-11-24 | Build on NVIDIA kvpress `0.5.5` instead of writing the pipeline from scratch | The paper's own experiments use kvpress; kvpress `0.5.5` already contains `KVgradPress` (Eq. 8) and KVzip / KVzip+ baselines |
| 2026-11-24 | Own contribution = (1) implement the paper's score variants Eqs. (6), (7), (8) ourselves and test them against kvpress, (2) measured T4 latency / memory benchmarks, (3) ablation on Qwen3-0.6B | kvpress has only the final score and reports accuracy only |
| 2026-11-24 | Hardware: Google Colab T4, fp32, `sdpa` attention, Qwen3-0.6B only | No local GPU; T4 lacks bf16 and FlashAttention-2; larger models do not fit KVgrad's backward pass |
| 2026-11-25 | Compression ratios 0.5 / 0.75 / 0.875 | Paper notation 2:1 / 4:1 / 8:1 |
| 2026-11-25 | Sample RULER per task with a fixed seed and saved row ids | Random sampling of 1% missed a whole task (`vt`); every method must see identical inputs |

## Phase 1: setup and first baseline (2026-11-24 to 2026-11-25)

**Git.** Forked `krsnoki/DA4022` to `mathangiVS/DA4022`, added `upstream`, created the shared `kvgrad`
branch, set the git identity, and pushed a first commit (`c5f8feb`) adding a `.gitignore` that keeps results
and model weights out of the repository.

**Colab environment.** T4 runtime; `pip install "kvpress[eval]==0.5.5" standard-pipes`; kvpress source
cloned at tag `v0.5.5` for its `evaluation/` scripts; results written to Google Drive.

| Component | Version |
|---|---|
| Python | 3.13.16 |
| torch | 2.14.1+cu130 (CUDA 13.0) |
| transformers | 5.2.0 |
| kvpress | 0.5.5 |
| datasets | 4.8.5 |
| GPU | Tesla T4, 14.6 GB usable |

**Problems and fixes.**

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pipes'` on `import kvpress` | Python 3.13 removed `pipes`; kvpress 0.5.5 pins `fire<0.7`, which imports it | `pip install standard-pipes` (now in `requirements-colab.txt`) |
| `TypeError: Object of type set is not JSON serializable` from `evaluate.py` | In Colab `!` commands, text inside `{...}` is evaluated as Python, which turned the `model_kwargs` argument into a set | Write the settings to a YAML file and pass `--config_file` |

**Smoke test** (one 1,457-token passkey example, single measurement, includes first-call warm-up, so not a result):

| Method | Answer | Time | Peak memory |
|---|---|---:|---:|
| no compression | 71432 (correct) | 7.32 s | 3,292 MB |
| KVzip, ratio 0.5 | 71432 (correct) | 4.02 s | 4,119 MB |
| KVgrad, ratio 0.5 | 71432 (correct) | 5.52 s | 6,126 MB |

**First RULER baseline** with kvpress's `evaluate.py`: `no_press`, `fraction=0.01` (65 random examples of
6,500), seed 42, fp32, `sdpa`; 3.6 min (~2.8 s per example).

| Task | String match | Paper (no press) |
|---|---:|---:|
| niah_single_1 / 2 / 3 | 100 / 100 / 100 | 100 / 100 / 99.4 |
| niah_multikey_1 / 2 / 3 | 100 / 100 / 100 | 97.8 / 91.6 / 90.0 |
| niah_multiquery | 100 | 98.0 |
| niah_multivalue | 95 | 90.7 |
| fwe | 72.2 | 79.5 |
| qa_1 / qa_2 | 20 / 25 | 58.2 / 33.4 |
| cwe | 0 | 0.5 |
| vt | not sampled | 78.1 |

Consistent with the paper given ~5 examples per task; `vt` was missed by random sampling, which motivated
per-task sampling in Phase 2.

## Phase 2: benchmark harness and dry run (2026-11-25)

**Added (commit `402ad5c`):**

- `run_ruler.py`: runs several presses on one fixed, per-task-stratified RULER sample; records RULER string
  match (same metric as kvpress), latency split into prefill + compression and decoding, peak GPU memory,
  theoretical kept KV-cache size, and environment metadata; writes `summary.csv` in the schema of
  `Performance_Metrics/README_metrics.md`; resumes after a disconnect.
- `configs/ruler4k_qwen3_0.6b.yaml`, `requirements-colab.txt`, `README.md`.
- `Attention/tests/test_kvgrad_run_ruler_smoke.py`: tiny random Qwen3 + toy RULER file, CPU only; checks
  all four presses end to end, output schema, the scoring function and resuming. 3 tests pass locally and on Colab.

**Finding while testing.** kvpress evicts entries by replacing their keys with "fake keys" that attention
ignores, rather than deleting them (`kvpress/attention_patch.py`: "does not reduce peak memory"). Measured
peak memory therefore shows the scoring overhead, not cache savings; the script also reports the
theoretical kept-cache size. The tiny test model needed a query bias so that such fake keys exist, which
trained models satisfy naturally.

**Dry run:** `--per_task 2` (26 examples: 2 from each of the 13 tasks), all 10 configurations, one
disconnect in the middle (finished configurations were skipped on restart, as designed). About 56 min of
measured GPU time plus ~5 min of loading and warm-up.

| Method | Ratio | Quality (macro string match) | Latency median | Peak memory |
|---|---:|---:|---:|---:|
| no_press | 0 | 76.92 | 2,575 ms | 4,271 MB |
| kvzip | 0.5 | 71.79 | 11,612 ms | 5,609 MB |
| kvzip | 0.75 | 64.74 | 11,689 ms | 5,609 MB |
| kvzip | 0.875 | 55.19 | 11,564 ms | 5,609 MB |
| kvzip_plus | 0.5 | 75.64 | 12,591 ms | 5,609 MB |
| kvzip_plus | 0.75 | 75.00 | 12,487 ms | 5,609 MB |
| kvzip_plus | 0.875 | 56.73 | 12,409 ms | 5,609 MB |
| kvgrad | 0.5 | 75.64 | 18,376 ms | 9,169 MB |
| kvgrad | 0.75 | 75.64 | 18,357 ms | 9,169 MB |
| kvgrad | 0.875 | 75.96 | 18,706 ms | 9,169 MB |

Latency breakdown (from `dryrun/summary_2pertask_seed42.csv`):

| Method | Prefill + compression (median) | vs. no-compression prefill | Decode (median) |
|---|---:|---:|---:|
| no_press | 1,723 ms | 1.0x | 859 ms |
| kvzip | ~9,970 ms | ~5.8x | ~1,270 ms |
| kvzip_plus | ~10,900 ms | ~6.3x | ~1,310 ms |
| kvgrad | ~16,750 ms | ~9.7x | ~1,270 ms |

(Averages of the medians over the three ratios; scoring cost does not depend on the ratio. Median context
3,807 tokens, median answer 16-18 tokens.)

The extra cost of every press is in the scoring step (KVgrad's ~1.7x KVzip's). Decoding is ~1.5x slower with a
press, because kvpress computes fake keys for the masked entries at every decoding step.

Comparison at 8:1 with the paper's Table 7 (macro over 13 tasks, computed by us): no press 78.2,
KVzip 55.6, KVzip+ 60.1, KVgrad 72.2. Our dry run gives 76.9 / 55.2 / 56.7 / 76.0: same ordering,
with KVgrad close to no compression while KVzip and KVzip+ lose ~20 points. With 2 examples per task these
are indicative only.

**Problems and fixes.**

| Symptom | Cause | Fix |
|---|---|---|
| `git_commit` recorded as `402ad5c-dirty` on Colab | Running the tests created `__pycache__` / `.pytest_cache` folders that git did not ignore | Ignore rules added in `Attention/tests/.gitignore` and the experiment `.gitignore` |
| Model version could change between sessions | `model_revision: null` downloads the latest Qwen3-0.6B | Pinned the revision used in the dry run, `c1899de289a04d12100db370d81485cdf75e47ca`, in the config |

**Saved:** the dry-run `summary.csv` is committed as `dryrun/summary_2pertask_seed42.csv` (its `commit` field reads
`402ad5c-dirty` for the reason in the table above; the code was unchanged).

**Cost estimate for the full run** (all 10 configurations ≈ 130 s of GPU time per sampled example):
20 per task ≈ 9.4 h, 10 per task ≈ 4.7 h, 8 per task ≈ 3.8 h.

## Full baseline run (started 2026-11-25)

- Decided: 10 examples per task (130 rows), all 10 baseline configurations in one Colab command (resumable),
  pinned model revision. Config committed as `per_task: 10` (commit `fcbfc14`).
- Work handed over to a second team member, who started the run on their own Colab / Drive at 16:38 UTC
  (`kvgrad_results/phase2_full`, code at commit `80ff1aa`, clean). Environment identical to the dry run.
- Colab offered no GPU for a while before the run; Kaggle (30 free T4 hours/week, background runs) was noted as
  a fallback.
- Note: commit `80ff1aa` changed one word in the course file `Attention/README_attention.md`
  ("Implement" -> "implement"); to be reverted before the pull request so the PR only touches KVgrad files.

## Phase 3: our implementation of the score variants (2026-11-25)

**Added:**

- `Attention/methods/context_compression/kvgrad/`: `scores.py` (the four score variants of the paper's
  ablation: Eq. 6 `taylor_sum`, `taylor_max`, Eq. 7 summed `bound_sum`, Eq. 8 `bound_max`), `press.py`
  (`KVgradVariantsPress`, built on kvpress's KVgrad replay and eviction, with a score cache), README with
  inputs/outputs as required by `Attention/README_attention.md`.
- `run_ruler.py`: presses `kvgrad_bound_max`, `kvgrad_bound_sum`, `kvgrad_taylor_max`, `kvgrad_taylor_sum`,
  option `share_scores`, column `scoring_cached` and flag in `summary.csv`. Existing presses unchanged.
- `configs/ruler4k_qwen3_0.6b_ablation.yaml`: the four variants at 8:1, 16:1, 32:1 (where the paper's Table 4
  shows the variants separating for Qwen3-0.6B).
- Tests (12 in total, CPU): `test_kvgrad_variants.py` and an end-to-end variants test in the smoke test; the
  tiny-model builder moved to `Attention/tests/_kvgrad_tiny_model.py`.

**Verification:** our Eq. (8) reproduces kvpress's `KVgradPress` scores within rtol 1e-5 / atol 1e-7 and evicts
exactly the same KV pairs (one and several replay chunks); Eq. (6) equals the autograd value of
$v \cdot \nabla_v \Phi$ to rtol 1e-10; the Eq. (7) bound holds elementwise.

**Problems and fixes.**

| Symptom | Cause | Fix |
|---|---|---|
| Taylor scores differed from the float64 reference by ~1e-7 | the gradient was always cast to float32 | promote only half-precision gradients |
| Untimed warm-up would put the first example's scores in the cache | warm-up used the shared-cache press | warm-up uses a press without cache |
| Variants test saw cached scores inside the first configuration | toy dataset reused identical contexts across tasks (cache correctly hit) | toy contexts made unique per row |

**Ablation cost estimate:** first configuration ~20 s per example (one replay for all four variants), the other
11 ~3.5 s per example; about 2 h for 130 rows.

## Next steps

1. Finish the full baseline run (`phase2_full`), commit its `summary.csv`, update the paper card.
2. Phase 4: run `configs/ruler4k_qwen3_0.6b_ablation.yaml` into the same output folder; commit results and plots.
3. Phase 5: revert the change to `Attention/README_attention.md`, complete `PAPER_CARD.md`, add result rows to
   `Performance_Metrics/README_metrics.md`, link the card from `PAPER_INDEX.md`, open the pull request (`Closes #8`).
