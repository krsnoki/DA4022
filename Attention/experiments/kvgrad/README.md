# KVgrad experiments (RULER, Qwen3-0.6B, NVIDIA T4)

Benchmark scripts for the paper **KVgrad: Query-Agnostic KV Cache Eviction via Gradient-based Global
Importance Scoring** ([OpenReview](https://openreview.net/forum?id=cg1wTCJjjk), AdaptFM workshop, ICML 2026).
Tracking issue: krsnoki/DA4022#8.

Status: `implementing` (benchmark harness and our score implementation tested; full baseline run and ablation run in progress).

Our implementation of the scores lives in [`Attention/methods/context_compression/kvgrad/`](../../methods/context_compression/kvgrad/README.md).

- [`PAPER_CARD.md`](PAPER_CARD.md): paper implementation card (method, expected and measured results, limitations)
- [`PROGRESS.md`](PROGRESS.md): log of what has been done, problems met and fixes, next steps

## What `run_ruler.py` measures

For every press and compression ratio, on the **same** stratified RULER sample (`per_task` examples from
each of the 13 tasks, fixed seed, row ids saved to `samples_*.csv`):

| Field | Meaning |
|---|---|
| `string_match_per_task`, `quality_value` | RULER string match, identical to kvpress's `calculate_metrics.py`; `quality_value` is the macro average over tasks |
| `latency_median_ms`, `latency_p95_ms` | Wall time of one full pipeline call (tokenisation, prefill, compression, decoding), batch size 1, after `warmup` untimed calls, with CUDA synchronisation |
| `prefill_compress_median_ms`, `decode_median_ms` | The same call split at `generate_answer`: prefill + scoring/eviction vs. answer generation |
| `peak_memory_mb` | `torch.cuda.max_memory_allocated()` per example (maximum over examples) |
| `kv_cache_mb_kept_median` | Theoretical KV-cache size of the kept entries at the median context length |

**Memory caveat.** kvpress evicts KV pairs by *masking* them with fake keys, not by freeing them, so
`peak_memory_mb` does **not** shrink with the compression ratio. It measures the extra memory needed for
*scoring* (KVgrad's backward pass, KVzip's replay). The memory a real eviction backend would save is
reported separately as `kv_cache_mb_kept_median`.

Presses (`kvpress==0.5.5`, definitions as in its `evaluate_registry.py`):

| Name | Press |
|---|---|
| `no_press` | no compression (baseline) |
| `kvzip` | `KVzipPress(compression_ratio=r)` |
| `kvzip_plus` | `KVzipPress(compression_ratio=r, kvzip_plus_normalization=True)` |
| `kvgrad` | `KVgradPress(compression_ratio=r)`, the upstream kvpress implementation of the paper's Eq. (8) |
| `kvgrad_bound_max` | **ours**, Eq. (8) (KVgrad); same scores and evictions as `kvgrad` (tested) |
| `kvgrad_bound_sum` | **ours**, Eq. (7) summed over probes (paper ablation "w/o PSP") |
| `kvgrad_taylor_max` | **ours**, signed first-order Taylor terms, max over probes ("w/o MAS") |
| `kvgrad_taylor_sum` | **ours**, Eq. (6) first-order Taylor ("w/o both") |

Compression ratio `r` is the fraction of KV pairs evicted: 0.5, 0.75, 0.875, 0.9375 and 0.96875 correspond
to the paper's 2:1, 4:1, 8:1, 16:1 and 32:1.

With `share_scores: true` (default), the `kvgrad_*` presses of one run share a score cache: the first
configuration replays each context once and stores all four variants; later configurations on the same
context skip the replay. Their latency then excludes scoring, which `summary.csv` flags
(`scoring_cached_fraction` in `metrics.json`); quality is unaffected.

## Run on Google Colab (T4)

```bash
git clone -b kvgrad https://github.com/mathangiVS/DA4022.git /content/DA4022
pip install -r /content/DA4022/Attention/experiments/kvgrad/requirements-colab.txt
# restart the Colab session once after installing
cd /content/DA4022/Attention/experiments/kvgrad
python run_ruler.py --config configs/ruler4k_qwen3_0.6b.yaml \
    --output_dir /content/drive/MyDrive/kvgrad_results/phase2
```

Score-variant ablation (Phase 4), into the same output folder so it re-uses the same sampled rows:

```bash
python run_ruler.py --config configs/ruler4k_qwen3_0.6b_ablation.yaml \
    --output_dir /content/drive/MyDrive/kvgrad_results/phase2_full
```

Finished configurations are skipped and interrupted ones resume from their partial `predictions.csv`,
so after a Colab disconnect, re-run the same command. Use `--presses` / `--compression_ratios` /
`--per_task` to run a subset, for example a quick dry run with `--per_task 2`.

## Tests (CPU, no downloads)

```bash
python -m pytest Attention/tests -q
```

- `test_kvgrad_run_ruler_smoke.py`: tiny random Qwen3 and a toy RULER file; runs every press through the
  script and checks the outputs, the summary schema, the scoring function, resuming, and the score cache
  (our `kvgrad_bound_max` must answer exactly like kvpress's `kvgrad`).
- `test_kvgrad_variants.py`: our scores against autograd and against kvpress's `KVgradPress` (see the
  method README).

## Environment notes

- Colab (Nov 2026): Python 3.13.16, torch 2.14.1+cu130, transformers 5.2.0, kvpress 0.5.5, Tesla T4 16 GB.
- `standard-pipes` is required on Python 3.13, because kvpress 0.5.5 pins `fire<0.7`, which imports the
  removed `pipes` module.
- fp32 is used because the T4 has no bf16 (fp16 risks overflow); the paper ran Qwen3-0.6B on an RTX A5000 and does not state its precision.

Model weights and raw outputs are not committed (see `.gitignore`). Only small summary tables are:
`dryrun/summary_2pertask_seed42.csv` (dry run, 2 examples per task); final results will be added when the
full runs are complete.
