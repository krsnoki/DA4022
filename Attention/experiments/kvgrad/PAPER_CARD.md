# Paper implementation card: KVgrad: Query-Agnostic KV Cache Eviction via Gradient-based Global Importance Scoring

- **Paper / OpenReview:** https://openreview.net/forum?id=cg1wTCJjjk (AdaptFM workshop, ICML 2026; Kwak, Joo, Hwang, Byun, Moon, Seoul National University)
- **Primary module:** `Attention/` (KV cache and long-context memory)
- **Secondary modules:** `Performance_Metrics/` (latency and memory protocol)
- **Owner / reviewer:** mathangiVS (team members: sanjayganeshprabu, teja123git, Vipulsoc, kvdhruv) / reviewer: TBD
- **Tracking issue:** krsnoki/DA4022#8
- **Status:** `implementing` (benchmark harness and our score implementation tested; full baseline and ablation runs in progress)
- **Upstream code / licence:** NVIDIA kvpress, https://github.com/NVIDIA/kvpress, Apache-2.0. Version `0.5.5` already ships `KVgradPress` (`kvpress/presses/kvgrad_press.py`), which implements the paper's final score, Eq. (8). The intermediate scores of the paper's ablation, Eqs. (6) and (7), are not in kvpress. No separate code release by the authors was found.
- **Backbone IDs:** `Qwen/Qwen3-0.6B`, Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca` (pinned in the config)
- **Dataset IDs and split:** `simonjegou/ruler`, `data_dir=4096`, split `test` (RULER as packaged by kvpress, 13 tasks x 500 examples, ~4K-token contexts); per-task stratified sample with seed 42
- **Hardware assumptions:** single NVIDIA T4 (16 GB, 14.6 GB usable) on Google Colab; fp32 (T4 has no bf16); `sdpa` attention (FlashAttention-2 unsupported on T4, `eager` rejected by KVzip/KVgrad)

## Research summary

### Problem
Long-context inference is memory-bandwidth bound: every generated token reads the whole key-value (KV)
cache. Query-agnostic eviction removes cache entries once, using only the context, so the compressed
cache can be reused for any later question. Existing query-agnostic scores (KVzip, KVzip+) judge an entry
only by its contribution to the attention output **in its own layer**, ignoring how that contribution
propagates through later layers to the model's final representation.

### Method
Like KVzip, KVgrad appends a "Repeat the previous context:" prompt followed by a copy of the context, and
uses these replay tokens $p \in P$ as probes. With $A^{\ell}_{p,c}$ the attention from probe $p$ to cache
entry $c$ at layer $\ell$, $v^{\ell}_{c}$ its value vector, $W^{\ell}_{O}$ the output projection and
$a^{\ell}_{p} = \sum_{t} A^{\ell}_{p,t}\, v^{\ell}_{t} W^{\ell}_{O}$ the attention output, the scores compared in
this project are:

| Score | Definition | Paper |
|---|---|---|
| KVzip | $S_{\ell,c} = \max_{p} A^{\ell}_{p,c}$ | Eq. (3) |
| KVzip+ | $S_{\ell,c} = \max_{p} A^{\ell}_{p,c}\, \lVert v^{\ell}_{c} W^{\ell}_{O}\rVert_2 \,/\, \lVert x^{\ell}_{p}\rVert_2$ | Eq. (4) |
| First-order Taylor | $\sum_{p} A^{\ell}_{p,c}\,\big(v^{\ell}_{c} W^{\ell}_{O}\big)\cdot \nabla_{a^{\ell}_{p}}\Phi$ | Eq. (6) |
| Magnitude-aware bound | $\lvert T^{\ell}_{p,c}\rvert \le A^{\ell}_{p,c}\,\lVert v^{\ell}_{c} W^{\ell}_{O}\rVert_2\,\lVert \nabla_{a^{\ell}_{p}}\Phi\rVert_2$ | Eq. (7) |
| **KVgrad** | $S_{\ell,c} = \max_{p}\, A^{\ell}_{p,c}\,\lVert v^{\ell}_{c} W^{\ell}_{O}\rVert_2\,\lVert \nabla_{a^{\ell}_{p}}\Phi\rVert_2$ | **Eq. (8)** |

Here $\Phi = \sum_{p} \lVert x^{L+1}_{p} \rVert_2^2$ summarises the final hidden states, so
$\nabla_{a^{\ell}_{p}}\Phi$ (one backward pass) measures how much the output depends on layer $\ell$'s
attention output at probe $p$. Eq. (6) is the first-order estimate of removing $v^{\ell}_{c}$; Eq. (7) keeps only
magnitudes because raw gradient directions are noisy; Eq. (8) replaces the sum over probes by a max so an
entry that matters to a single probe is not averaged away. The lowest-scoring fraction of entries
(`compression_ratio`) is evicted across all layers and heads.

### Expected behaviour

Score-variant ablation for Qwen3-0.6B reported in the paper's appendix Table 4 (RULER, 10% of the
examples, string match):

| Variant | 8:1 | 16:1 | 32:1 |
|---|---:|---:|---:|
| `taylor_sum`, Eq. (6) | 73.15 | 54.51 | 24.37 |
| `taylor_max` | 72.24 | 55.83 | 24.45 |
| `bound_sum`, Eq. (7) summed | 72.33 | 55.96 | 36.32 |
| `bound_max`, Eq. (8) KVgrad | 69.66 | 57.21 | 40.20 |

For this small model the variants are close at 8:1 and separate at 16:1 and 32:1, where the magnitude-aware
max (KVgrad) is best; hence the ablation is run at 8:1, 16:1 and 32:1.

Main comparison:
Paper-reported RULER string match for Qwen3-0.6B (macro average over the 13 tasks, **computed by us from
the paper's appendix Table 7**; the paper reports per-task values):

| Compression | No press | KVzip | KVzip+ | KVgrad |
|---|---:|---:|---:|---:|
| none | 78.2 | | | |
| 2:1 (`compression_ratio=0.5`) | | 78.0 | 78.1 | 78.1 |
| 4:1 (`0.75`) | | 68.5 | 72.7 | 77.1 |
| 8:1 (`0.875`) | | 55.6 | 60.1 | 72.2 |

Cost: KVgrad needs one extra forward and backward pass over the replayed context, which kvpress estimates
at 3-4x the prefill cost (KVzip: 2-3x). The paper's trainable MLP surrogate, KVgrad (MLP), removes this cost
at inference; it is out of scope here.

## Implementation

- **Input contract:** a RULER example (`context`, `question`, `answer_prefix`, `max_new_tokens`); batch size 1; contexts of ~3.4K-4K tokens.
- **Output contract:** generated answer string; per-example latency (total, prefill + compression, decode), peak GPU memory, input/output token counts; per-configuration `metrics.json`; `summary.csv` in the schema of `Performance_Metrics/README_metrics.md`.
- **Configuration:** `Attention/experiments/kvgrad/configs/ruler4k_qwen3_0.6b.yaml` (model, dtype, attention backend, sample size, seed, presses, compression ratios 0.5 / 0.75 / 0.875).
- **Reproduction command:**
  `python Attention/experiments/kvgrad/run_ruler.py --config Attention/experiments/kvgrad/configs/ruler4k_qwen3_0.6b.yaml --output_dir <dir>`
- **Own implementation:** [`Attention/methods/context_compression/kvgrad/`](../../methods/context_compression/kvgrad/README.md).
  `scores.py` computes the four score variants of the paper's ablation (Eqs. 6, 7 summed, max-Taylor, Eq. 8)
  from the replay quantities; `press.py` (`KVgradVariantsPress`) plugs them into kvpress's replay and eviction,
  with an optional cache so one replay serves all variants and ratios. Benchmark names: `kvgrad_taylor_sum`,
  `kvgrad_taylor_max`, `kvgrad_bound_sum`, `kvgrad_bound_max`.
- **Ablation configuration:** `configs/ruler4k_qwen3_0.6b_ablation.yaml` (the four variants at 8:1, 16:1, 32:1).
- **Tests:** `python -m pytest Attention/tests -q` (CPU, tiny random Qwen3, no downloads; 12 tests):
  - Eq. (6): Taylor terms summed over probes and heads equal $v \cdot \nabla_v \Phi$ from autograd (float64, rtol 1e-10).
  - Eq. (7): the magnitude bound holds for every probe and entry.
  - Reference check: our `bound_max` reproduces kvpress's `KVgradPress` scores (rtol 1e-5, atol 1e-7) and evicts exactly the same KV pairs, with one and with several replay chunks; through `run_ruler.py` it gives identical answers.
  - Score cache: cached scores evict exactly what a fresh press evicts, for every variant and ratio.
  - Benchmark script: end-to-end run of all presses, output schema, scoring function, resuming.

## Results

All local numbers so far are a **dry run** with 2 examples per task (26 examples). They check that the
pipeline works and are too small for conclusions. Quality is RULER string match (macro over 13 tasks).
Latency is the median wall time of one full pipeline call (tokenisation, prefill, compression, decoding),
batch size 1, after one warm-up call. Peak memory is `torch.cuda.max_memory_allocated()`.

| Source | Method | Ratio | Quality | Latency (median) | Throughput | Peak memory | Energy | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Paper-reported | no press | 0 | 78.2 | n/a | n/a | n/a | n/a | all 6,500 examples, RTX A5000, Table 7 macro |
| Paper-reported | KVzip / KVzip+ / KVgrad | 0.875 | 55.6 / 60.1 / 72.2 | n/a | n/a | n/a | n/a | Table 7 macro |
| Local baseline | no_press | 0 | 76.92 | 2,575 ms | ~0.39 ex/s | 4,271 MB | not measured | dry run, n=26 |
| Local | kvzip | 0.5 / 0.75 / 0.875 | 71.79 / 64.74 / 55.19 | 11,612 / 11,689 / 11,564 ms | ~0.09 ex/s | 5,609 MB | not measured | dry run, n=26 |
| Local | kvzip_plus | 0.5 / 0.75 / 0.875 | 75.64 / 75.00 / 56.73 | 12,591 / 12,487 / 12,409 ms | ~0.08 ex/s | 5,609 MB | not measured | dry run, n=26 |
| Local | kvgrad (kvpress) | 0.5 / 0.75 / 0.875 | 75.64 / 75.64 / 75.96 | 18,376 / 18,357 / 18,706 ms | ~0.05 ex/s | 9,169 MB | not measured | dry run, n=26 |

Machine-readable rows: [`dryrun/summary_2pertask_seed42.csv`](dryrun/summary_2pertask_seed42.csv).

Latency breakdown (dry run):

| Method | Prefill + compression (median) | vs. no-compression prefill | Decode (median) |
|---|---:|---:|---:|
| no_press | 1,723 ms | 1.0x | 859 ms |
| kvzip | ~9,970 ms | ~5.8x | ~1,270 ms |
| kvzip_plus | ~10,900 ms | ~6.3x | ~1,310 ms |
| kvgrad | ~16,750 ms | ~9.7x | ~1,270 ms |

(Averages of the medians over the three ratios; scoring cost does not depend on the ratio. Median context
3,807 tokens, median answer 16-18 tokens.)

Observations from the dry run (to be confirmed with the full run):

- The ordering at 8:1 matches the paper: KVzip (55.2) ≈ KVzip+ (56.7) far below KVgrad (76.0), which stays close to no compression (76.9).
- Cost per example relative to no compression: KVzip ~4.5x, KVzip+ ~4.9x, KVgrad ~7.1x. Almost all of it is in the scoring step: prefill + compression takes ~5.8x (KVzip), ~6.3x (KVzip+) and ~9.7x (KVgrad) the plain prefill, so KVgrad's scoring costs ~1.7x KVzip's.
- Decoding is ~1.5x *slower* with any press (~1.27-1.31 s vs 0.86 s for the same answer length): kvpress masks evicted keys by computing fake keys at every decoding step, so in this implementation compression adds decode work instead of saving it.
- Peak memory: KVgrad needs ~9.2 GB vs ~5.6 GB for KVzip and ~4.3 GB without compression, because the backward pass stores activations.
- Peak memory does not depend on the compression ratio. kvpress evicts by **masking** entries with fake keys rather than freeing them, so the measured peak reflects the scoring overhead, not cache savings. The theoretical KV cache is 224 KiB per token in fp32 (28 layers x 8 KV heads x 128 dims x 2 x 4 bytes), about 0.9 GB for a 4K context, of which a fraction `1 - ratio` would be kept.

## Differences and limitations

- **Partial reproduction:** one model (Qwen3-0.6B of the paper's four), RULER only (no LongBench), a stratified subset of RULER instead of all 6,500 examples used by the paper, three compression ratios (the paper also reports 16:1 and 32:1).
- **Precision and hardware:** fp32 on a T4; the paper ran Qwen3-0.6B on an RTX A5000 and does not state its precision. Absolute latencies are not comparable to the paper.
- **Upstream implementation:** the baseline KVgrad numbers use kvpress's `KVgradPress`; our own `kvgrad_bound_max` computes identical scores (tested) and is used for the ablation. Replay, hooks and eviction are reused from kvpress; the score computation is ours.
- **Ablation latency:** in the ablation run, configurations after the first reuse cached scores, so their latency excludes scoring (flagged in `summary.csv`); KVgrad's real latency comes from the baseline run.
- **Memory:** kvpress masks evicted entries, so no real memory saving is measured; theoretical kept-cache size is reported instead.
- **Latency scope:** includes tokenisation and generation of the task's answer length; not a pure prefill benchmark.
- **Not reproduced:** KVgrad (MLP) surrogate, time-to-first-token figure, larger models.
- **Environment workaround:** Colab's Python 3.13 removed `pipes`, which `fire<0.7` (required by kvpress 0.5.5) imports; fixed with the `standard-pipes` backport.
- **Sample size:** the dry run has 2 examples per task, so per-task scores move in steps of 50 points; only the full run should be cited.

## Reproduction checklist

- [x] Exact paper link and version verified
- [x] Base model revision pinned (`c1899de289a04d12100db370d81485cdf75e47ca`); dataset revision not pinned (`simonjegou/ruler` loaded at its latest version; sampled row ids are saved per run)
- [x] Baseline runs on the same setup (same sampled rows for every method)
- [x] Tests and seed handling added (12 CPU tests incl. comparison with kvpress's reference implementation; seed 42; saved sample ids)
- [x] Resource and quality metrics reported (latency, peak memory, quality; energy marked not measured)
- [x] Licence/data terms checked (kvpress Apache-2.0; RULER via kvpress Hugging Face dataset)
