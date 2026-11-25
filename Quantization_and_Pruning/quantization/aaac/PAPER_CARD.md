# Paper implementation card: AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization

- **Paper / OpenReview:** https://arxiv.org/abs/2605.08692v2 (v2, 2026-06-05) · https://openreview.net/forum?id=ddOlFn6d0B
- **Primary module:** `Quantization_and_Pruning/quantization/aaac/`
- **Secondary modules:** `Backbones/Generative_Pretrained/` (loader), `Dataloaders/calibration/` (C4 sampling, activation energy), `Dataloaders/wikitext2.py` (evaluation windows), `Performance_Metrics/` (result rows)
- **Owner / reviewer:** <GitHub handles>
- **Status:** partial-reproduction
- **Issue:** #<issue number>
- **Upstream code / licence:** not found. Implemented from the paper (§§3–4, Algorithm 1).
- **Backbone IDs:** `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987`
- **Dataset IDs and split:** calibration `allenai/c4` (en, train) @ `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, 4 windows × 2,048 tokens, seed 42; evaluation `wikitext` / `wikitext-2-raw-v1`, test split (revision not pinned)
- **Hardware assumptions:** tests on CPU. The measured run fitted on CPU and evaluated on Apple Silicon (MPS, FP16). CUDA is supported but was not used for the reported numbers.

## Research summary

### Problem
Standard 4-bit weight quantization (RTN, AWQ, GPTQ) maps every weight group onto the same fixed 16-level grid. Real weight groups have different shapes: some have one outlier and most values near zero, others are spread evenly. A single uniform grid wastes levels on both kinds. Methods that fix this (OmniQuant, QuIP#) take hours to run.

### Method
Each quantized layer learns two small 16-entry codebooks instead of using a fixed grid. Weights are split into groups of g = 128 and divided by the group's absmax scale. Every selection group of S weights picks whichever codebook gives the lower reconstruction error, where each input column is weighted by its calibration activation energy I[k] = Σ_t X[t,k]². The codebooks are then refined with weighted Lloyd (k-means) updates: 3 outer rounds of re-selection × 10 inner updates. When S = g, the codebook choice is stored in the sign bit of the scale, which is always positive. So the only extra storage is the two BF16 tables, 64 bytes per layer.

### Expected behaviour
Paper-reported (Table 2, g = 128, W4A16, WikiText-2 perplexity, Qwen2.5-0.5B): FP16 13.07, RTN 15.54, AWQ 15.01, AAAC S = 128 14.47, AAAC S = 16 14.27. The paper reports quantization in minutes on one GPU (under 2 minutes for a 7B model on an H100) and no extra memory beyond the model. It does not give a quantization time for Qwen2.5-0.5B.

## Implementation

- **Input contract:** a pretrained causal LM; per-layer activation energy `{layer name: FP32 [in]}` from C4 calibration; `AAACConfig(group_size=128, selection_size=128)`. Weights are `W[out, in]` with groups running along `in`.
- **Output contract (per quantized layer):** packed indices `uint8 [out, in/2]` (two 4-bit indices per byte); codebooks `bfloat16 [2, 16]`; scales `float16 [out, in/g]` (sign = selector when S = g); selector bitset `uint8` when S < g. `AAACLinear` rebuilds `abs(scale) × table[selector][index]` and runs a dense linear layer. Biases, embeddings, norms and `lm_head` are unchanged.
- **Quantized layers:** all 168 `nn.Linear` layers inside the 24 decoder layers (q, k, v, o, gate, up, down). `embed_tokens` and `lm_head` (tied) stay in FP16.
- **Configuration:** `Quantization_and_Pruning/quantization/aaac/benchmark_config.json`
- **Reproduction command:** `python -m Quantization_and_Pruning.quantization.aaac.benchmark` (from the repo root; see the folder README quickstart)
- **Tests:** `python -m pytest -q` (68 tests, CPU only, no downloads; `tests/test_aaac_smoke.py` runs the whole pipeline on a tiny random Qwen2)

## Results

Quality is WikiText-2 test perplexity over 146 non-overlapping 2,048-token windows (298,862 predicted tokens), computed from the total next-token loss. Latency is one forward pass (prefill) over a single 2,048-token window, batch 1, 2 warm-up and 10 timed runs. Memory is in MiB.

| Source | Quality (PPL ↓) | Latency median / p95 (ms) | Throughput (tok/s) | Peak memory (MiB) | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper, FP16 | 13.07 | | | | | Table 2 |
| Paper, RTN | 15.54 | | | | | Table 2 |
| Paper, AWQ | 15.01 | | | | | Table 2 |
| Paper, AAAC S = 128 | 14.47 | | | | | Table 2 |
| Paper, AAAC S = 16 | 14.27 | | | | | Table 2 |
| Local, unmodified (FP16) | 13.07 | 1471.9 / 1482.2 | 1391 | 7171 | not measured | matches paper |
| Local, RTN INT4 g = 128 | 18.16 | 1972.8 / 2057.4 | 1038 | 7360 | not measured | symmetric grid, see below |
| Local, AAAC g = 128, S = 128 | 15.01 | 2443.8 / 2914.6 | 838 | 7600 | not measured | +0.54 vs paper |

Storage and quantization cost (measured):

| Variant | Quantized layer bytes | Total checkpoint | Quantization time |
|---|---:|---:|---:|
| Unmodified | – | 988,065,536 B (942.3 MiB) | – |
| RTN | 184,504,320 B (indices 178,913,280 + scales 5,591,040) | 456,916,736 B (435.8 MiB) | 3.3 s |
| AAAC S = 128 | 184,515,072 B (same + codebooks 10,752 = 168 × 64) | 456,927,488 B (435.8 MiB) | 1006.7 s (CPU fit) |

The remaining 272,412,416 B in both quantized checkpoints is the FP16 embedding (tied with `lm_head`), which is not quantized.

Environment of the measured run: macOS 27 (arm64), Apple Silicon MPS, Python 3.9.6, torch 2.8.0, transformers 4.57.6. Raw output: `results/full_results_s128.json`.

## Differences and limitations

- **RTN baseline is weaker than the paper's.** Ours is symmetric absmax INT4 with the same scales and layers as AAAC (levels −7..7; code −8 is stored but never used). We get 18.16 against the paper's 15.54. The paper's RTN is probably an asymmetric min/max grid with a zero point, which uses all 16 levels, but we have not confirmed this. So our RTN → AAAC gain (−3.15 PPL) overstates the gain the paper reports (−1.07). Compare our AAAC with the paper's AAAC, not with our RTN.
- **AAAC is 0.54 PPL worse than the paper** (15.01 vs 14.47). The paper leaves several details open, and our choices may differ from the authors': scale rule (FP16 absmax), codebook initialisation (weighted quantiles of a peaky/flat split), error weighting in normalised space, and tie rules (see the folder README §3.6). Evaluation also ran in FP16 on MPS. We have not tested which of these causes the gap.
- **Hardware differs.** We used MPS + CPU, not CUDA. The 1,007 s quantization time is a CPU fit and is not comparable to the paper's GPU timings.
- **Latency and throughput come from the dense reference runtime.** `AAACLinear` rebuilds the full weight matrix on every forward pass, so AAAC and RTN are slower than FP16 here. This says nothing about native INT4 kernels, which we did not build.
- **Peak memory can't be compared across variants.** On MPS it is `max(driver allocated, process peak RSS)`, and peak RSS is a lifetime maximum. All three variants ran in one process, so each value includes the ones before it.
- **S = 16 was not run on the full model.** It is implemented and tested (codec, fit, smoke test) but has no measured perplexity.
- **The WikiText-2 dataset revision is not pinned.**
- **Energy was not measured.**

## Reproduction checklist

- [x] Exact paper link and version verified (arXiv v2)
- [x] Base model and dataset revisions pinned (model and C4; WikiText-2 not pinned)
- [x] Baseline runs on the same setup (same eval tokens, scales and layers)
- [x] Tests and seed handling added (68 CPU tests; calibration seed 42; deterministic fit)
- [x] Resource and quality metrics reported (energy not measured)
- [x] Licence/data terms checked (no upstream code; Qwen2.5 Apache-2.0; C4 ODC-BY; WikiText CC BY-SA)
