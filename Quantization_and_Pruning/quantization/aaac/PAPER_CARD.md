# Paper implementation card: AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization

- **Paper / OpenReview:** https://arxiv.org/abs/2605.08692v2 (v2, 2026-06-05) · https://openreview.net/forum?id=ddOlFn6d0B
- **Primary module:** `Quantization_and_Pruning/quantization/aaac/`
- **Secondary modules:** `Backbones/Generative_Pretrained/` (loader), `Dataloaders/calibration/` (C4 sampling, activation energy), `Dataloaders/wikitext2.py` (evaluation windows), `Performance_Metrics/` (result rows)
- **Owner / reviewer:** <GitHub handles>
- **Status:** tested
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
Paper-reported (Table 2, g = 128, W4A16, WikiText-2 perplexity, Qwen2.5-0.5B): FP16 13.07, RTN 15.54, AWQ 15.01, AAAC S = 128 14.47, AAAC S = 16 14.27. Codebooks are initialised from plain quantiles of the normalised weights (Eq. 9–10). The paper reports quantization in minutes on one GPU (under 2 minutes for a 7B model on an H100) and no extra memory beyond the model. It does not give a quantization time for Qwen2.5-0.5B.

## Implementation

- **Input contract:** a pretrained causal LM; per-layer activation energy `{layer name: FP32 [in]}` from C4 calibration; `AAACConfig(group_size=128, selection_size=128)`. Weights are `W[out, in]` with groups running along `in`.
- **Output contract (per quantized layer):** packed indices `uint8 [out, in/2]` (two 4-bit indices per byte); codebooks `bfloat16 [2, 16]`; scales `float16 [out, in/g]` (sign = selector when S = g); selector bitset `uint8` when S < g. `AAACLinear` rebuilds `abs(scale) × table[selector][index]` and runs a dense linear layer. Biases, embeddings, norms and `lm_head` are unchanged.
- **Quantized layers:** all 168 `nn.Linear` layers inside the 24 decoder layers (q, k, v, o, gate, up, down). `embed_tokens` and `lm_head` (tied) stay in FP16.
- **Configuration:** `Quantization_and_Pruning/quantization/aaac/benchmark_config.json`
- **Reproduction command:** `python -m Quantization_and_Pruning.quantization.aaac.benchmark` (from the repo root; runs unmodified, `rtn_asym`, `aaac_s128_paperinit`, `aaac_s16_paperinit`; see the folder README quickstart)
- **Tests:** `python -m pytest -q` (77 tests, CPU only, no downloads; `tests/test_aaac_smoke.py` runs the whole pipeline on a tiny random Qwen2)

## Results

Quality is WikiText-2 test perplexity over 146 non-overlapping 2,048-token windows (298,862 predicted tokens), computed from the total next-token loss, FP16 evaluation. Latency is one forward pass (prefill) over one 2,048-token window, batch 1, 2 warm-up and 10 timed runs. Memory is in MiB.

| Source | Quality (PPL ↓) | Latency median / p95 (ms) | Throughput (tok/s) | Peak memory (MiB) | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper, FP16 | 13.07 | | | | | Table 2 |
| Paper, RTN | 15.54 | | | | | Table 2 |
| Paper, AAAC S = 128 | 14.47 | | | | | Table 2 |
| Paper, AAAC S = 16 | 14.27 | | | | | Table 2 |
| Local, unmodified (FP16) | **13.07** | 147.6 / 148.5 | 13,877 | 5713 | not measured | matches paper |
| Local, RTN INT4 g = 128 (zero-point) | **15.54** | 253.4 / 254.1 | 8,081 | 5745 | not measured | matches paper |
| Local, AAAC g = 128, S = 128 | **14.76** | 320.1 / 320.8 | 6,399 | 6049 | not measured | +0.29 vs paper |
| Local, AAAC g = 128, S = 16 | **14.37** | 327.9 / 329.2 | 6,245 | 6073 | not measured | +0.10 vs paper |

The paper's conclusions hold on these numbers: AAAC beats RTN at both settings (−0.78 and −1.17 PPL; paper −1.07 and −1.27), AAAC S = 128 beats the paper's AWQ figure (15.01), and S = 16 beats S = 128.

Storage and quantization cost (measured):

| Variant | Quantized layer bytes | Total checkpoint | Quantization time |
|---|---:|---:|---:|
| Unmodified | – | 988,065,536 B (942.3 MiB) | – |
| RTN zero-point | 187,299,840 B (indices 178,913,280 + FP16 scales 5,591,040 + uint8 zero points 2,795,520) | 459,712,256 B (438.4 MiB) | 0.3 s |
| AAAC S = 128 | 184,515,072 B (indices + scales + codebooks 10,752 = 168 × 64; selector in the scale sign) | 456,927,488 B (435.8 MiB) | 974.9 s (CPU fit) |
| AAAC S = 16 | 187,310,592 B (as above + selector bitset 2,795,520) | 459,723,008 B (438.4 MiB) | 1006.3 s (CPU fit) |

The remaining 272,412,416 B in each quantized checkpoint is the FP16 embedding (tied with `lm_head`), which is not quantized.

Environment: Apple Silicon Mac, MPS evaluation in FP16, AAAC fit on CPU, Python 3.12, torch 2.15.0, transformers 4.57.6, datasets 3.6.0. Raw output: `results/full_results_paper_options.json`.

## Differences and limitations

- **AAAC is still 0.29 (S = 128) and 0.10 (S = 16) PPL above the paper.** FP16 and RTN match exactly, so the evaluation protocol and baseline are right; the remaining gap is inside the AAAC fit. The paper leaves the scale formula, quantile interpolation method and tie-breaking unspecified. We use FP16 group absmax scales, linear-interpolated quantiles and lowest-index / table-0 ties. We have not tested which of these causes the gap.
- **Hardware differs.** Evaluation ran on Apple MPS in FP16 and the fit on CPU, not on an H100. Perplexity is essentially hardware-independent here (our FP16 and RTN match to 0.01), but the ~16-minute CPU fit time is not comparable to the paper's GPU timings.
- **Latency and throughput come from the dense reference runtime.** `AAACLinear` and `RTNLinear` rebuild the full weight matrix on every forward pass, so both are slower than FP16 here. This says nothing about native INT4 kernels, which we did not build.
- **Peak memory can't be compared across variants.** On MPS it is `max(driver allocated, process peak RSS)`, and peak RSS is a lifetime maximum. All variants ran in one process, so each value includes the ones before it.
- **The WikiText-2 dataset revision is not pinned.**
- **Energy was not measured.**

## Reproduction checklist

- [x] Exact paper link and version verified (arXiv v2)
- [x] Base model and dataset revisions pinned (model and C4; WikiText-2 not pinned)
- [x] Baseline runs on the same setup (same eval tokens and layers; FP16 and RTN match the paper)
- [x] Tests and seed handling added (77 CPU tests; calibration seed 42; deterministic fit)
- [x] Resource and quality metrics reported (energy not measured)
- [x] Licence/data terms checked (no upstream code; Qwen2.5 Apache-2.0; C4 ODC-BY; WikiText CC BY-SA)
