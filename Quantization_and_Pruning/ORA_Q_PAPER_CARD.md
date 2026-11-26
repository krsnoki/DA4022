# Paper implementation card: Structural Outlier-Aware Post-Training Quantization for Monocular Depth Estimation (ORA-Q)

- **Paper / OpenReview:** https://openreview.net/forum?id=oqacjavnry (AdaptFM @ ICML 2026)
- **Primary module:** `Quantization_and_Pruning/`
- **Secondary modules:** `Backbones/` (wrapper, registry, model card), `Dataloaders/` (NYUv2 loader, registry, dataset card), `Performance_Metrics/` (depth metrics, result row)
- **Owner / reviewer:** <your GitHub handle> / not assigned
- **Issue:** <link>
- **Status:** <partial-reproduction | reproduced, after your run>
- **Upstream code / licence:** none found (checked the paper and GitHub on 2026-11-20). Reimplemented from the paper.
- **Backbone IDs:** `vision.depth-anything-v2-small`
- **Dataset IDs and split:** `vision.nyuv2.test` (654 Eigen test images), `vision.nyuv2.calibration` (16 Eigen train images, seed 0)
- **Hardware assumptions:** one CUDA GPU (results measured on <Kaggle T4>). All unit tests run on CPU.

## Research summary

### Problem
4-bit post-training quantization (PTQ) of ViT-based depth models (Depth Anything) loses a lot of accuracy. Existing PTQ methods apply one calibration rule to every operator, but activation outliers play different roles in different operators.

### Method
ORA-Q diagnoses each operator from 16 unlabeled calibration images, then gives it its own group size, calibrator and scaling mode.

1. **Structural analysis.** For each group g and image i, range r = max - min. Stability score S = mean over groups of std over images of log r. Low S: the range pattern is the same for every image, so static scales work. The attention-value product (matmul2) is input-dependent and gets dynamic scales.
2. **Distributional analysis.** Clip one operator's input at its 99.5th percentile in the FP model and measure the relative L1 change of the depth output. Score < tau = 0.15: clip-tolerant (outliers only stretch the range), use percentile calibration. Otherwise clip-sensitive (outliers carry depth information), use min-max.
3. **Quantization.** Uniform asymmetric fake quantization per group. Paper settings: matmul1 per dim (d = 64), fc2 g = 64, proj g = 32, conv g = 8, matmul2 dynamic per adjacent source-token pair (g = 2). Percentile p is searched in {0.999, 0.9999, 0.99999} by MSE.

### Expected behaviour (paper-reported, DAv2-S, NYUv2, W4A4, Table 3)

| Method | delta1 | AbsRel |
|---|---:|---:|
| FP32 | 0.9736 | 0.0513 |
| MinMax | 0.4873 | 0.2815 |
| Percentile | 0.6675 | 0.1935 |
| ORA-Q | 0.8507 | 0.1200 |

## Implementation

| Part | File |
|---|---|
| Fake quantizer, grouping, percentile search, weight quantizer | `quantization/ptq/ora_q/quantizer.py` |
| ActQuantizer, QuantLinear/Conv2d, attention function, model surgery, calibration | `quantization/ptq/ora_q/qmodules.py` |
| Stability score, clip score, role report | `quantization/ptq/ora_q/diagnose.py` |
| Config + role report -> quantized model | `quantization/ptq/ora_q/build.py` |
| Configs | `configs/ora_q/{minmax,percentile,ora_q}.yaml` |
| Experiment script, results, role report | `experiments/ora_q/` |
| Tests | `tests/test_ora_q.py` |

- **Input contract:** FP32 Depth Anything V2 Small (HF transformers 5.x), 16 calibration images, bit-widths (W4A4).
- **Output contract:** `role_report.json` (per-operator S, clip score, role, granularity, calibrator, mode), a fake-quantized model in memory, delta1 / AbsRel, latency and peak memory rows in `results.csv`.
- **Reproduction command:**
  `python Quantization_and_Pruning/experiments/ora_q/run_ora_q.py --mat <nyu_depth_v2_labeled.mat> --splits <splits.mat> --methods fp32 minmax percentile ora_q --num-images 654`
- **Tests:** `pytest -q Quantization_and_Pruning/tests/test_ora_q.py` (16 tests, CPU, about 5 s, no download)

## Results

Local numbers from `experiments/ora_q/results.csv`. Latency: model forward only, batch 1, 10 warm-up + 50 timed runs, `torch.cuda.synchronize()` around each. Peak memory: `torch.cuda.max_memory_allocated`. Energy: not measured.

| Source | delta1 | AbsRel | Latency median (ms) | Peak memory (MB) | Notes |
|---|---:|---:|---:|---:|---|
| Paper, ORA-Q | 0.8507 | 0.1200 | | | RTX 4090, not reported |
| Local FP32 | | | | | |
| Local W4A4 MinMax | | | | | |
| Local W4A4 Percentile | | | | | |
| Local W4A4 ORA-Q | | | | | |

**Local role report:** <paste S and clip score per operator from role_report.json>

## Differences and limitations

Choices the paper does not state, and what we did:

| Item | Paper | This implementation |
|---|---|---|
| Calibration images | 16 random, split not given | 16 from the NYUv2 Eigen **train** split, seed 0 (repo rule: never test) |
| conv group size | g = 16 in Table 2, g = 8 in Sec. 4 | g = 8 |
| Threshold on S | not given | S is reported; matmul2 is dynamic by architecture, as the paper states |
| Weight quantization | not given | 4-bit per output channel min-max, same for all methods |
| qkv, fc1 activations | per-tensor percentile (Table 5 base) | same |
| matmul1 "per-dim" | per dim, d = 64 | one scale per head and per dim |
| matmul2 pairs | A[:, G_k] and V[G_k, :] | per head, pairs of adjacent source tokens, min-max over queries / head dim |
| Percentile tails | not given | two-sided (1 - p, p) |
| Clip score | top 0.5% to 99.5th percentile | per image, top side, all layers of one operator class at once |
| Combining S over layers | not given | mean over layer instances |
| Calibration order | not given | all quantizers observe in one FP pass |
| Not quantized | not given | patch embedding conv, ConvTranspose2d upsamplers, LayerNorm, softmax, GELU, residual adds |
| Evaluation | "standard practice" | least-squares scale and shift in inverse-depth space per image, Eigen crop, depth (0.001, 10] m |

Other limitations:
- Fake quantization only: integer kernels are not used, so the quantized model is **slower** than FP32 and uses no less memory at run time. Latency here measures the fake-quant overhead, not INT4 speed. Weight storage at 4 bits would be about 8x smaller than FP32 in theory (not measured).
- Only DAv2-Small on NYUv2. KITTI, the other Depth Anything variants and the cross-dataset / cross-model cosine checks (Table 2, Fig. 1) are not reproduced.
- Paper Table 3 lists identical numbers for RepQ-ViT on NYUv2 and KITTI; likely a copy error in the paper.

## Reproduction checklist

- [x] Exact paper link verified
- [ ] Base model revision and dataset checksums pinned (fill after run)
- [x] Baselines run on the same setup (same weights quantizer, images, seed, metric)
- [x] Tests and seed handling added
- [ ] Resource and quality metrics reported (fill after run)
- [ ] Licence/data terms checked
