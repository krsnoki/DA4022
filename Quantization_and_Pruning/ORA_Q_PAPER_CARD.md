# Paper implementation card: Structural Outlier-Aware Post-Training Quantization for Monocular Depth Estimation (ORA-Q)

- **Paper / OpenReview:** https://openreview.net/forum?id=oqacjavnry (AdaptFM @ ICML 2026)
- **Primary module:** `Quantization_and_Pruning/`
- **Secondary modules:** `Backbones/` (wrapper, registry, model card), `Dataloaders/` (NYUv2 loader, registry, dataset card), `Performance_Metrics/` (depth metrics, result row)
- **Owner / reviewer:** maiHydrogen / not assigned
- **Issue:** https://github.com/krsnoki/DA4022/issues/3
- **Status:** partial-reproduction (result rows: `partial`). See "Status and comparison with the paper".
- **Upstream code / licence:** none found (checked the paper and GitHub on 2026-11-20). Reimplemented from the paper.
- **Backbone IDs:** `vision.depth-anything-v2-small`
- **Dataset IDs and split:** `vision.nyuv2.test` (654 Eigen test images), `vision.nyuv2.calibration` (16 Eigen train images, seed 0)
- **Hardware assumptions:** one CUDA GPU (results measured on a Kaggle Tesla T4, torch 2.11.0+cu130). All unit tests run on CPU.

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
- **Reproduction command** (from the repo root, with `MAT` and `SPLITS` set to your copies of `nyu_depth_v2_labeled.mat` and `splits.mat`):

  ```bash
  python Quantization_and_Pruning/experiments/ora_q/run_ora_q.py --mat "$MAT" --splits "$SPLITS" --methods fp32 minmax percentile ora_q --num-images 654 --seed 0 --revision 5426e4f0f36572d16453bbda7a8389317b1bef99 --rediagnose --role-report /tmp/ora_q_role_report.json --out /tmp/ora_q_results.csv
  ```

  The last three options keep a new run from reusing the committed role report or appending to the committed CSV. The committed results were produced at commit `26518ab`, where the runner wrote `status=measured`.
- **Tests:** `python -m pytest -q Quantization_and_Pruning/tests/test_ora_q.py` (CPU, tiny random model, no download). Last local run: 17 passed, 1 expected failure (Python 3.14.4, torch 2.15.0+cpu, transformers 5.24.1).

## Results

Local numbers are from `experiments/ora_q/results.csv`: DAv2-Small at revision `5426e4f0f36572d16453bbda7a8389317b1bef99`, 654 Eigen test images, seed 0, batch 1, input 518x686, output 480x640, Tesla T4, torch 2.11.0+cu130, commit `26518ab`. Quantized rows use fake quantization with float32 compute. Latency: model forward only (no preprocessing) on the first test image, 10 warm-up + 50 timed runs, `torch.cuda.synchronize()` around each. Throughput: images/s, 1000 / median latency. Peak memory: `torch.cuda.max_memory_allocated`. Energy: not measured. Paper rows are paper-reported, not local.

| Source | delta1 | AbsRel | Latency median / p95 (ms) | Throughput (img/s) | Peak memory (MB) | Energy | Calibration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper, ORA-Q (RTX 4090) | 0.8507 | 0.1200 | not reported | not reported | not reported | not reported | not reported |
| Local FP32 | 0.9557 | 0.0617 | 77.28 / 78.15 | 12.94 | 310.0 | not measured | n/a |
| Local W4A4 MinMax | 0.4573 | 0.2962 | 165.33 / 166.29 | 6.05 | 698.3 | not measured | 1.9 |
| Local W4A4 Percentile | 0.4668 | 0.2915 | 166.58 / 167.81 | 6.0 | 697.0 | not measured | 27.9 |
| Local W4A4 ORA-Q | 0.6606 | 0.2002 | 187.4 / 189.28 | 5.34 | 698.1 | not measured | 28.6 |

The runner's calibration timer also includes the diagnosis when the diagnosis runs in that call.

Sanity check on the first 20 test images (from the run notes; its CSV is not committed): FP32 delta1 0.9584, W16A16 delta1 0.9583.

### Status and comparison with the paper

The repo lists status labels but does not define them, so the label is our own conservative choice.

- The delta1 order MinMax < Percentile < ORA-Q < FP32 holds.
- ORA-Q is 0.6606 against 0.8507 in the paper, 0.1901 lower. The headline number is not reproduced.
- ORA-Q is 0.1938 above our Percentile baseline. In the paper the difference is 0.1832.
- Percentile is 0.4668 against 0.6675 in the paper. MinMax is 0.4573 against 0.4873. Our MinMax and Percentile differ by only 0.0095.
- FP32 is 0.9557 against 0.9736 in the paper, so part of the setup differs even without quantization.
- We have not found the cause of these gaps. Evaluation protocol, preprocessing, checkpoint and the unstated choices below are untested candidates.

### Local role report

From `experiments/ora_q/role_report.json` (tau = 0.15, conv g = 8):

| Operator | S | Clip score | Role | Calibrator | Mode |
|---|---:|---:|---|---|---|
| matmul1 | 0.0788 | 0.0602 | clip-tolerant | percentile | static |
| fc2 | 0.1123 | 0.2718 | clip-sensitive | minmax | static |
| proj | 0.1014 | 0.0353 | clip-tolerant | percentile | static |
| conv | 0.1489 | 0.2187 | clip-sensitive | minmax | static |
| matmul2 | 0.4816 | n/a | input-dependent | minmax | dynamic |

Calibration train indices: 22, 66, 105, 241, 352, 402, 683, 684, 862, 881, 897, 999, 1112, 1171, 1311, 1391.

### Ablation: conv g = 16

The paper gives g = 8 in Sec. 4 and g = 16 in Table 2. We ran ORA-Q once more with g = 16 (`experiments/ora_q/results_ablation_conv_g16.csv`).

| conv g | delta1 | AbsRel | Latency median (ms) | Peak memory (MB) |
|---:|---:|---:|---:|---:|
| 8 (headline, committed config) | 0.6606 | 0.2002 | 187.4 | 698.1 |
| 16 (ablation) | 0.6438 | 0.2091 | 189.35 | 698.4 |

- The g = 16 setting was a config edit on Kaggle and is not committed. The CSV's commit field (`26518ab`) has g = 8.
- It reused the g = 8 role report. Roles, calibrators and modes do not depend on group size, because the clip score clips each whole input. The S value and granularity shown for conv do not describe the g = 16 run.
- This is one run each. The ablation ran separately, and percentile calibration samples at random, so the 0.0168 delta1 difference may not come from group size alone. It does not show which value the paper used.

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
- Fake quantization only: integer kernels are not used. In our run every quantized model had higher latency and higher peak memory than FP32, so these numbers show no runtime speed or memory benefit.
- Model size was not measured. The repo asks for it in quantization reports.
- One seed and one full run per configuration. The seed is set once per script call, and percentile calibration samples at random, so results can depend on which methods ran earlier.
- Save/load: a saved `state_dict` reloads with no output drift only into a model that was built with the same config and role report and then calibrated, which creates the scale buffers. Loading into an uncalibrated model fails. This is recorded as an expected-failure test. Both tests use the tiny model.
- Not recorded for the Kaggle run: Python, transformers, driver and CUDA runtime versions, and accelerator memory.
- NYUv2 checksums (in the dataset card) come from copies downloaded again on 2026-11-27. The files used in the reported run were not hashed. With the new `splits.mat`, seed 0 gives the same 16 calibration indices as the role report.
- Only DAv2-Small on NYUv2. KITTI, the other Depth Anything variants and the cross-dataset / cross-model cosine checks (Table 2, Fig. 1) are not reproduced.
- Paper Table 3 lists identical numbers for RepQ-ViT on NYUv2 and KITTI; likely a copy error in the paper.

## Reproduction checklist

- [x] Exact paper link verified (title and URL match the official AdaptFM accepted-paper list)
- [x] Base model revision and dataset checksums pinned (checksums taken after the run, see limitations)
- [x] Baselines run on the same setup (same weights quantizer, images, seed, metric)
- [x] Tests and seed handling added (see the seed and save/load limitations)
- [ ] Resource and quality metrics reported (model size and energy not measured)
- [ ] Licence/data terms checked (model: apache-2.0; NYUv2: no licence text found on the dataset page)
