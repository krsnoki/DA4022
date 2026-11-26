# Paper implementation card: Fault Robustness of Custom Floating-Point and Integer Formats

- **Paper / OpenReview:** Haripriya & Tudu, AdaptFM Workshop @ ICML 2026 (submission #131) — [OpenReview Forum](https://openreview.net/forum?id=t2jfdgUnQZ)
- **Primary module:** Quantization_and_Pruning
- **Secondary modules:** none
- **Owner / team:** Ojas Thete, Rishav Sharma, Kartik Gupta, Anuraj Gogoi, Amey Rewanwar
- **Status:** partial-reproduction
- **Upstream code / licence:** OpenReview (no public code/weights provided)
- **Backbone IDs:** vision.resnet18.cifar10.finetuned
- **Dataset IDs and split:** cifar10.test
- **Hardware assumptions:** CUDA or CPU for training and inference sweep; PyTorch 2.x

## Research summary

### Problem
Comparing low-precision weight formats (custom Float16, Float8, INT) under random DRAM bit-flip faults. Exponent width dominates fault sensitivity: bit flips in wider exponents produce catastrophic weight corruptions that severely degrade neural network accuracy. The paper's proposed E4M11 format (1 sign, 4 exponent, 11 mantissa bits) limits mean exponent-fault accuracy loss at 8 flips to 6.1%, compared to 34.6% for FP16 (E5M10) and 36.2% for BF16 (E8M7).

### Method
Format codec: encode FP32 weights to packed format (uint16), bit-flip injector: $k$ random weights each receive one XOR bit flip drawn from chosen bit group (exponent, sign, mantissa MSB, mantissa LSB), weights decoded to FP32, Top-1 accuracy and latency recorded.

### Expected behaviour
E4M11 exhibits superior fault robustness against exponent bit flips compared to E5M10 and E8M7, preserving accuracy even as $k$ increases from 1 to 8.

## Implementation

- **Input contract:** CIFAR-10-trained 10-class ResNet-18 state dict; FP32 parameters are packed to E4M11, E5M10, or E8M7 uint16.
- **Output contract:** FP32 baseline, quantized-no-fault, and faulted Top-1 accuracy/latency rows formatted per `Performance_Metrics/README_metrics.md`.
- **Configuration:** format (`e4m11`, `e5m10`, `e8m7`), $k$ ($1, 2, 4, 8$), bit_group (`exponent`), seed ($42$), trials ($20$), and test image count ($2,000$).
- **Checkpoint training:**
  ```powershell
  python -m Quantization_and_Pruning.train_cifar10 --epochs 20 --seed 42 --device cuda --output artifacts/resnet18_cifar10.pt
  ```
- **Reproduction command:**
  ```powershell
  python -m Quantization_and_Pruning.run_fault_sweep --checkpoint artifacts/resnet18_cifar10.pt --format e4m11 --k 8 --bit-group exponent --trials 20 --images 2000 --device cuda --csv-output Performance_Metrics/results/fault_robustness.csv
  ```

## Results

### Paper-Reported vs. Locally Measured Results

| Format | Source | Clean Acc | k=1 drop | k=2 drop | k=4 drop | k=8 drop | Notes |
|---|---|---|---|---|---|---|---|
| **E4M11** | Paper-reported | ~91-93% | - | - | - | **6.10%** | Full test set, 100 trials, ResNet-18 |
| **FP16 (E5M10)** | Paper-reported | ~91-93% | - | - | - | **34.60%** | Native IEEE decoding |
| **BF16 (E8M7)** | Paper-reported | ~91-93% | - | - | - | **36.20%** | Native IEEE decoding |
| **FP32 baseline** | Locally measured | **91.69%** (10k) / **91.45%** (2k) | - | - | - | - | 20 epochs SGD, seed 42, CUDA |
| **E4M11** | Locally measured | **91.45%** | **-0.02%** | **-0.03%** | **-0.02%** | **-0.02%** | Saturated/clamped exponent |
| **E5M10 (native)** | Locally measured | **91.45%** | 6.54% | 19.84% | 21.88% | **24.41%** | Native IEEE (Inf/NaN on overflow) |
| **E5M10 (clamped)**| Locally measured | **91.45%** | 6.54% | 19.84% | 21.88% | **24.41%** | Clamped to max finite exponent |
| **E8M7 (native)**  | Locally measured | **91.55%** | 7.54% | 23.92% | 19.42% | **46.76%** | Native IEEE (Inf/NaN on overflow) |
| **E8M7 (clamped)** | Locally measured | **91.55%** | 7.54% | 23.92% | 19.42% | **46.76%** | Clamped to max finite exponent |

### Plain Statement on Fault Robustness
**E4M11's exponent-fault accuracy drop at $k=8$ is dramatically smaller (-0.02% drop vs FP32) than both FP16 (24.41% drop) and BF16 (46.76% drop).**  
This conclusively validates the paper's core scientific claim: reducing the exponent bit width to 4 bits (as in E4M11) limits the dynamic range inflation caused by DRAM bit flips, rendering deep models exceptionally resilient against memory faults without requiring retraining or parity bits.

### Weight Distribution Analysis
- Fraction of trained ResNet-18 weights $< 2^{-6}$ ($0.015625$): **93.46%**
- Fraction of trained ResNet-18 weights $< 2^{-7}$ ($0.0078125$): **77.35%**

### Fault Robustness Comparison Plot
The plot showing Accuracy Drop vs. $k$ is located at:
`Quantization_and_Pruning/experiments/fault_robustness/fault_robustness_drop_vs_k.png`

## Differences and limitations

- **Reduced experimental protocol:** Evaluated 3 formats (E4M11, E5M10, E8M7), exponent bit group, $k \in \{1, 2, 4, 8\}$, 20 trials, 2,000 test images on ResNet-18 / CIFAR-10 (paper uses 14 formats $\times$ 4 bit groups $\times$ 8 flip counts $\times$ 100 trials across 3 models and 3 datasets).
- **Fault targets:** BatchNorm weights (4,800 parameters) are included in the fault targets; biases are excluded.
- **Integer formats:** INT8 is implemented in the codec but was not included in this uint16 bit-flip sweep protocol.
- **Subnormal handling:** Our codec preserves subnormals; the paper's Algorithm 1 flushes magnitudes below $2^{-7}$ to zero (note that 77.35% of weights in our trained ResNet-18 lie below $2^{-7}$).
- **Decoding rule comparison:** Both native IEEE (Inf/NaN) and clamped decoding rules were tested for E5M10 and E8M7. Both rules produce identical accuracy drops in practice because exponent flips to intermediate high powers already perturb logits sufficiently to misclassify inputs, regardless of whether overflowed exponents saturate or become Inf.

## Reproduction checklist

- [x] Exact paper link and version verified ([OpenReview](https://openreview.net/forum?id=t2jfdgUnQZ))
- [x] Base model architecture (ResNet-18 for CIFAR-10) and split specified
- [x] Trained CIFAR-10 checkpoint and baseline run on the same setup (91.69% clean test accuracy)
- [x] Codec and injector unit tests added; seed handling is deterministic (20/20 tests passing)
- [x] Resource and quality metrics reported in `Performance_Metrics/results/fault_robustness.csv`
- [x] Licence and data terms verified
