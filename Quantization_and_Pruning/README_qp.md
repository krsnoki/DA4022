# Quantization, Pruning and Compression

## Purpose

Implement methods that reduce weight/activation precision, parameter count, effective model depth/width, or executed tokens. Shared quantization and pruning primitives should live here; methods whose main contribution is routing, attention, or runtime policy should remain in their primary module and call shared compression utilities.

## Suggested structure

```text
Quantization_and_Pruning/
├── README_qp.md
├── quantization/
│   ├── ptq/
│   ├── qat/
│   ├── weight_only/
│   └── activation/
├── pruning/
│   ├── structured/
│   ├── unstructured/
│   ├── layer_depth/
│   └── token_activation/
├── configs/
├── tests/
└── experiments/
```

## Quantization paper shortlist

- [AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization](https://openreview.net/forum?id=ddOlFn6d0B)
- [CafeQ: Calibration-free Quantization via Learned Transformations and Adaptive Rounding](https://openreview.net/forum?id=CBtx52vftl)
- [COAT: COrrelation-Aware Orthogonal Transform for LLM Quantization](https://openreview.net/forum?id=6JGbm48VoG)
- [Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models](https://openreview.net/forum?id=wxKn35RAhW)
- [Leech Lattice Vector Quantization for Efficient LLM Compression](https://openreview.net/forum?id=HFk5TQ2ILj)
- [What Matters for NVFP4 Training? A Scaling Study of Low-Precision Pre-Training Recipes](https://openreview.net/forum?id=jlkIyaG32w)
- [Activation Quantization of Vision Encoders Needs Prefixing Registers](https://openreview.net/forum?id=p6y2soVAlT)
- [Zero-Shot Quantization for Vision-Language-Action Models via Trajectory Curvature and Attention Guidance](https://openreview.net/forum?id=9Ljgbk6H3O)
- [Modality-Aware Block Rotation for Vision-Language-Action Model Quantization](https://openreview.net/forum?id=07e99XvHxF)
- [SRA-MoE: Output-Aware Selective Router Alignment for MoE Quantization](https://openreview.net/forum?id=H0NoX02erJ) — coordinate with `Mixture_of_Experts/`.

## Pruning/compression paper shortlist

- [LEAP: Learnable End-to-End Adaptive Pruning of Large Language Models](https://openreview.net/forum?id=tIsX30TADF)
- [SubspacePath Pruner: Inference-time Pruning via Probe-based Representation–Parameter Coupling](https://openreview.net/forum?id=nsIOhilpBE)
- [Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning](https://openreview.net/forum?id=QuWWYol2nP)
- [Weight Concentration Regularization for Improving Pruning Robustness Under High Sparsity](https://openreview.net/forum?id=sRdWDNWenw)
- [Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs](https://openreview.net/forum?id=U37zxXMEE0)
- [SparseSAM: Structured Sparsification of Activations in Segment Anything Models](https://openreview.net/forum?id=RzsvMA4pyG)
- [SFPruner: Single-Forward Visual Token Subset Selection for Resource-Efficient Multimodal Foundation Model Inference](https://openreview.net/forum?id=d9H5uBnHeL)
- [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://openreview.net/forum?id=UZ5V7ub8mi) — cross-list with `Mixture_of_Experts/`.
- [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://openreview.net/forum?id=ZqRh99OQEK) — cross-list with `Mixture_of_Experts/`.

See [`PAPER_INDEX.md`](../PAPER_INDEX.md) for other cross-module papers and the [official AdaptFM list](https://adaptfm.gitlab.io/accepted-papers).

## Shared implementation requirements

For quantization, specify weight/activation/KV precision, group size, symmetric/asymmetric settings, calibration algorithm and sample count, rounding/scale rules, supported operators, and whether the implementation actually executes low-bit kernels or stores quantized weights but dequantizes for compute.

For pruning, specify what is removed (weight, channel, head, layer, token, activation, expert), the sparsity pattern, whether the resulting shape/layout is hardware-efficient, and whether any recovery fine-tuning is performed.

## Required tests and reports

- Compare against a full-precision or uncompressed baseline.
- Test save/load and numerical drift.
- Report model size, quality, latency, throughput, peak memory, and calibration cost.
- Verify the actual runtime dtype/kernel; do not call a method INT4-speed if it dequantizes to FP16 before every operation.
- Report sparsity and actual speed independently; unstructured sparsity may not accelerate standard dense kernels.
- Keep calibration and evaluation datasets separate and record calibration seed/sample count.
