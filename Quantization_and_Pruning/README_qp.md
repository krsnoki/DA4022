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
- [COAT: COrrelation-Aware Orthogonal Transform for LLM Quantization](https://openreview.net/forum?id=6JGbm48VoG) · [Implementation card](quantization/ptq/coat/PAPER.md)
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

## Run COAT

Run commands from the repository root in an isolated Python 3.11 environment. The CPU profile calibrates, quantizes, reloads its artifact, and compares the three variants on held-out synthetic tokens. Once the dependencies are installed, the dry-run is offline.

```bash
python -m pip install -r Quantization_and_Pruning/requirements-coat-cpu.txt
COAT_RUN=$(mktemp -d)
python -m Quantization_and_Pruning.experiments.run_coat --dry-run --output-dir "$COAT_RUN"
python -m pytest -q
```

For the 7B profile, use a separate Linux CUDA environment and one 48 GiB RTX A6000. Obtain access to the gated Meta checkpoint first and set `COAT_RUN` on a volume with room for the checkpoint, datasets and outputs. The CLI validates the pinned order-172 factor before loading the model. [Meta checkpoint and access terms](https://huggingface.co/meta-llama/Llama-2-7b-hf) · [Sloane factor](https://neilsloane.com/hadamard/had.172.will.txt)

```bash
python -m pip install -r Quantization_and_Pruning/requirements-coat-gpu.txt
COAT_RUN="$HOME/coat-llama2-7b"
mkdir -p "$COAT_RUN"
export HF_HOME="$COAT_RUN/hf"
hf auth login
curl -fL https://neilsloane.com/hadamard/had.172.will.txt -o "$COAT_RUN/had.172.will.txt"
python -m Quantization_and_Pruning.experiments.run_coat \
  --config Quantization_and_Pruning/configs/coat_llama2_7b.yaml \
  --hadamard-172 "$COAT_RUN/had.172.will.txt" \
  --cache-dir "$HF_HOME/hub" --output-dir "$COAT_RUN"
```

The run directory contains `report.json`, the versioned adaptation output, and the fixed generation inputs. The 7B profile uses 128 WikiText2 training windows of 2048 tokens, GPTQ, and packed W4A4KV4 with FP16 compute. The config pins the evaluation revisions and harness version. Keep generated binary files outside the PR. [7B config](configs/coat_llama2_7b.yaml) · [Measured CPU and pending GPU results](../Performance_Metrics/README_metrics.md#coat-benchmark)

### TinyLlama on a T4

Use the GPU requirements above and one 16 GB NVIDIA T4. The pinned TinyLlama checkpoint needs no Hugging Face login. The profile calibrates on 128 WikiText-2 training windows of 512 tokens, scores WikiText-2 test perplexity in 1024-token windows, and times a 512-token prefill and 128 fixed decode steps. `report.json` includes FP16, rotation-only FP16 and packed W4A4KV4 with FP16 compute, along with peak CUDA allocated memory. [TinyLlama config](configs/coat_tinyllama_1b.yaml) · [Sloane's order-44 factor](https://neilsloane.com/hadamard/had.44.pal.txt)

```bash
COAT_RUN="$HOME/coat-tinyllama-t4"
mkdir -p "$COAT_RUN"
curl -fL https://neilsloane.com/hadamard/had.44.pal.txt -o "$COAT_RUN/had.44.pal.txt"
CUDA_VISIBLE_DEVICES=0 python -m Quantization_and_Pruning.experiments.run_coat \
  --config Quantization_and_Pruning/configs/coat_tinyllama_1b.yaml \
  --hadamard-44 "$COAT_RUN/had.44.pal.txt" --output-dir "$COAT_RUN"
```

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
