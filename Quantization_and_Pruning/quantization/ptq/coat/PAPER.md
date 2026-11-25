# COAT: COrrelation-Aware Orthogonal Transform for LLM Quantization

- **Paper / OpenReview** Indranil Patra, Azhar Yousuf, Manu Mathew and Chandra Sekhar Seelamantula, 2026. [AdaptFM paper entry](https://openreview.net/forum?id=6JGbm48VoG) · [Paper PDF](https://openreview.net/pdf/00398473d87c76f95dace60f68c2caf0abb298b2.pdf)
- **Primary module** `Quantization_and_Pruning/`
- **Secondary modules** `Backbones/`, `Dataloaders/` and `Performance_Metrics/`
- **Owner / reviewer** Unassigned
- **Status** `partial`. Measured on the synthetic CPU fixture and the pretrained TinyLlama 1.1B on NVIDIA T4. The Llama-2 7B RTX A6000 run is pending.
- **Upstream code / licence** The paper provides no COAT code or code licence. The checkpoint uses Meta's Llama 2 Community License. [GPTQ reference](https://github.com/IST-DASLab/gptq) · [Meta model card](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Backbone IDs** `llama.tiny.synthetic`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, and `meta-llama/Llama-2-7b-hf`
- **Dataset IDs and split** `wikitext2` train for calibration, `wikitext2` test, `ptb` test and `c4_en` validation for perplexity. `synthetic_causal_lm` provides the local fixture inputs. [Dataset registry](../../../../Dataloaders/registry.yaml)
- **Hardware assumptions** CPU for the synthetic fixture, one NVIDIA T4 for TinyLlama, and one 48 GiB RTX A6000 on Linux CUDA for the 7B benchmark. FP16 7B weights alone require roughly 13 GiB. GPTQ replays one decoder layer at a time with calibration hidden states backed by CPU memory.

## Research summary

### Problem

Channels with very different activation variances can make 4-bit quantization lose information. Learning rotations for large models also adds calibration compute and memory.

### Method

COAT estimates activation covariance on calibration tokens and multiplies its eigenvector matrix by a normalized Hadamard matrix to equalize marginal variances. It folds the residual and headwise value rotations into weights and uses online Hadamard transforms after RoPE and the gated MLP product. The paper combines this with GPTQ and 4-bit activations and KV cache.

### Expected behaviour

With quantizers disabled, the rotated model should preserve logits within floating-point tolerance. The paper reports W4A4KV4 quality and rotation calibration cost. The reference runtime dequantizes packed weights and cache data for floating-point operations. Its latency and memory depend on that execution path.

## Implementation

- **Input contract** A registered Llama backbone and causal-LM batches with `int64[B,T]` input IDs and a `bool[B,T]` attention mask.
- **Output contract** A versioned adaptation with packed projection weights, fused floating-point tensors, online transforms, exact calibration inputs, and provenance. The benchmark writes per-variant quality and timing rows.
- **Baseline** The untouched backbone uses the same revision, test inputs and evaluation settings as COAT. A rotation-only variant measures the effect of the online transforms with quantization disabled.
- **Configuration** Both GPU profiles pin model and dataset revisions and use FP16 inference. TinyLlama calibrates on 128 windows of 512 tokens with seed 0. The 7B profile uses 128 windows of 2048 tokens with seed 0. [TinyLlama config](../../../configs/coat_tinyllama_1b.yaml) · [7B config](../../../configs/coat_llama2_7b.yaml)
- **Reproduction command** From the repository root in the CPU environment, run the command below. The GPU setup, factor download and exact CUDA command are in the quantization guide. [Run COAT](../../../README_qp.md#run-coat)
- **Tests** `python -m pytest -v` covers covariance and rotations, floating-point prefill and cached decode, GPTQ, packed KV cache, and adaptation reload.

```bash
COAT_RUN=$(mktemp -d)
python -m Quantization_and_Pruning.experiments.run_coat --dry-run --output-dir "$COAT_RUN"
```

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---|---|---|---|---|---|
| Paper-reported, Llama-2 7B FP16 | Mean PPL 16.88, mean zero-shot accuracy 61.16% | — | — | — | — | Paper Table 1 |
| Paper-reported, Llama-2 7B COAT | Mean PPL 22.38, mean zero-shot accuracy 56.62% | — | — | — | — | Paper Table 1, W4A4KV4 |
| Local FP32 baseline | Synthetic PPL 129.8261 | Prefill 2.90 ms, decode 11.28 ms | 348.05 decode tokens/s | — | — | `partial` / `synthetic-smoke`, median timings |
| Local COAT rotation-only FP32 | Synthetic PPL 129.8261 | Prefill 5.68 ms, decode 18.44 ms | 213.44 decode tokens/s | — | — | `partial` / `synthetic-smoke`, median timings |
| Local COAT W4A4KV4 with FP32 compute | Synthetic PPL 129.8717 | Prefill 14.55 ms, decode 55.59 ms | 66.58 decode tokens/s | — | — | `partial` / `synthetic-smoke`, median timings |
| Local Llama-2 7B variants | Pending | Pending | Pending | Pending | — | RTX A6000 run pending |

### Measured TinyLlama GPU run

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` ran on one NVIDIA Tesla T4 with a 512-token prompt and 128 fixed decode steps. The W4A4KV4 variant uses packed INT4 storage / FP16 compute. The metrics guide records the environment and workload for both the synthetic and GPU runs. [COAT benchmark](../../../../Performance_Metrics/README_metrics.md#coat-benchmark)

| Variant | WikiText-2 test PPL | Prefill median / p95 (ms) | Decode median / p95 (ms) | Prefill tokens/s | Decode tokens/s | Prefill peak allocated (MB) | Decode peak allocated (MB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP16 baseline | 8.918699027193576 | 76.5203805000283 / 77.68567220005025 | 3640.8193019999544 / 3727.6137939000478 | 6697.6467625801415 | 35.06186323060157 | 2302.605824 | 2230.564352 | `partial` |
| COAT rotation-only FP16 | 8.918598526694602 | 178.43607200006772 / 183.51887234980495 | 8903.963900500003 / 8956.405533099996 | 2877.444361822512 | 14.382164145491837 | 2341.226496 | 2264.99072 | `partial` |
| COAT W4A4KV4 | 11.691915222981782 | 353.917150000143 / 355.0558034499999 | 22151.110946000015 / 22241.88096125001 | 1446.0447016543008 | 5.774415324247129 | 937.375744 | 892.555264 | `partial` |

## Differences and limitations

- The paper reports covariance-derived R1/R2, random Hadamard R3/R4, WikiText2 calibration with 128 × 2048 tokens, GPTQ, per-token asymmetric activations and a 4-bit KV cache.
- Repository defaults pool valid pre-gain RMSNorm rows into global R1 and pool the attention output heads per layer into R2. GPTQ uses whole-row symmetric `[-7, 7]`, block size 128, 1% damping and fixed column order. The KV cache uses per-token, per-head affine quantization. The evaluation uses pinned dataset snapshots, a 128-window C4 subset, and pinned raw-accuracy metrics.
- The local T4 benchmark uses the pretrained TinyLlama chat checkpoint, WikiText-2 test, and the eager PyTorch reference runtime. The paper's 7B results use a different checkpoint and evaluation suite. Native INT4 kernel performance remains unmeasured.

## Reproduction checklist

- [x] Paper entry, PDF and Table 1 checked
- [x] Base model and dataset revisions pinned
- [x] Matching baseline and COAT variants run on the synthetic setup
- [x] Numerical tests and seed handling verified on CPU
- [x] Synthetic quality and timing reported
- [x] TinyLlama T4 perplexity, latency and peak allocated memory reported
- [ ] Run the matched 7B GPU benchmark and record measured results
- [ ] Licence and data access terms confirmed for the benchmark run
