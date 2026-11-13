# Paper implementation card: Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models

- **Paper / OpenReview:** https://openreview.net/forum?id=wxKn35RAhW · arXiv: https://arxiv.org/abs/2608.20988 (v1, Pandey, Chavan, Lele, Dayal, Gupta; ICML 2026 AdaptFM workshop)
- **Primary module:** `Quantization_and_Pruning/` (quantization-aware fine-tuning, `quantization/qat/`)
- **Secondary modules:** `Attention/` (the method modifies pre-softmax attention logits); `Performance_Metrics/` (result rows)
- **Owner / reviewer:** <GitHub handles>
- **Status:** `tested` (implementation + CPU tests). GPU results: not yet produced.
- **Upstream code / licence:** **not found**. No code link on arXiv, OpenReview or the authors' pages (checked 2026-11-13). This is an independent re-implementation from the paper text.
- **Backbone IDs:** `multimodal.siglip_base_p16_224`, `multimodal.siglip_base_p16_384`, `text.qwen2_5_0_5b.base` (optional `text.qwen2_5_1_5b.base`); tests use `test.tiny_qwen2.random`, `test.tiny_siglip.random`
- **Dataset IDs and split:** `text.alpaca.train` (fine-tuning), `text.wikitext2_raw.test` (PPL), `text.wikitext2_raw.train` (GPTQ calibration), `vision.imagenet1k_256.train` (SigLIP fine-tuning + calibration, disjoint), `vision.imagenet1k_256.validation` (evaluation). See `Dataloaders/registry.yaml`.
- **Hardware assumptions:** training needs one CUDA GPU with 16 GB (Kaggle T4/P100 tested by design, not yet run). Tests and smoke runs need only CPU.

## Research summary

### Problem
Low-bit post-training quantization (e.g. W4A4) adds rounding error to the attention logits. The softmax amplifies this error most where its Jacobian is large, i.e. where attention mass is split across a few tokens. One-hot or very flat rows are much less sensitive.

### Method
For one attention row `p = softmax(z)` the Jacobian is `J = diag(p) − p pᵀ`. Its squared Frobenius norm has the closed form

```
||J||_F² = Σ p_i² − 2 Σ p_i³ + (Σ p_i²)²        ∈ [0, 1/4]
```

During a short fine-tuning phase, zero-mean Gaussian noise is added to the scaled pre-softmax logits:

```
Z = QKᵀ/√d,   Z̃ = Z + ε,   ε_i ~ N(0, σ_i²),   A = softmax(Z̃ + mask)
σ_i^(l) = clamp( α · sqrt( E_batch,heads ||J||_F² at query row i, layer l ), σ_min, σ_max )
```

Defaults: `α = 0.5`, `[σ_min, σ_max] = [0.005, 10]`, σ refreshed every 100 optimiser steps, and the first 100 steps use a constant σ = 0.1. Noise is off at inference, so the method adds **no inference cost**. A second-order expansion shows the noise acts as a curvature (Hessian-trace) penalty on the logits.

### Expected behaviour (paper-reported, not reproduced here)
| Setting | Baseline | + noise | Note |
|---|---:|---:|---|
| SigLIP-B/16-384, RepQ-ViT W4A4, ImageNet top-1 | 32.5 | 69.8 | the "+37%" claim is **+37.3 percentage points** absolute |
| SigLIP-B/16-384, ERQ W4A4 | 51.2 | 60.1 | |
| Qwen2.5-3B, SpinQuant W4A4KV4, Avg of 7 zero-shot | 60.34 | 62.62 | |
| Llama-3.2-3B, SpinQuant W4A4KV4, Avg of 7 zero-shot | 60.25 | 60.15 | slightly worse |
| Llama-3.2-3B, GPTQ W4A4, WikiText-2 PPL | 454.35 | 496.38 | worse |

The LLM results in the paper are mixed. The paper also contains no ablation against **uniform noise of the same magnitude**. This implementation adds that ablation as `mode: uniform_matched`.

## Implementation

| File | What |
|---|---|
| `jacobian.py` | closed-form `‖J‖_F²` (O(N) per row, float32); explicit `diag(p) − ppᵀ` for tests |
| `controller.py` | `JacobianNoiseConfig` + `JacobianNoiseController`: warm-up, update schedule, σ per layer and query position, clamping, ablation modes, state dict |
| `hf_attention.py` | `"jacobian_noise"` attention implementation registered with `transformers.AttentionInterface` (and an eager mask builder); `attach_jacobian_noise` / `detach_jacobian_noise` |
| `../../activation/fake_quant.py` | shared simulated quantization: per-channel weights, per-token activations, optional Q/K quantization |
| `../../weight_only/gptq.py` | block-sequential GPTQ on the same quantization grid |

- **Input contract:** any Hugging Face model whose attention modules have `q_proj`/`k_proj` and dispatch through `ALL_ATTENTION_FUNCTIONS`. Tested on Qwen2 (causal, GQA, padding) and SigLIP (vision and text towers) with transformers 4.57.6. `include=` restricts noise to a sub-model, e.g. `"vision_model"`.
- **Output contract:** in `eval()` mode, or with `mode: none`, outputs equal eager attention exactly (tested). In `train()` mode, noise is added per query row. The attention-probability output is the noisy softmax.
- **Configuration:** `Quantization_and_Pruning/configs/jacobian_noise/*.yaml`. The `noise:` block maps 1:1 to `JacobianNoiseConfig`.
- **Reproduction command:** see `Quantization_and_Pruning/experiments/jacobian_noise/README.md`.
- **Tests:** `python -m pytest Quantization_and_Pruning/tests -q` (CPU, ~15 s).

```python
from Quantization_and_Pruning.quantization.qat.jacobian_noise import JacobianNoiseConfig, attach_jacobian_noise, detach_jacobian_noise

ctrl = attach_jacobian_noise(model, JacobianNoiseConfig(mode="jacobian"))   # or include="vision_model"
for batch in loader:                      # ordinary training loop
    ctrl.set_query_mask(batch["attention_mask"])   # optional: ignore padded rows in the statistics
    model(**batch).loss.backward(); optimizer.step(); optimizer.zero_grad()
    ctrl.step()                           # once per optimiser step
detach_jacobian_noise(model)              # restores the previous attention implementation before save_pretrained
```

### Noise modes
| `mode` | Meaning |
|---|---|
| `none` | same code path, no noise: the fine-tuning **baseline** |
| `jacobian` | paper method (per layer × query position) |
| `jacobian_layer` | one σ per layer (ablation of the paper's row-wise claim) |
| `uniform_matched` | one global σ whose RMS equals that of the `jacobian` σs at the same step. Same noise energy without Jacobian shaping, so it isolates the effect of the Jacobian |
| `constant` | fixed `constant_sigma` (sanity check) |

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | see table above | n/a | n/a | n/a | n/a | 8×A100, 3B LLMs, SigLIP-384 |
| Local baseline | pending | | | | not_measured | to be filled from `results.csv` |
| Local method | pending | | | | not_measured | |

CPU sanity check of the evaluation pipeline (pretrained weights, no fine-tuning, *tiny subsets*, status `cpu-check`, not a result):

| Model | Setting | FP16 | W8A8 | W6A6 | W4A4 (RTN) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | WikiText-2 PPL, 3 windows × 512 tokens | 17.27 | 17.52 | 20.23 | 3254.6 |
| SigLIP-B/16-224 | ImageNet zero-shot top-1, 64 val images | 70.3 | 71.9 | – | 6.3 |

## Differences and limitations

- **Scale:** our models are Qwen2.5-0.5B and SigLIP-B/16-224 (optionally -384) instead of 3B LLMs, and training is ~1000–1500 steps on one 16 GB GPU instead of 5–10 epochs on 8×A100. Results are labelled `approximate`.
- **PTQ methods:** RTN and GPTQ weights + per-token RTN activations (+ optional Q/K) are implemented. AWQ, SpinQuant, RepQ-ViT and ERQ are **not** implemented, so we cannot compare against the paper's exact PTQ rows.
- **Simulated quantization:** quant-dequant in floating point. There are no INT4 kernels, so measured latency and memory are *not* those of real low-bit inference.
- **Choices where the paper is silent:**
  - Statistics come from the clean masked softmax under `no_grad`, on the micro-batches of the step before each update (`collect: last_step`).
  - The mask is additive, so noise before or after it is equivalent. We add it before to avoid fp16 overflow.
  - Padded query rows are excluded from the statistics.
  - Unseen positions fall back to the layer mean.
  - All parameters are trained (LoRA optional for LMs).
  - LM loss is the response-only causal-LM loss.
  - The SigLIP objective is a zero-shot-head cross-entropy with a frozen text tower (the paper does not state it).
- **Hyperparameters changed for budget:** LR 1e-5 instead of 5e-7, LR warm-up 100 instead of 500 steps, sequence length 512 instead of 2048. They are identical across all noise modes.
- **ImageNet:** a public 256-px parquet copy is streamed. Absolute accuracies can differ slightly from full-resolution ImageNet; baseline and method share the pipeline.
- **Gradient checkpointing** replays the same noise, because torch restores the RNG state (tested). A custom `noise.seed` generator is therefore disallowed with checkpointing.

## Reproduction checklist

- [x] Exact paper link and version verified (arXiv 2608.20988v1)
- [x] Base model and dataset revisions pinned (`Backbones/registry.yaml`, `Dataloaders/registry.yaml`)
- [x] Baseline runs on the same setup (`mode: none`, same configs and seeds)
- [x] Tests and seed handling added
- [ ] Resource and quality metrics reported (pending GPU runs)
- [x] Licence/data terms checked (Alpaca CC-BY-NC-4.0; ImageNet research terms; models Apache-2.0)
