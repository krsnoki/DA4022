# Paper implementation card: Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models

- **Paper / OpenReview:** https://openreview.net/forum?id=wxKn35RAhW · arXiv: https://arxiv.org/abs/2608.20988 (v1, Pandey, Chavan, Lele, Dayal, Gupta; ICML 2026 AdaptFM workshop)
- **Primary module:** `Quantization_and_Pruning/` (quantization-aware fine-tuning, `quantization/qat/`)
- **Secondary modules:** `Attention/` (the method modifies pre-softmax attention logits); `Performance_Metrics/` (result rows)
- **Owner / reviewer:** @TakshayBansal
- **Status:** `partial-reproduction` (results labelled `partial` / `approximate`). Implementation and CPU tests are complete; SigLIP-B/16-224 GPU results are reported below (2 seeds, Kaggle T4). The Qwen2.5 pipeline is implemented and CPU-tested but has no GPU results yet.
- **Upstream code / licence:** **not found**. No code link on arXiv, OpenReview or the authors' pages (checked 2026-11-13). This is an independent re-implementation from the paper text.
- **Backbone IDs:** `multimodal.siglip_base_p16_224`, `multimodal.siglip_base_p16_384`, `text.qwen2_5_0_5b.base` (optional `text.qwen2_5_1_5b.base`); tests use `test.tiny_qwen2.random`, `test.tiny_siglip.random`
- **Dataset IDs and split:** `text.alpaca.train` (fine-tuning), `text.wikitext2_raw.test` (PPL), `text.wikitext2_raw.train` (GPTQ calibration), `vision.imagenet1k_256.train` (SigLIP fine-tuning + calibration, disjoint), `vision.imagenet1k_256.validation` (evaluation). See `Dataloaders/registry.yaml`.
- **Hardware assumptions:** training needs one CUDA GPU with 16 GB (reported results: Kaggle Tesla T4). Tests and smoke runs need only CPU.

## Contributors

Developed as a DA4022 course project by:

| Name | Contribution |
|---|---|
| Takshay Bansal | Led the implementation of the noise controller and the Hugging Face attention backend. Ran the Kaggle GPU experiments, and set up the GitHub fork and pull request. |
| Ayanna Mondal | Analysed the paper and derived the closed-form softmax-Jacobian norm and noise schedule used in the implementation. Wrote the paper card's research summary. |
| Shirshendu Sengupta | Built the simulated quantization tooling (fake-quant W/A/QK and GPTQ) and the evaluation script for the bit-width sweep. |
| Swastik Pramanik | Wrote the training scripts for SigLIP and Qwen2.5 and the dataset/backbone registry entries, and tuned the learning rate in the pilot runs. |
| Youkta Mandavkar | Wrote the CPU test suite, analysed the results (including the uniform-noise ablation), and prepared the documentation and result tables. |

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
- **Baseline:** the same fine-tuning run with `noise.mode: none` (same backbone, data, seed, resolution, batch size and evaluation settings; only `noise.*` differs), plus the unmodified pretrained checkpoint (`EVAL_PRETRAINED=1`).
- **Reproduction command** (one seed, all modes, train + eval + summary; the reported table runs it for seeds 0 and 1 and adds `noise.alpha=4`, see [experiments README §4](../../../experiments/jacobian_noise/README.md#4-reproducing-the-reported-siglip-results)):
  `MODEL=siglip224 SEEDS=0 MODES="none jacobian uniform_matched" bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh`
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

### Local results: SigLIP-B/16-224, ImageNet zero-shot top-1 (%)

Status `partial` / `approximate`. Mean ± std over 2 seeds; 5,000 seeded validation images, the same for every model. Weights: per-channel symmetric RTN; activations: per-token asymmetric RTN; quantization is simulated. Fine-tuning: 500 steps × 32 images, LR 5e-7, identical for every row except `noise.*`. Commit `0a24904`, Kaggle Tesla T4, torch 2.11 / transformers 4.57.6, bf16. Per-run resolved configs, environment, σ history and training logs: [`reported/siglip224/runs/`](../../../experiments/jacobian_noise/reported/siglip224/runs/). Raw rows: [`reported/siglip224/results.csv`](../../../experiments/jacobian_noise/reported/siglip224/results.csv); tables: [`summary.md`](../../../experiments/jacobian_noise/reported/siglip224/summary.md).

| Training | FP16 | W6A6 | W5A5 | W4A4 |
|---|---:|---:|---:|---:|
| Pretrained, no fine-tuning | 73.40 | 72.34 | 67.78 | 5.98 |
| Fine-tune, no noise (**baseline**) | 72.28 ± 0.00 | 71.26 ± 0.00 | 65.40 ± 0.00 | 6.90 ± 0.00 |
| + uniform noise, energy matched to α=0.5 | 72.21 ± 0.01 | 71.38 ± 0.06 | 65.95 ± 0.04 (+0.55) | 7.06 ± 0.28 |
| + Jacobian noise, α=0.5 (paper default) | 72.19 ± 0.07 | 71.37 ± 0.35 | 66.24 ± 0.25 (+0.84) | 6.73 ± 0.04 |
| + Jacobian noise, α=4 | 72.52 ± 0.03 | 71.18 ± 0.14 | 66.50 ± 0.59 (+1.10) | 7.33 ± 0.61 |

"FP16" means unquantized; on the T4 the model ran in bf16 (see the `dtype` column). Values in parentheses are Δ vs the baseline.

**Reading of the results:**
- **W5A5** is the only informative low-bit column. There, the ranking is baseline < uniform noise < Jacobian α=0.5 < Jacobian α=4, which is the direction the paper predicts. The gains are small (+0.55 to +1.10 points).
- **Jacobian vs uniform noise at the same energy** (α=0.5) differs by +0.29 points, about the seed spread. At this scale we cannot attribute the gain to the Jacobian shaping. The α=4 run has no energy-matched uniform control, so its +1.10 may come from noise strength alone.
- **Noise does not hurt unquantized accuracy** (FP16 within ±0.3 points).
- **The pretrained model beats every fine-tuned model** at W5A5. Here, noise reduces the accuracy lost to fine-tuning rather than improving on the original checkpoint.
- **W4A4 with RTN collapses** to ~7% for every model, so it cannot test the paper's headline W4A4 claim (which used RepQ-ViT/ERQ).
- **Attention sharpness is unchanged:** the mean ‖J‖² over layers is 0.0615–0.0617 for every model ([`jacobian_diagnostics.jsonl`](../../../experiments/jacobian_noise/reported/siglip224/jacobian_diagnostics.jsonl)). σ is 0.09–0.15 across layers at α=0.5 (≈1.0 at α=4), so Jacobian-shaped noise is close to uniform.
- **Seed caveat:** the seed changes only the noise RNG. Data order (`data.seed=0`) and the pretrained start are shared, so the two no-noise runs are identical (std 0.00) and the ± values understate run-to-run variance.
- **Learning rate:** in a pilot, LR 1e-5 cut FP16 accuracy from 72.9 to 56.5 after 200 steps; LR 5e-7 (the paper's value) kept it at 72.1. The configs use 5e-7.

### Resources

| Source | Quality | Latency (median, batch 32) | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | see "Expected behaviour" | n/a | n/a | n/a | n/a | 8×A100, 3B LLMs, SigLIP-384 |
| Local, unquantized | table above | 467–529 ms | 60–68 img/s (derived) | 825 MB | not_measured | T4, bf16, image tower only |
| Local, simulated W/A quant | table above | 612–685 ms | 47–52 img/s (derived) | 1236 MB | not_measured | quant-dequant overhead; not real low-bit speed |

Noise is training-only, so a noise-trained checkpoint has exactly the baseline's inference cost. Throughput is derived from the measured latency (32 images ÷ median latency), not measured separately. Energy was not measured (no power sensor access on Kaggle). Latency is measured on a separate batch of 32 random images (`latency.batch_size`); peak memory is over the evaluation pass (batch 64, the `batch_size` column of `results.csv`). Training cost: ~14 min per 500-step run plus ~11 min of evaluation per checkpoint on one T4.

CPU sanity check of the evaluation pipeline (pretrained weights, no fine-tuning, *tiny subsets*, status `cpu-check`, not a result):

| Model | Setting | FP16 | W8A8 | W6A6 | W4A4 (RTN) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | WikiText-2 PPL, 3 windows × 512 tokens | 17.27 | 17.52 | 20.23 | 3254.6 |
| SigLIP-B/16-224 | ImageNet zero-shot top-1, 64 val images | 70.3 | 71.9 | – | 6.3 |

## Differences and limitations

- **Scale:** reported results use SigLIP-B/16-224 (the paper used -384) with 500 steps on one 16 GB GPU, instead of 5–10 epochs on 8×A100. The Qwen2.5-0.5B path has no GPU results. Results are labelled `approximate`.
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
- **Hyperparameters changed for budget:** SigLIP uses the paper's LR 5e-7 but LR warm-up 100 instead of 500 steps. The (untested on GPU) Qwen config uses LR 1e-5 and sequence length 512 instead of 2048. All settings are identical across noise modes.
- **ImageNet:** a public 256-px parquet copy is streamed. Absolute accuracies can differ slightly from full-resolution ImageNet; baseline and method share the pipeline.
- **Gradient checkpointing** replays the same noise, because torch restores the RNG state (tested). A custom `noise.seed` generator is therefore disallowed with checkpointing.

## Assets, licences and downloads

Nothing is committed except code, configs and small result files (about 200 KB in `experiments/jacobian_noise/reported/`). All assets are downloaded automatically from the Hugging Face Hub on first use, at the commit revisions pinned in `Backbones/registry.yaml` (`revision`) and `Dataloaders/registry.yaml` (`version`). A pinned commit hash identifies the exact file contents, so it serves as the checksum.

| Asset | Hub ID @ revision | Licence / terms |
|---|---|---|
| SigLIP-B/16-224 | `google/siglip-base-patch16-224` @ `7fd15f06…` | Apache-2.0 |
| SigLIP-B/16-384 | `google/siglip-base-patch16-384` @ `41aec1c8…` | Apache-2.0 |
| Qwen2.5-0.5B / 1.5B | `Qwen/Qwen2.5-0.5B` @ `060db649…`, `Qwen/Qwen2.5-1.5B` @ `8faed761…` | Apache-2.0 |
| ImageNet-1k (256 px) | `benjamin-paine/imagenet-1k-256x256` @ `1bd04004…` | ImageNet terms: non-commercial research only |
| Alpaca | `tatsu-lab/alpaca` @ `dce01c9b…` | CC-BY-NC-4.0 |
| WikiText-2 | `Salesforce/wikitext` (wikitext-2-raw-v1) @ `b08601e0…` | CC-BY-SA-3.0 / GFDL |

Set `HF_HOME` to choose the cache location. ImageNet is streamed, not stored.

## Reproduction checklist

- [x] Exact paper link and version verified (arXiv 2608.20988v1)
- [x] Base model and dataset revisions pinned (`Backbones/registry.yaml`, `Dataloaders/registry.yaml`)
- [x] Baseline runs on the same setup (`mode: none`, same configs and seeds)
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported (SigLIP-B/16-224, 2 seeds; LLM pending)
- [x] Licence/data terms checked (Alpaca CC-BY-NC-4.0; ImageNet research terms; models Apache-2.0)
