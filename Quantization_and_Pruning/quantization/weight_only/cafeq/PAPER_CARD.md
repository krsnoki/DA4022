# Paper implementation card: CafeQ: Calibration-free Quantization via Learned Transformations and Adaptive Rounding

- **Paper / OpenReview:** https://openreview.net/forum?id=CBtx52vftl (AdaptFM). Implemented from the anonymous ICML 2026 submission PDF ("Preliminary work. Under review", 17 pages, dated 2026-05-09). Re-check the forum for a newer version before reporting results.
- **Primary module:** `Quantization_and_Pruning/` (`quantization/weight_only/cafeq/`)
- **Secondary modules:** `Attention/` (the V/O pairing relies on attention being linear in the values), `Performance_Metrics/`
- **Owner / reviewer:** @AbhinavSandilya @kireetibabu07 @kawaiibaka0908 @Vedant_Raje @Avinash-Reddy-IITG
- **Status:** tested
- **Upstream code / licence:** none found (anonymous submission). The paper's GPTQ / FlatQuant comparisons used the public FlatQuant codebase, which this implementation does not reuse.
- **Backbone IDs:** `text.qwen2_5_0_5b.base` (CPU-feasible primary), `text.gemma_2_2b.base` (paper setting). Cards: `Backbones/Generative_Pretrained/`.
- **Dataset IDs and split:** `text.wikitext2.raw`, split `test`, role `evaluation`. **Calibration data: none** (`calibration_dataset: null`).
- **Hardware assumptions:** any PyTorch device. Unit tests and the tiny smoke run need CPU only; Qwen2.5-0.5B in float32 should fit in about 8 GB of system RAM (not yet measured). The paper-budget Gemma 2 2B config expects a GPU.

## Research summary

### Problem
Post-training weight quantization reduces the memory traffic that dominates LLM decoding. Plain round-to-nearest (RTN) uniform quantization is hurt by outlier weights, which stretch each group's range. Most fixes (GPTQ, learned rotations such as SpinQuant or FlatQuant) need calibration activations, which may be unavailable or private, cost forward/backward passes, and can overfit to one domain. CafeQ asks how far you can get using **only the weights**.

### Method
1. **Weight-only proxy loss.** Using Cauchy–Schwarz, the expected output error is bounded by the Frobenius reconstruction error of the weights, `||M⁻¹ Q(MW) − W||_F`. The paper shows that this proxy correlates with downstream accuracy (Fig. 1, Spearman ≈ −0.64 / −0.68). Rounding is not differentiable, so the expected error under *stochastic* rounding is bounded instead (Eq. 4); that bound is smooth enough to optimise.
2. **Single matrices (Sec. 3.1).** For q, k, gate, up and down projections, learn an invertible **block-diagonal** `M` along the contraction dimension. Each block starts as a random rotation and is trained by gradient descent on Eq. 4. At inference `x ↦ x M⁻¹` is applied online, which costs `O(d·k)` per token (about 1.4% extra FLOPs at k = 128 in Table 1).
3. **Coupled matrices (App. A).** For `(W_v, W_o)`, `W_v W_o = (W_v M)(M⁻¹ W_o)` for any invertible `M`, so the transform is absorbed into the weights at zero inference cost. `M` is unconstrained and learned with Adam on a LogSumExp of channel-wise maxima of both transformed matrices (t = 5).
4. **Adaptive rounding for coupled matrices (App. B, Alg. 1).** The method alternates `Ŵ₂ ← Q(Ŵ₁⁺ W₁ W₂)` and `Ŵ₁ ← Q(W₁ W₂ Ŵ₂⁺)`, so each matrix's rounding compensates for the other's error. Both results stay on the ordinary uniform grid.

### Expected behaviour (paper-reported, not measured here)
- Table 1 (Gemma 2 2B down_proj, 4-bit): relative ℓ2 loss 0.176 (uniform) → 0.155 (random Hadamard) → 0.094 (learned, k = 128).
- Table 6 (V·O product, 4-bit): 0.182 (uniform) → 0.149 (learned transform) → 0.117 (+ adaptive rounding).
- Table 2 (Gemma 2 2B, 4-bit per-channel, average score): unquantized 48.0, uniform 41.3, random 43.6, CafeQ 45.6. At 3-bit per-channel: 23.3 / 26.1 / 35.1.
- Table 5 (Qwen2.5-7B, 4-bit per-channel): WikiText-2 ppl FP16 8.36, GPTQ 9.27, FlatQuant 8.46, CafeQ 8.56. Some cells in this table look inconsistent (FlatQuant ARC-C 51.71 against FP16 79.82; PIQA above FP16), so do not use its per-task numbers as reproduction targets.

## Implementation

| Paper component | Code |
|---|---|
| Uniform quantizer, Eq. (1) | `quantization/common/uniform.py` (`QuantSpec`, `quantize_dequantize`) |
| Block-diagonal / Hadamard transforms | `quantization/common/transforms.py` |
| Eq. (4) surrogate + learning `M` (Sec. 3.1) | `cafeq/block_diagonal.py` |
| Paired V/O transform, LogSumExp / sum-sq pseudo-losses, orthonormal penalty (App. A, F) | `cafeq/paired.py` |
| Algorithm 1, generic + GQA-aware V/O version | `cafeq/joint_rounding.py` |
| Model surgery, baselines (`rtn`, `random_hadamard`), report | `cafeq/apply.py` |
| Inference module, save/load | `cafeq/modules.py` |
| Config schema and validation | `cafeq/config.py`, configs in `configs/cafeq/` |
| WikiText-2 loader | `Dataloaders/evaluation/wikitext2.py` |

- **Input contract:** a Hugging Face decoder-only causal LM whose blocks sit at `model.model.layers` and expose `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` (Llama, Mistral, Qwen2/2.5, Gemma, Gemma 2). Any dtype is accepted; optimisation runs in float32 or float64 internally.
- **Output contract:** the model is modified in place. Target linears become `CafeQLinear` (same `in_features`/`out_features`, dequantized weights in the model dtype, optional `inv_blocks` buffer of shape `(in/k, k, k)`). `quantize_model` returns a JSON-serialisable report with per-matrix relative ℓ2 errors (RTN / init / final), V/O paired errors, the Algorithm 1 trace, timing, and an *analytic* extra-FLOPs estimate. Embeddings, `lm_head`, norms and activations are untouched, as in the paper.
- **Configuration:** `Quantization_and_Pruning/configs/cafeq/qwen2_5_0_5b_w4_perchannel.yaml` (scaled-down), `gemma2_2b_w4_perchannel_paper.yaml` (paper budgets), `tiny_smoke.yaml` (tests). Key flags: `quantization.{method,bits,group_size,targets}`, `block_transform.{block_size,iterations,loss}`, `paired_transform.{pseudo_loss,temperature,orth_weight}`, `adaptive_rounding.iterations`.
- **Reproduction command:**
  ```bash
  python Quantization_and_Pruning/experiments/cafeq/run_cafeq.py \
      --config Quantization_and_Pruning/configs/cafeq/qwen2_5_0_5b_w4_perchannel.yaml --baseline
  # baselines on the identical setup
  ... --method rtn
  ... --method random_hadamard
  # CPU smoke run, no downloads
  python Quantization_and_Pruning/experiments/cafeq/run_cafeq.py \
      --config Quantization_and_Pruning/configs/cafeq/tiny_smoke.yaml --tiny --baseline
  ```
- **Tests:** `python -m pytest Quantization_and_Pruning/tests -q` (CPU; the model tests use tiny randomly initialised Llama / Qwen2 / Gemma 2 configs and download nothing).

What the tests check:
- the quantizer stays within the Eq. (2) error bound, uses at most 2^N levels, keeps dtypes, and its stochastic rounding is unbiased;
- the grouped Eq. (4)-style bound upper-bounds the Monte-Carlo stochastic-rounding error;
- the learned block transform beats RTN on outlier-heavy weights;
- the paired transform preserves `W₁W₂` exactly;
- the paper's 2×2 toy example (PQE 0.64 → 0.36) reproduces;
- the fast GQA PQE matches a per-head brute-force computation;
- the V/O transform leaves model outputs unchanged under GQA, with and without v-bias;
- all methods run end to end on Llama, Qwen2 and Gemma 2 layouts, the target selection used for the Table 3 ablation works, save/load round-trips, and the run script works.

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | Gemma 2 2B avg 45.6 (FP 48.0); Qwen2.5-7B WikiText-2 ppl 8.56 (FP16 8.36) | +3.3% wall-clock, down_proj, k = 128 (Table 1) | — | — | — | 4-bit per-channel; GPU kernel details not given |
| Local baseline | not measured | not measured | not measured | not measured | not_measured | run with `--baseline`; rows reserved in `Performance_Metrics/results/cafeq.csv` |
| Local method | not measured | not measured | not measured | not measured | not_measured | fake quantization: latency will *not* reflect low-bit speed-ups |

## Differences and limitations

1. **Fake quantization.** Weights are stored dequantized in the model dtype and computed with ordinary dense matmuls. No INT3/INT4 kernels exist here, so measured latency and memory cannot show quantization speed-ups. `quantize_to_codes` gives the integer codes for storage-size accounting.
2. **Eq. (4) layout ambiguity.** The printed bound uses `||(MW)_j||_∞` (row j of `MW`), which matches quantization groups along the *output* dimension, while Sec. 4 says groups run along the *contraction* dimension. `loss: paper_eq4` implements the printed formula literally (the default). `loss: grouped_bound` implements the intermediate bound with the actual groups, and a unit test checks that it really upper-bounds the stochastic error. In a NumPy check both lowered the true RTN error well below random rotation.
3. **Unspecified optimiser for single-matrix transforms.** The paper gives 20k iterations and learning rates {0.1, 0.5, 1.0} (App. E) but not the optimiser. We default to Adam (lr 1e-3) on a loss normalised by `||W||_F²`; SGD is available.
4. **"Number of Hadamard matrices" (App. E sweep) is not explained in the paper and not implemented.**
5. **GQA is not discussed in the paper.** Our choice: one `M_g` (head_dim × head_dim) per KV head, applied to `W_v`'s rows for that head and to the `W_o` columns of *every* query head attached to it. The LogSumExp loss sees `W_v,g` and the concatenation of those `W_o` blocks. Algorithm 1 builds its targets per head, then quantizes the full `v_proj` / `o_proj` with their real grouping (`o_proj` groups span several heads), so the rounding map matches inference. Exactness is unit-tested.
6. **LogSumExp scaling.** With raw weights (|w| around 1e-2) and t = 5 the LogSumExp is nearly flat, so by default channel maxima are divided by the initial max |W| of each matrix (`normalize: true`). Set it to false for the literal formula.
7. **Safety additions.** The single-matrix transform keeps the best surrogate value and falls back to its initialisation if the true error got worse (`fallback_if_worse`). Algorithm 1 returns the best pair seen, including the independently rounded pair (`keep_best`), because it is not monotone in practice.
8. **Q/K pairing is not implemented** (RoPE breaks commutativity); the paper also leaves it to future work. Q and K get single-matrix block transforms.
9. **Online transforms are not fused.** q/k/v share an input, as do gate/up, but each matrix applies its own `M⁻¹`, so real overhead is higher than the paper's < 3% unless a fused kernel is written.
10. **Precision.** `inv_blocks` and the dequantized weights are cast to the model dtype; in bfloat16 this adds a small extra error compared with the paper's (unknown) setup.
11. **Budgets.** The CPU config uses 2k / 5k iterations instead of 20k / 100k. Label any such result `approximate`.
12. **Evaluation scope.** Only WikiText-2 perplexity is wired in. The paper's zero-shot / few-shot suites (Table 7) need `lm-evaluation-harness`, which is not integrated yet.
13. **Reverse transform note in Table 2** does not apply: after adaptive rounding the stored V/O weights are already the quantized transformed pair.

## Reproduction checklist

- [ ] Exact paper link and version verified
- [ ] Base model and dataset revisions pinned (`backbone.revision`, dataset revision)
- [ ] Baseline runs on the same setup (`--baseline`, `--method rtn`, `--method random_hadamard`)
- [x] Tests and seed handling added (`seed` in config drives all random transforms)
- [ ] Resource and quality metrics reported
- [ ] Licence/data terms checked (Gemma 2 is gated; Qwen2.5 and WikiText terms on their cards)
