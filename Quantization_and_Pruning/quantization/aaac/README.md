# AAAC: Activation-Aware Adaptive Codebooks (4-bit weight quantization)

PyTorch reference implementation of AAAC (arXiv 2605.08692v2, Algorithm 1) for
`Qwen/Qwen2.5-0.5B`, with a matched INT4 round-to-nearest (RTN) baseline and a
WikiText-2 benchmark.

## Quickstart

Tested with Python 3.12 (Python 3.14 breaks `datasets==3.6.0` through `dill`).
Run everything from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch "transformers==4.57.6" "datasets==3.6.0" "numpy>=2" pytest

# 1. tests (CPU, no downloads)
python -m pytest -q

# 2. calibration: 4 seeded 2,048-token C4 windows -> per-layer activation energy
#    (a pre-computed file is already in results/, so this step is optional)
python -m Quantization_and_Pruning.quantization.aaac.run_calibration

# 3. benchmark: unmodified vs RTN vs AAAC (S = 128) on WikiText-2
python -m Quantization_and_Pruning.quantization.aaac.benchmark

# quick partial run (label results "partial")
python -m Quantization_and_Pruning.quantization.aaac.benchmark --max-windows 2
```

If the model download hangs, prefix the command with
`HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120`.

## Files

| File | Owner area | What it does |
|------|-----------|--------------|
| `Backbones/Generative_Pretrained/qwen2_5.py` | Member 1 | pinned Qwen2.5-0.5B loader |
| `Dataloaders/calibration/` | Member 1 | C4 sampler, activation-energy hooks, layer targets |
| `config.py`, `scaling.py`, `codebook_init.py`, `assignment.py`, `fit.py` | Member 2A | scaling, quantile init, weighted assignment, fit driver |
| `lloyd.py` | Member 2B | weighted Lloyd update, BF16 finalisation |
| `codec.py`, `linear.py` | Member 3 | 4-bit packing, selector encoding, `AAACLinear` |
| `rtn.py`, `benchmark.py`, `benchmark_config.json`, `Dataloaders/wikitext2.py` | Member 4 | RTN baseline, WikiText-2 perplexity and timing |
| `run_calibration.py` | Member 1 | writes `results/qwen25_05b_c4_activation_energy.pt` |
| `results/` | | calibration energies and measured benchmark JSON |

## 1. Where the inputs come from (Aditya Jain)

```python
from Backbones.Generative_Pretrained import load_qwen25_05b
from Dataloaders.calibration import iter_quantization_targets
import torch

model, tok = load_qwen25_05b(device="cpu", dtype=torch.float32)
energy = torch.load("Quantization_and_Pruning/quantization/aaac/results/qwen25_05b_c4_activation_energy.pt")   # {layer name: FP32 [in]}

for t in iter_quantization_targets(model, energy):   # 168 layers, one at a time
    t.weight      # FP32 copy of W, shape [out, in]
    t.importance  # FP32 I[k] = sum over tokens of X[t, k]^2, shape [in]
```

## 2. How to use the fit

```python
from Quantization_and_Pruning.quantization.aaac import AAACConfig, fit_aaac_layer, fit_aaac_model

cfg = AAACConfig(group_size=128, selection_size=128)   # main run; selection_size=16 is the follow-up
fit = fit_aaac_layer(t.weight, t.importance, cfg)      # one layer
fits = fit_aaac_model(model, energy, cfg)              # every eligible layer -> {name: AAACLayerFit} on CPU

W_hat = fit.dequantize()                               # dense FP32 reference reconstruction [out, in]
```

### What the fit returns (`AAACLayerFit`), the hand-off to Member 3 (codec)

| Field | Shape | dtype | Meaning |
|-------|-------|-------|---------|
| `scales` | `[out, in/g]` | FP16 (`cfg.scale_dtype`) | per-group absmax, **always > 0** |
| `codebooks` | `[2, 16]` | BF16 (`cfg.codebook_dtype`) | table 0 and table 1 (2 × 16 × 2 B = **64 bytes per layer**) |
| `selectors` | `[out, in/S]` | bool | `False` = table 0, `True` = table 1 |
| `indices` | `[out, in]` | uint8, values 0..15 | 4-bit index into the selected table |
| `group_size`, `selection_size` | int | | g and S |
| `objective_history` | list[float] | | weighted error after each assignment (see §4) |

Decode rule (this is exactly what `fit.dequantize()` computes; the packed codec must
give identical numbers):

```
W_hat[o, k] = float(scales[o, k // g]) * float(codebooks[selectors[o, k // S], indices[o, k]])
```

For S = g the codec may store the selector in the sign bit of the scale
(`scale = -scale` when the selector is True) and decode with `abs(scale)`; this
works because every scale is strictly positive. For S = 16 it stores a packed
selector bitset instead.

---

## 3. The algorithm step by step

All maths happens in **normalised space** `u = W / scale` in float32. `I[k]` is
the importance of input column `k`, shared by every output row.

### 3.1 Row-wise scaling (`scaling.py`)

* Each row is cut into contiguous groups of `g` weights (`in % g == 0` is required;
  Qwen2.5-0.5B widths 896 and 4864 are multiples of 128).
* `scale = max |W|` over the group, clamped to `[finfo.tiny, finfo.max]`, then rounded
  to FP16.
* **All-zero group:** scale = `torch.finfo(float16).tiny` (= 6.1e-5). The group's
  normalised values are 0 and decode to at most ~6e-5 × |table entry|, i.e. ~0.
* `u = W / rounded_scale`, so encoding and decoding use exactly the same scale.
  `|u|` can exceed 1 by at most 2^-11 because of the FP16 rounding.
* **RTN (Member 4) should call `compute_group_scales` too**, so RTN and AAAC use
  the same scales on the same layers.

### 3.2 Quantile initialisation (`codebook_init.py`)

* `weighted_quantiles(values, weights, probs)`: weighted inverse CDF. Sort values
  (stable), cumulative-sum the weights in float64, and take the first value whose
  cumulative weight is `>= p × total`. No interpolation, so the result is always a
  real data value. Zero-weight values are never picked. If all weights are 0,
  uniform weights are used.
* `quantile_codebook`: 16 quantiles at the bin mid-points `p = (j + 0.5) / 16`, so
  each codeword starts with 1/16 of the importance mass.

### 3.3 Initial codebook construction (`codebook_init.py::initial_codebooks`)

1. For every selection group (S contiguous weights of a row), compute the
   importance-weighted mean `|u|`. If the group's importance sums to 0, use the
   plain mean.
2. Sort the groups by that value (stable, so ties follow row-major order) and split
   them into two equal halves.
3. **Table 0** = `quantile_codebook` of the lower half ("peaky" groups, where one
   outlier sets the scale and most values sit near 0). **Table 1** = the same for
   the upper half ("flat" groups).
4. Edge cases: with only one selection group, both tables are identical. If a half
   has zero total importance, its table is built from the whole layer.

### 3.4 Weighted group assignment (`assignment.py`)

For each selection group and each table `c`:

```
E_c = sum over k in group of  I[k] * min_j (u[o, k] - table_c[j])^2
```

The group takes the table with the smaller `E_c`.

* **Tie-breaking:** the nearest codeword is the *lowest index* among equally near
  ones. Between tables, table 1 is chosen only if `E_1 < E_0` *strictly*, so an
  exact tie goes to table 0.
* **Chunking:** the nearest-neighbour search makes a `[rows, in, 16]` tensor. It is
  processed in blocks of whole rows so this tensor holds at most
  `cfg.chunk_elements` (default 2^24 ≈ 64 MB in FP32) elements. Results are
  bit-identical for any chunk size (tested).
* `encode_indices` returns, for every weight, the index of its nearest entry in its
  group's selected table (uint8) and the squared distance.

### 3.5 The deterministic fit (`fit.py::fit_aaac_layer`)

```
scales, U   = scaling                                   (3.1)
C           = initial_codebooks(U, I, S)   float32       (3.2, 3.3)
repeat outer_iters (3):
    selectors = assign_selection_groups(U, I, C, S)      (3.4)
    repeat inner_iters (10):
        C = lloyd.weighted_lloyd_update(...)
C = lloyd.finalize_codebooks(C, bfloat16)                sort + round
selectors = assign_selection_groups(U, I, C_bf16, S)     re-run with rounded tables
indices   = encode_indices(U, C_bf16, selectors, S)      re-run encoding
```

* **Deterministic:** no random numbers, stable sorts, fixed tie rules, and no
  atomic scatter operations. Running twice gives bit-identical tensors (tested).
* If `importance` is all zero, the fit uses uniform importance (plain k-means).
* The input weight and importance are never modified. Work runs on
  `weight.device`; `fit_aaac_model` moves results to CPU.
* `update_fn` and `finalize_fn` must return `[2, 16]` tensors of dtype float32 and
  `cfg.codebook_dtype` respectively, with finite values. Otherwise the fit raises
  `ValueError`.

### 3.6 Implementation choices

| Choice | Value |
|--------|-------|
| Tensor layout | `W[out, in]`, groups run along `in` (contiguous within a row) |
| Scale rule | group absmax, FP16, strictly positive, all-zero group → `finfo(fp16).tiny` |
| Error weights | `I[k]` in normalised space (not multiplied by `scale²`) |
| Codeword tie | lowest index |
| Table tie | table 0 |
| Table dtype | float32 while fitting, BF16 when stored |
| Constraint | `g % S == 0` and `in % g == 0` (no padding) |
| Eligible layers | Aditya's set: 168 decoder `nn.Linear` layers; `embed_tokens`/`lm_head` stay FP |

Why "normalised space, weights `I[k]`": inside one selection group the scale is
constant (`S` divides `g`), so multiplying by `scale²` would not change which table
a group picks. Using `I[k]` alone for both assignment and the Lloyd update keeps
both steps minimising the **same** objective, so the objective never goes up.

---

## 4. Weighted Lloyd update and BF16 finalisation (`lloyd.py`)

### `weighted_lloyd_update`

1. Assign every weight to a cell (table c, entry j) with `encode_indices`, using
   the same lowest-index tie rule as the rest of the fit.
2. `new[c, j] = sum(I[k] * u) / sum(I[k])` over the weights in that cell,
   accumulated in float64.
3. Empty cells (no members, or zero total importance) keep their old value.
4. Returns a new float32 `[2, 16]` tensor; the input is not modified.

Per-cell sums use a stable sort plus cumulative sum instead of `bincount` /
`index_add_`, so the update is deterministic on CUDA as well as CPU.

### `finalize_codebooks`

Sorts each table ascending and rounds it to BF16. The fit then re-runs group
assignment and index encoding with the rounded tables, so stored indices always
match stored tables.

### Sanity numbers

Member 2A ran this fit on **real Qwen2.5-0.5B layers** with the committed C4 energies and
a throwaway Lloyd update (before `lloyd.py` was finished). The error is
`Σ I[k] (W_hat − W)²`. RTN is symmetric INT4 (levels −7..7) with the same scales. `rtn.py` stores codes −8..7, but since |u| ≤ 1 code −8 is never used.

| Layer | Shape | S | AAAC error | RTN error | RTN / AAAC |
|-------|-------|---|-----------|-----------|-----------|
| `layers.0.self_attn.k_proj` | 128 × 896 | 128 | 824 | 1106 | 1.3× |
| `layers.0.self_attn.k_proj` | 128 × 896 | 16 | 728 | 1106 | 1.5× |
| `layers.12.mlp.down_proj` | 896 × 4864 | 128 | 1641 | 2943 | 1.8× |
| `layers.12.mlp.down_proj` | 896 × 4864 | 16 | 1423 | 2943 | 2.1× |
| `layers.23.self_attn.o_proj` | 896 × 896 | 128 | 16210 | 33100 | 2.0× |
| `layers.23.self_attn.o_proj` | 896 × 896 | 16 | 13290 | 33100 | 2.5× |

* `objective_history` should go down every outer round, e.g. `layers.12.mlp.down_proj`
  with S = 128 gave `[1.08e6, 5.76e5, 5.12e5, 4.86e5]`.
* S = 16 should beat S = 128 on every layer.
* Time on CPU: about 16 s for a `down_proj` layer and 2–3 s for an 896 × 896 layer.
  Nearly all of it is the 30 calls to `encode_indices` inside the Lloyd loop. If you
  need more speed, you can use `torch.searchsorted` on the sorted tables in place of
  the 16-way `argmin`, but keep the lowest-index tie rule.

## 5. Tests

From the repository root:

```bash
python -m pytest Quantization_and_Pruning/quantization/aaac/tests -q
```

| File | Covers |
|------|--------|
| `test_aaac_scaling.py` | absmax per contiguous group, all-zero groups get a positive scale, FP16 limits, exact reconstruction, validation |
| `test_aaac_codebook_init.py` | weighted quantiles by hand, zero weights ignored, uniform fallback, 16 sorted entries, peaky/flat split, determinism |
| `test_aaac_assignment.py` | **weighted** assignment flips with importance, tie → table 0 / lowest index, matches brute force, encoding uses the selected table, chunk-size invariance |
| `test_aaac_fit.py` | output shapes/dtypes for S = 128 and S = 16, bit-identical reruns, final re-assignment with BF16 tables, update called 3 × 10 times, `dequantize`, whole-model driver, validation, end-to-end objective never increases |
| `test_aaac_lloyd.py` | weighted centroid vs plain mean, brute-force match, empty and zero-importance cells keep their value, determinism, chunk invariance, BF16 finalisation, stored indices match BF16 tables |
| `test_aaac_codec_and_integration.py` | exact 4-bit round trip, selector round trip for S = g and S < g, all-zero group convention, exact packed byte counts, bit-exact parity with the fit and model integration |
| `test_aaac_rtn.py` | RTN packing bytes and reconstruction, same target layers as AAAC |
| `test_aaac_smoke.py` | end-to-end on a tiny random Qwen2 (no downloads): calibration, fit, packing, model swap, exact byte counts, AAAC < RTN weighted error, perplexity for all three variants |

`tests/aaac_helpers.py` holds the stand-ins `keep_tables` (no-op update) and
`round_tables` (plain BF16 cast), used to test the pipeline independently of `lloyd.py`.
