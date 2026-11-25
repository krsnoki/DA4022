# AAAC learning algorithm (Member 2)

PyTorch reference implementation of the **fit** step of AAAC (*Activation-Aware
Adaptive Codebooks for 4-bit LLM Weight Quantisation*, Algorithm 1). It turns one
weight matrix plus its calibration importance into the pieces needed to store a
4-bit layer.

Member 2's work is split between two people:

| Part | Owner | Status | Files |
|------|-------|--------|-------|
| **2A** Row-wise scaling, quantile initialisation, weighted group assignment, initial codebook construction, deterministic fit function | Rahul Yadav | **done**, tested | `config.py`, `scaling.py`, `codebook_init.py`, `assignment.py`, `fit.py` |
| **2B** Weighted Lloyd updates, keep empty centroids, BF16 finalisation (+ tests for empty cells and BF16 finalisation) | Vanshika | **TODO** | `lloyd.py` (two stubs to fill in) |

`fit_aaac_layer` already calls the two 2B functions with their final signatures, so
once `lloyd.py` is filled in, the full fit runs with no other changes.

---

## 1. Where the inputs come from (Aditya Jain)

```python
from Backbones.Generative_Pretrained import load_qwen25_05b
from Dataloaders.calibration import iter_quantization_targets
import torch

model, tok = load_qwen25_05b(device="cpu", dtype=torch.float32)
energy = torch.load("qwen25_05b_c4_activation_energy.pt")   # {layer name: FP32 [in]}

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

> Until Member 2B fills in `lloyd.py`, calling the fit with default settings raises
> `NotImplementedError`. Tests run the pipeline by passing stand-ins
> (`update_fn=`, `finalize_fn=`, see `tests/aaac_helpers.py`).

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
        C = lloyd.weighted_lloyd_update(...)             <- Member 2B
C = lloyd.finalize_codebooks(C, bfloat16)                <- Member 2B
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

## 4. Hand-off: what is left

Fill in the two functions in **`lloyd.py`**. Their docstrings hold the exact
contract. In short:

### `weighted_lloyd_update(normalized, importance, codebooks, selectors, S, *, chunk_elements) -> [2,16] float32`

1. Assign every weight to a cell: `idx, _ = encode_indices(normalized, codebooks, selectors, S, chunk_elements=chunk_elements)`.
   The weight's table is `selectors.repeat_interleave(S, dim=1)`.
2. For each (table c, entry j): `new[c, j] = Σ I[k]·u / Σ I[k]` over the weights in that cell.
3. **Empty cells** (no members, or total importance 0) **keep their old value**.
4. Return a *new* tensor (do not modify `codebooks` in place). Accumulate in float64.

A simple deterministic way to do step 2 on CPU: flatten `key = table * 16 + idx`
and use `torch.bincount(key, weights=..., minlength=32)` twice (numerator and
denominator). Note that `bincount`/`index_add_` with weights is **not**
deterministic on CUDA. On GPU, use a one-hot matmul or sort-based reduction, or run
the update on CPU.

### `finalize_codebooks(codebooks, codebook_dtype=torch.bfloat16) -> [2,16] codebook_dtype`

Round the tables to BF16 and return a new tensor. The fit itself already re-runs
assignment and encoding with the rounded tables.

### Tests you should add (e.g. `tests/test_aaac_lloyd.py`)

* **Weighted centroid:** a hand-made example where the weighted mean differs from
  the plain mean.
* **Empty cells:** a table entry that no weight is nearest to, and a cell whose
  members all have `I = 0`, both keep their old value.
* **BF16 finalisation:** the output dtype is `bfloat16`, values equal
  `tables.to(bfloat16)`, and `fit.indices` / `fit.selectors` match a fresh
  `encode_indices` / `assign_selection_groups` on the BF16 tables.
* **Determinism:** two calls give identical tensors.
* `tests/test_aaac_fit.py::test_end_to_end_with_real_lloyd_update` is currently
  **skipped**. It turns on automatically once `lloyd.py` stops raising
  `NotImplementedError`, and checks that the objective does not increase across
  outer iterations.

### Sanity numbers to expect

I ran this fit on **real Qwen2.5-0.5B layers** with the committed C4 energies and
a simple throwaway weighted Lloyd update (not committed). The error is
`Σ I[k] (W_hat − W)²`. RTN is symmetric INT4 (levels −7..7) using the same scales.

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

If your Lloyd update gives numbers in this range, it is very likely correct.

---

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
| `test_aaac_fit.py` | output shapes/dtypes for S = 128 and S = 16, bit-identical reruns, final re-assignment with BF16 tables, update called 3 × 10 times, `dequantize`, whole-model driver, validation, end-to-end (skipped until 2B) |

`tests/aaac_helpers.py` holds the stand-ins `keep_tables` (no-op update) and
`round_tables` (plain BF16 cast), used so the pipeline can be tested before 2B lands.
