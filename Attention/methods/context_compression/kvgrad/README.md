# KVgrad score variants (our implementation)

Implementation of the importance scores of **KVgrad: Query-Agnostic KV Cache Eviction via Gradient-based
Global Importance Scoring** ([OpenReview](https://openreview.net/forum?id=cg1wTCJjjk), AdaptFM, ICML 2026)
as a [kvpress](https://github.com/NVIDIA/kvpress) press. Paper card and results:
[`Attention/experiments/kvgrad/PAPER_CARD.md`](../../../experiments/kvgrad/PAPER_CARD.md).

## What is ours and what is reused

- **Reused from kvpress 0.5.5 (Apache-2.0):** the context replay ("Repeat the previous context"), the hooks
  that capture attention weights and attention outputs, the backward pass of
  $\Phi = \sum_p \lVert x^{L+1}_p\rVert_2^2$, and the eviction step (`KVgradPress` / `KVzipPress`).
- **Ours:** `scores.py` computes the four score variants of the paper's ablation from those quantities, and
  `press.py` plugs them into the replay (`KVgradVariantsPress`), with an optional score cache so that one
  replay serves every variant and compression ratio.

## Score variants

With $\psi_{p,c} = A^{\ell}_{p,c}\, v^{\ell}_c W^{\ell(h)}_O$ (contribution of entry $c$ to the attention
output $a_p$ through query head $h$) and $\phi_p = \nabla_{a^{\ell}_p}\Phi$:

| `variant` | Score | Paper |
|---|---|---|
| `taylor_sum` | $\sum_p \langle \psi_{p,c}, \phi_p \rangle$ | Eq. (6), "w/o both" |
| `taylor_max` | $\max_p \langle \psi_{p,c}, \phi_p \rangle$ | "w/o MAS" |
| `bound_sum` | $\sum_p A^{\ell}_{p,c}\,\lVert v^{\ell}_c W^{\ell}_O\rVert_2\,\lVert \phi_p\rVert_2$ | Eq. (7) summed, "w/o PSP" |
| `bound_max` | $\max_p A^{\ell}_{p,c}\,\lVert v^{\ell}_c W^{\ell}_O\rVert_2\,\lVert \phi_p\rVert_2$ | **Eq. (8), KVgrad** |

MAS = magnitude-aware scoring, PSP = probe-specific preservation (max instead of sum over probes).
Scores are computed per query head and reduced to the KV head by a max over its GQA group, as in kvpress.
The lowest scores are evicted; signed Taylor scores are used as they are.

## Inputs and outputs

| | |
|---|---|
| Input | one context (batch size 1, as in kvpress), any length; replayed in chunks of `chunk_size` (default 512) tokens |
| Tensors (`score_variants`) | `attn_weights` (bsz, kv_heads, kv_groups, q_len, chunk_len); `values` (bsz, kv_heads, chunk_len, head_dim); `grad` (bsz, q_len, hidden); `o_proj_weight` (hidden, heads x head_dim) |
| Output | one score per (layer, KV head, context position); eviction indices stored as `masked_key_indices` on each attention module (kvpress convention) |
| Masks | causal; the first `n_sink` (4) tokens are never evicted |
| Cache behaviour | evicted entries are **masked**, not freed (kvpress fake-key mechanism): no real memory saving |
| Dtypes | tested in float32 (CPU and T4) and float64 (math tests); half-precision gradients are promoted to float32 |
| Models | decoder-only models supported by kvpress (tested: Qwen3); not Gemma3 (unsupported by KVzip in kvpress) |

## Usage

```python
import sys

sys.path.insert(0, "Attention")
from transformers import pipeline
import kvpress  # registers the "kv-press-text-generation" pipeline
from methods.context_compression.kvgrad import KVgradVariantsPress

pipe = pipeline(
    "kv-press-text-generation",
    model="Qwen/Qwen3-0.6B",
    device="cuda:0",
    model_kwargs={"attn_implementation": "sdpa", "dtype": "float32"},
)
press = KVgradVariantsPress(compression_ratio=0.875, variant="bound_max")
answer = pipe(context, question=question, press=press)["answer"]
```

## Tests

`python -m pytest Attention/tests/test_kvgrad_variants.py -q` (CPU, no downloads):

- Eq. (6) Taylor terms equal $v \cdot \nabla_v \Phi$ from autograd (float64, rtol 1e-10).
- The Eq. (7) bound holds for every probe and entry.
- `bound_max` reproduces kvpress's `KVgradPress` scores (rtol 1e-5, atol 1e-7) and evicts exactly the same
  KV pairs, with one and with several replay chunks.
- Cached scores give exactly the same evictions as a fresh press, for every variant and ratio.
