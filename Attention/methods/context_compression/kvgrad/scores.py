"""
KVgrad score variants (paper: https://openreview.net/forum?id=cg1wTCJjjk, Sec. 3.2 and Table 1).

Notation, for one layer l and one chunk of replayed context:

    A[p, c]   attention from replay probe p to cached entry c (per query head)
    v_c       cached value vector of entry c (per KV head)
    W_O       output projection; W_O^(h) is the slice of query head h
    psi[p, c] = A[p, c] * (v_c W_O^(h))   contribution of entry c to the attention output a_p
    phi[p]    = grad_{a_p} Phi              gradient of the objective Phi = sum_p ||x_p^{L+1}||^2

The four scores of the paper's ablation (Table 1 / Table 4) combine two design choices:

    | name        | probes (PSP)  | magnitude-aware (MAS) | formula                                | paper          |
    |-------------|---------------|-----------------------|----------------------------------------|----------------|
    | taylor_sum  | sum over p    | no                    | sum_p <psi[p,c], phi[p]>               | Eq. (6)        |
    | taylor_max  | max over p    | no                    | max_p <psi[p,c], phi[p]>               | w/o MAS        |
    | bound_sum   | sum over p    | yes                   | sum_p A[p,c] ||v_c W_O|| ||phi[p]||    | Eq. (7) summed |
    | bound_max   | max over p    | yes                   | max_p A[p,c] ||v_c W_O|| ||phi[p]||    | Eq. (8) KVgrad |

As in kvpress, scores are computed per query head and reduced to one score per KV head by a max over
the query heads of its GQA group. Low scores are evicted first; the signed Taylor scores are used as
they are (an entry whose removal would *increase* Phi to first order gets a negative score).

Tensor layout (batch size 1 is all kvpress supports):

    attn_weights   (bsz, num_kv_heads, num_kv_groups, q_len, chunk_len)
    values         (bsz, num_kv_heads, chunk_len, head_dim)
    grad           (bsz, q_len, hidden_size)
    o_proj_weight  (hidden_size, num_kv_heads * num_kv_groups * head_dim)
    output         (bsz, num_kv_heads, chunk_len) for every variant
"""

from __future__ import annotations

import torch

VARIANTS = ("taylor_sum", "taylor_max", "bound_sum", "bound_max")


def value_output_norm(o_proj_weight: torch.Tensor, values: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
    """||v_c W_O^(h)||_2 for every query head h, shape (bsz, num_kv_heads, num_kv_groups, chunk_len)."""
    wo = _split_heads(o_proj_weight, values.shape[1], num_kv_groups, values.shape[-1])
    return torch.einsum("j h g i, b h t i -> b h g t j", wo, values).norm(dim=-1)


def taylor_terms(
    attn_weights: torch.Tensor, values: torch.Tensor, grad: torch.Tensor, o_proj_weight: torch.Tensor
) -> torch.Tensor:
    """
    First-order terms T[p, c] = A[p, c] * (v_c W_O^(h)) . grad_{a_p} Phi of Eq. (6), per query head.
    Shape (bsz, num_kv_heads, num_kv_groups, q_len, chunk_len).

    Computed as A * (grad W_O^(h)) . v_c, which avoids materialising v_c W_O for every (p, c) pair.
    """
    num_kv_heads, num_kv_groups = attn_weights.shape[1], attn_weights.shape[2]
    wo = _split_heads(o_proj_weight, num_kv_heads, num_kv_groups, values.shape[-1])
    grad_heads = torch.einsum("b q j, j h g i -> b h g q i", grad.to(wo.dtype), wo)
    dots = torch.einsum("b h g q i, b h t i -> b h g q t", grad_heads, values.to(wo.dtype))
    return attn_weights * dots.to(attn_weights.dtype)


def score_variants(
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    grad: torch.Tensor,
    o_proj_weight: torch.Tensor,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, torch.Tensor]:
    """All requested variants for one layer and chunk; each has shape (bsz, num_kv_heads, chunk_len)."""
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants {sorted(unknown)}; choose from {VARIANTS}")
    out: dict[str, torch.Tensor] = {}

    if any(v.startswith("bound") for v in variants):
        # Same operations, in the same order, as kvpress KVgradPress._update_scores
        value_norm = value_output_norm(o_proj_weight, values, attn_weights.shape[2])
        gradient_norm = grad.detach().float().norm(dim=-1)
        bound = attn_weights * value_norm[:, :, :, None, :]
        bound = bound * gradient_norm[:, None, None, :, None]
        if "bound_max" in variants:
            out["bound_max"] = bound.amax(dim=(-3, -2))  # max over group and probes
        if "bound_sum" in variants:
            out["bound_sum"] = bound.sum(dim=-2).amax(dim=-2)  # sum over probes, max over group

    if any(v.startswith("taylor") for v in variants):
        grad_full = grad.detach()
        if grad_full.dtype in (torch.float16, torch.bfloat16):  # never accumulate dot products in half precision
            grad_full = grad_full.float()
        terms = taylor_terms(attn_weights, values, grad_full, o_proj_weight)
        if "taylor_max" in variants:
            out["taylor_max"] = terms.amax(dim=(-3, -2))
        if "taylor_sum" in variants:
            out["taylor_sum"] = terms.sum(dim=-2).amax(dim=-2)

    return out


def _split_heads(o_proj_weight: torch.Tensor, num_kv_heads: int, num_kv_groups: int, head_dim: int) -> torch.Tensor:
    """o_proj weight (hidden, heads * head_dim) -> (hidden, num_kv_heads, num_kv_groups, head_dim).

    Query head index = kv_head * num_kv_groups + group, the convention of transformers' repeat_kv.
    """
    hidden_size = o_proj_weight.shape[0]
    return o_proj_weight.view(hidden_size, num_kv_heads, num_kv_groups, head_dim)
