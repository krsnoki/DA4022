"""CafeQ Appendix B, Algorithm 1: alternating adaptive rounding for a matrix product.

Given coupled ``W1 : (d1, h)`` and ``W2 : (h, d2)`` (paper layout) and a base rounding map ``Q``::

    W1_hat <- Q(W1)
    repeat I times:
        W2_hat <- Q(pinv(W1_hat) W1 W2)
        W1_hat <- Q(W1 W2 pinv(W2_hat))

Both results are still plain ``Q`` outputs, so inference-time dequantization is unchanged.

Deviation (documented in the paper card): Algorithm 1 is not monotone in practice, so with
``keep_best=True`` we evaluate the PQE after every half-step and return the best pair seen,
including the independently rounded pair ``(Q(W1), Q(W2))``.

:func:`adaptive_rounding_vo` is the grouped-query-attention-aware version used on real
``v_proj`` / ``o_proj`` weights.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from ...common.uniform import QuantSpec, quantize_dequantize

__all__ = ["adaptive_rounding", "adaptive_rounding_vo", "vo_paired_error"]


def _rel_err(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> float:
    return float(torch.linalg.norm(a @ b - p) / torch.linalg.norm(p).clamp_min(1e-30))


def adaptive_rounding(
    w1: torch.Tensor,
    w2: torch.Tensor,
    q1: Callable[[torch.Tensor], torch.Tensor],
    q2: Callable[[torch.Tensor], torch.Tensor],
    iterations: int = 5,
    keep_best: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Generic Algorithm 1 on paper-layout matrices; ``q1``/``q2`` are the base rounding maps."""
    w1 = w1.double()
    w2 = w2.double()
    p = w1 @ w2
    w1h, w2h = q1(w1).double(), q2(w2).double()
    indep = _rel_err(w1h, w2h, p)
    best = (indep, w1h, w2h)
    trace: List[float] = []
    for _ in range(iterations):
        w2h = q2(torch.linalg.pinv(w1h) @ w1 @ w2).double()
        e = _rel_err(w1h, w2h, p)
        trace.append(e)
        if e < best[0]:
            best = (e, w1h, w2h)
        w1h = q1(w1 @ (w2 @ torch.linalg.pinv(w2h))).double()
        e = _rel_err(w1h, w2h, p)
        trace.append(e)
        if e < best[0]:
            best = (e, w1h, w2h)
    if keep_best:
        err, out1, out2 = best
    else:
        err, out1, out2 = (trace[-1] if trace else indep), w1h, w2h
    return out1, out2, {"rel_pqe_independent": indep, "rel_pqe_final": err, "trace": trace}


# ----------------------------------------------------------------------------------------------
# Grouped-query attention (GQA) aware V/O version, nn.Linear layout in and out.
# ----------------------------------------------------------------------------------------------

def _split_vo(wv: torch.Tensor, wo: torch.Tensor, n_heads: int, n_kv: int, head_dim: int):
    """Paper-layout per-head blocks: W1 (n_kv, d_model, dh) and W2 (n_heads, dh, d_model)."""
    d_model = wv.shape[1]
    w1 = wv.reshape(n_kv, head_dim, d_model).transpose(1, 2)
    w2 = wo.t().reshape(n_heads, head_dim, wo.shape[0])
    return w1, w2


def _merge_v(w1: torch.Tensor) -> torch.Tensor:
    n_kv, d_model, dh = w1.shape
    return w1.transpose(1, 2).reshape(n_kv * dh, d_model)


def _merge_o(w2: torch.Tensor) -> torch.Tensor:
    n_heads, dh, d_model = w2.shape
    return w2.reshape(n_heads * dh, d_model).t()


def _trace_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched ``trace(x @ y)``."""
    return (x * y.transpose(-1, -2)).sum(dim=(-1, -2))


def vo_paired_error(
    wv_hat: torch.Tensor,
    wo_hat: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    n_heads: int,
    n_kv: int,
    head_dim: int,
) -> float:
    """Relative PQE summed over heads: ``sqrt(sum_h ||U_g V_h - W1_g W2_h||^2 / sum_h ||W1_g W2_h||^2)``.

    Uses ``dh x dh`` Gram matrices so the ``d_model x d_model`` products are never formed.
    """
    r = n_heads // n_kv
    a, b = _split_vo(wv_hat.double(), wo_hat.double(), n_heads, n_kv, head_dim)
    c, d = _split_vo(wv.double(), wo.double(), n_heads, n_kv, head_dim)
    a = a.repeat_interleave(r, dim=0)  # (n_heads, d_model, dh), query head h uses KV group h // r
    c = c.repeat_interleave(r, dim=0)
    bbT = b @ b.transpose(-1, -2)
    ddT = d @ d.transpose(-1, -2)
    ab = _trace_prod(a.transpose(-1, -2) @ a, bbT)
    cd = _trace_prod(c.transpose(-1, -2) @ c, ddT)
    cross = _trace_prod(a.transpose(-1, -2) @ c, d @ b.transpose(-1, -2))
    num = (ab + cd - 2 * cross).sum().clamp_min(0)
    return float(torch.sqrt(num / cd.sum().clamp_min(1e-30)))


def adaptive_rounding_vo(
    wv: torch.Tensor,
    wo: torch.Tensor,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    spec_v: QuantSpec,
    spec_o: QuantSpec,
    iterations: int = 5,
    keep_best: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Algorithm 1 for ``v_proj`` (``(n_kv*dh, d_model)``) and ``o_proj`` (``(d_model, n_heads*dh)``).

    The products to preserve are per query head ``h``: ``W1_{g(h)} W2_h`` with ``g(h) = h // r``.
    Targets are computed per head, then each full matrix is quantized with its *own* grouping
    (``o_proj`` groups span several heads), so the rounding map is exactly the one used at
    inference.
    """
    if n_heads % n_kv != 0:
        raise ValueError(f"n_heads={n_heads} not divisible by n_kv={n_kv}")
    r = n_heads // n_kv
    orig_dtype = wv.dtype
    wv64, wo64 = wv.double(), wo.double()
    w1, w2 = _split_vo(wv64, wo64, n_heads, n_kv, head_dim)
    d_model = w1.shape[1]

    def qv(w1_blocks: torch.Tensor) -> torch.Tensor:
        return quantize_dequantize(_merge_v(w1_blocks), spec_v).double()

    def qo(w2_blocks: torch.Tensor) -> torch.Tensor:
        return quantize_dequantize(_merge_o(w2_blocks), spec_o).double()

    def err(v_hat, o_hat) -> float:
        return vo_paired_error(v_hat, o_hat, wv64, wo64, n_heads, n_kv, head_dim)

    v_hat, o_hat = qv(w1), qo(w2)
    indep = err(v_hat, o_hat)
    best = (indep, v_hat, o_hat)
    trace: List[float] = []
    # W2 grouped by KV head and concatenated along columns: (n_kv, dh, r * d_model)
    w2_cat = w2.reshape(n_kv, r, head_dim, d_model).permute(0, 2, 1, 3).reshape(n_kv, head_dim, r * d_model)

    for _ in range(iterations):
        # W2_hat <- Q(pinv(W1_hat) W1 W2), per head
        w1h, _ = _split_vo(v_hat, o_hat, n_heads, n_kv, head_dim)
        t = torch.linalg.pinv(w1h) @ w1  # (n_kv, dh, dh)
        o_hat = qo(t.repeat_interleave(r, dim=0) @ w2)
        e = err(v_hat, o_hat)
        trace.append(e)
        if e < best[0]:
            best = (e, v_hat, o_hat)
        # W1_hat <- Q(W1 W2 pinv(W2_hat)), per KV group over its query heads
        _, w2h = _split_vo(v_hat, o_hat, n_heads, n_kv, head_dim)
        w2h_cat = w2h.reshape(n_kv, r, head_dim, d_model).permute(0, 2, 1, 3).reshape(n_kv, head_dim, r * d_model)
        s = w2_cat @ torch.linalg.pinv(w2h_cat)  # (n_kv, dh, dh)
        v_hat = qv(w1 @ s)
        e = err(v_hat, o_hat)
        trace.append(e)
        if e < best[0]:
            best = (e, v_hat, o_hat)

    if keep_best:
        final, v_out, o_out = best
    else:
        final, v_out, o_out = (trace[-1] if trace else indep), v_hat, o_hat
    info = {"rel_pqe_independent": indep, "rel_pqe_final": final, "trace": trace}
    return v_out.to(orig_dtype), o_out.to(orig_dtype), info
