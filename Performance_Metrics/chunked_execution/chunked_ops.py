"""Chunked execution of alternating compute-bound / memory-bound workloads.

Idea (Schultheis, Kleinegger, Alistarh -- "One Simple Trick for Improving the
Performance of Energy-Limited Local Inference and Training"): split a large
matmul -> memory-bound op -> matmul pipeline into row-chunks so that compute
and memory phases interleave at a finer granularity. This smooths GPU power
draw and can avoid power/thermal clock throttling.

NOTE: the exact form of the 2nd matmul in the paper's Listing 1 should be
verified against the paper. Here it is implemented as the contraction over the
chunked dimension (P_c^T @ A_c accumulated into one shared buffer), which is
the LM-head weight-gradient pattern. See PAPER_CARD.md "Differences".
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _split_bounds(m: int, chunks: int) -> list[tuple[int, int]]:
    if chunks < 1:
        raise ValueError(f"chunks must be >= 1, got {chunks}")
    if chunks > m:
        raise ValueError(f"chunks ({chunks}) cannot exceed rows ({m})")
    base, rem = divmod(m, chunks)
    bounds, start = [], 0
    for i in range(chunks):
        end = start + base + (1 if i < rem else 0)
        bounds.append((start, end))
        start = end
    return bounds


def _softmax_fn(compile_softmax: bool):
    fn = lambda x: torch.softmax(x, dim=-1)  # noqa: E731
    return torch.compile(fn) if compile_softmax else fn


@torch.no_grad()
def mm_softmax_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    chunks: int = 1,
    *,
    compile_softmax: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """a: (M,K), b: (K,N) -> out: (N,K) = sum_c softmax(a_c @ b)^T @ a_c.

    chunks=1 is the unmodified baseline. Chunked intermediates are (M/chunks, N)
    instead of (M, N), which also lowers peak memory.
    """
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dims differ: {k} vs {k2}")
    sm = _softmax_fn(compile_softmax)
    if out is None:
        out = torch.zeros((n, k), device=a.device, dtype=a.dtype)
    else:
        out.zero_()
    for lo, hi in _split_bounds(m, chunks):
        a_c = a[lo:hi]
        p = sm(a_c @ b)                      # compute-bound then memory-bound
        out.addmm_(p.t(), a_c)               # compute-bound, accumulate in shared buffer
    return out


def chunked_lm_head_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunks: int = 1,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean cross-entropy of (hidden @ weight^T) vs targets, computed in chunks.

    hidden: (T, H), weight: (V, H), targets: (T,). Each chunk's logits are
    recomputed in backward (activation checkpointing), so only (T/chunks, V)
    logits are alive at once. Differentiable w.r.t. hidden and weight.
    """
    t_total = hidden.shape[0]
    valid = (targets != ignore_index).sum().clamp(min=1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for lo, hi in _split_bounds(t_total, chunks):
        tgt = targets[lo:hi]
        if (tgt == ignore_index).all():
            continue
        # replace ignored targets so CE sum is over valid tokens only
        mask = tgt != ignore_index
        safe_t = torch.where(mask, tgt, torch.zeros_like(tgt))

        def fn(h, w, t=safe_t, mk=mask):
            logits = (h @ w.t()).float()
            loss = torch.nn.functional.cross_entropy(logits, t, reduction="none")
            return (loss * mk).sum()

        total = total + checkpoint(fn, hidden[lo:hi], weight, use_reentrant=False)
    return total / valid