"""CafeQ Section 3.1: learned block-diagonal transform for a single weight matrix.

Paper layout vs. ``nn.Linear`` layout
------------------------------------
The paper writes ``Y = X W`` with ``W`` of shape ``(d1, d2) = (in, out)``. It learns an invertible
``M`` of shape ``(in, in)``, quantizes ``M W`` and evaluates ``X M^{-1} Q(M W)`` at inference.

``nn.Linear`` stores ``W_lin = W^T`` of shape ``(out, in)``. Therefore

* transformed linear weight: ``W_lin' = W_lin @ M^T``  (:func:`transform_weight`)
* inference:               ``y = F.linear(x @ M^{-1}, Q(W_lin'))``
* effective weight:        ``W_eff = Q(W_lin') @ M^{-T}``  (:func:`effective_weight`)

``M`` is block diagonal (``blocks`` of shape ``(in // k, k, k)``), so ``x @ M^{-1}`` costs
``O(in * k)`` extra FLOPs per token.

Surrogate loss (Eq. 4)
----------------------
Under stochastic rounding the expected reconstruction error satisfies::

    E||M^{-1} Q(MW) - W||_F^2 = sum_{i,j,k} (M^{-1})_{ij}^2 Var(Delta_{jk})

with ``Var(Delta_{jk}) <= R_{g(j,k)}^2 / (4 (2^N - 1)^2)`` where ``R`` is the range of the
quantization group containing entry ``(j, k)``.

* ``loss="paper_eq4"`` implements the paper's final printed bound literally::

      d2 / (2^N-1)^2 * sum_{i,j} (M^{-1})_{ij}^2 ||(MW)_j||_inf^2

* ``loss="grouped_bound"`` implements the tighter intermediate bound using the actual
  quantization groups (our addition; see the paper card for why).

Both are divided by ``||W||_F^2`` so the learning rate is scale-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ...common.transforms import block_diag_matmul, random_block_diagonal
from ...common.uniform import QuantSpec, group_ranges, quantize_dequantize, relative_l2
from .config import BlockTransformConfig

__all__ = [
    "transform_weight",
    "effective_weight",
    "surrogate_loss",
    "BlockTransformResult",
    "learn_block_transform",
]


def transform_weight(w_lin: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """``W_lin @ M^T`` for block-diagonal ``M``."""
    return block_diag_matmul(w_lin, blocks.transpose(-1, -2))


def effective_weight(w_q: torch.Tensor, inv_blocks: torch.Tensor) -> torch.Tensor:
    """Dense effective ``nn.Linear`` weight ``Q(W_lin M^T) @ M^{-T}`` (for error measurement)."""
    return block_diag_matmul(w_q, inv_blocks.transpose(-1, -2))


def surrogate_loss(
    w_lin: torch.Tensor,
    blocks: torch.Tensor,
    spec: QuantSpec,
    form: str = "paper_eq4",
    inv_blocks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Relative Eq. (4)-style upper bound on the expected quantization error. Differentiable in ``blocks``."""
    if inv_blocks is None:
        inv_blocks = torch.linalg.inv(blocks)
    out_f, in_f = w_lin.shape
    # c_j = sum_i (M^{-1})_{ij}^2 : squared column norms of M^{-1} (row-vector convention x @ M^{-1}).
    col_sq = (inv_blocks ** 2).sum(dim=-2).reshape(in_f)
    wt = transform_weight(w_lin, blocks)  # (out, in)
    levels = spec.levels
    if form == "paper_eq4":
        # ||(MW)_j||_inf, where (MW)_j is row j of the paper-layout M W == column j of wt.
        row_inf = wt.abs().amax(dim=0)
        bound = out_f * (col_sq * row_inf ** 2).sum() / levels ** 2
    elif form == "grouped_bound":
        g = spec.check(in_f)
        r = group_ranges(wt, spec)  # (out, in // g)
        s = (r ** 2).sum(dim=0).repeat_interleave(g)  # (in,)
        bound = (col_sq * s).sum() / (4 * levels ** 2)
    else:
        raise ValueError(f"unknown surrogate loss form {form!r}")
    return bound / (w_lin ** 2).sum().clamp_min(1e-30)


@dataclass
class BlockTransformResult:
    blocks: torch.Tensor
    inv_blocks: torch.Tensor
    weight_q: torch.Tensor  # dequantized Q(W_lin @ M^T), shape (out, in)
    info: Dict[str, float] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)


def _true_rel_error(w_lin: torch.Tensor, blocks: torch.Tensor, spec: QuantSpec):
    inv = torch.linalg.inv(blocks)
    wq = quantize_dequantize(transform_weight(w_lin, blocks), spec)
    return relative_l2(effective_weight(wq, inv), w_lin), inv, wq


def learn_block_transform(
    w_lin: torch.Tensor,
    spec: QuantSpec,
    cfg: BlockTransformConfig,
    block_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    log=None,
) -> BlockTransformResult:
    """Learn a block-diagonal ``M`` for one ``(out, in)`` weight and return its quantized form.

    Optimisation runs in float32 on ``w_lin.device``; the original weight is not modified.
    """
    w = w_lin.detach().float()
    out_f, in_f = w.shape
    k = block_size or cfg.block_size
    if in_f % k != 0:
        raise ValueError(f"block size {k} does not divide in_features={in_f}")

    init = random_block_diagonal(in_f, k, kind=cfg.init, generator=generator).to(w.device)
    rtn_err = relative_l2(quantize_dequantize(w, spec), w)
    init_err, _, _ = _true_rel_error(w, init, spec)

    param = torch.nn.Parameter(init.clone())
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam([param], lr=cfg.lr)
    else:
        opt = torch.optim.SGD([param], lr=cfg.lr)

    with torch.no_grad():
        best_loss = float(surrogate_loss(w, param, spec, cfg.loss))
    init_loss = best_loss
    best = param.detach().clone()
    history: List[float] = [init_loss]

    for step in range(1, cfg.iterations + 1):
        opt.zero_grad(set_to_none=True)
        try:
            loss = surrogate_loss(w, param, spec, cfg.loss)
        except RuntimeError as exc:  # singular block (torch.linalg.LinAlgError subclasses RuntimeError)
            if log:
                log(f"  block transform: stopping at step {step}: {exc}")
            break
        if not torch.isfinite(loss):
            if log:
                log(f"  block transform: non-finite loss at step {step}; stopping")
            break
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        if cfg.keep_best and lv < best_loss:
            best_loss, best = lv, param.detach().clone()
        if cfg.log_every and step % cfg.log_every == 0:
            history.append(lv)
            if log:
                log(f"  block transform step {step}: surrogate={lv:.5f}")

    blocks = best if cfg.keep_best else param.detach().clone()
    final_err, inv, wq = _true_rel_error(w, blocks, spec)
    fell_back = False
    if cfg.fallback_if_worse and final_err > init_err:
        blocks, fell_back = init, True
        final_err, inv, wq = _true_rel_error(w, blocks, spec)

    with torch.no_grad():
        final_loss = float(surrogate_loss(w, blocks, spec, cfg.loss))
    info = {
        "block_size": float(k),
        "surrogate_init": init_loss,
        "surrogate_final": final_loss,
        "rel_l2_rtn": rtn_err,
        "rel_l2_init": init_err,
        "rel_l2_final": final_err,
        "fell_back_to_init": float(fell_back),
    }
    return BlockTransformResult(
        blocks=blocks, inv_blocks=inv, weight_q=wq.to(w_lin.dtype), info=info, history=history
    )
