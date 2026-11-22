"""CafeQ Appendix A / F: learned paired transform for coupled matrices.

All matrices here are in the paper's ``Y = X W`` layout. For coupled ``W1 : (d1, d2)`` and
``W2 : (d2, d3)``, any invertible ``M : (d2, d2)`` satisfies ``(W1 M)(M^{-1} W2) = W1 W2``, so the
transform is absorbed into the weights and costs nothing at inference. CafeQ learns ``M`` by
minimising a smooth surrogate of the paired quantization error (PQE)

    PQE = || Q(W1 M) Q(M^{-1} W2) - W1 W2 ||_F.

With ``U = W1 M`` and ``V = M^{-1} W2`` and channel-wise maxima
``m_u(i) = max_j |U_ij|``, ``m_v(j) = max_i |V_ij|``, the default surrogate is

    LogSumExp(U, V) = (1/t) log( sum_i exp(t m_u(i)) + sum_j exp(t m_v(j)) ).

``M`` is unconstrained (Adam), optionally with the orthonormality penalty
``lambda * ||M M^T - I||_F / sqrt(d)`` (Appendix F.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ...common.transforms import random_orthogonal
from .config import PairedTransformConfig

__all__ = [
    "channel_maxima",
    "pseudo_loss",
    "orth_penalty",
    "paired_error",
    "PairedTransformResult",
    "learn_paired_transform",
]


def channel_maxima(u: torch.Tensor, v: torch.Tensor):
    """``m_u`` of length ``d1`` and ``m_v`` of length ``d3`` (Appendix A)."""
    return u.abs().amax(dim=1), v.abs().amax(dim=0)


def pseudo_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    cfg: PairedTransformConfig,
    scale_u: float = 1.0,
    scale_v: float = 1.0,
) -> torch.Tensor:
    mu, mv = channel_maxima(u, v)
    mu, mv = mu / scale_u, mv / scale_v
    if cfg.pseudo_loss == "logsumexp":
        t = cfg.temperature
        return torch.logsumexp(t * torch.cat([mu, mv]), dim=0) / t
    if cfg.pseudo_loss == "sum_sq":
        # Appendix F.1 (1): mean of squared channel-wise maxima of each matrix.
        return (mu ** 2).mean() + (mv ** 2).mean()
    if cfg.pseudo_loss == "sum_sq_weighted":
        # Appendix F.1 (2): weight each side by the Frobenius norm of its partner.
        return (v / scale_v).norm() * (mu ** 2).mean() + (u / scale_u).norm() * (mv ** 2).mean()
    raise ValueError(f"unknown pseudo loss {cfg.pseudo_loss!r}")


def orth_penalty(m: torch.Tensor) -> torch.Tensor:
    d = m.shape[0]
    eye = torch.eye(d, dtype=m.dtype, device=m.device)
    return torch.linalg.norm(m @ m.t() - eye) / d ** 0.5


def paired_error(u_hat: torch.Tensor, v_hat: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, relative: bool = True) -> float:
    """``||U_hat V_hat - W1 W2||_F`` (optionally relative to ``||W1 W2||_F``), computed in float64."""
    p = w1.double() @ w2.double()
    e = torch.linalg.norm(u_hat.double() @ v_hat.double() - p)
    if relative:
        e = e / torch.linalg.norm(p).clamp_min(1e-30)
    return float(e)


@dataclass
class PairedTransformResult:
    m: torch.Tensor
    m_inv: torch.Tensor
    info: Dict[str, float] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)


def learn_paired_transform(
    w1: torch.Tensor,
    w2: torch.Tensor,
    cfg: PairedTransformConfig,
    generator: Optional[torch.Generator] = None,
    pqe_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    log=None,
) -> PairedTransformResult:
    """Learn ``M`` for coupled paper-layout matrices ``w1 : (d1, d2)`` and ``w2 : (d2, d3)``.

    ``pqe_fn(U, V)`` (optional) returns the true paired error for logging/info; it is called only
    at the start and end, never inside the optimisation loop.
    """
    w1 = w1.detach().float()
    w2 = w2.detach().float()
    d2 = w1.shape[1]
    if w2.shape[0] != d2:
        raise ValueError(f"inner dimensions differ: {tuple(w1.shape)} x {tuple(w2.shape)}")

    if cfg.init == "identity":
        m0 = torch.eye(d2, device=w1.device)
    else:
        m0 = random_orthogonal(d2, generator).to(w1.device)

    su = float(w1.abs().max().clamp_min(1e-12)) if cfg.normalize else 1.0
    sv = float(w2.abs().max().clamp_min(1e-12)) if cfg.normalize else 1.0

    def objective(m: torch.Tensor) -> torch.Tensor:
        u = w1 @ m
        v = torch.linalg.solve(m, w2)  # M^{-1} W2 without forming the inverse
        loss = pseudo_loss(u, v, cfg, su, sv)
        if cfg.orth_weight > 0:
            loss = loss + cfg.orth_weight * orth_penalty(m)
        return loss

    param = torch.nn.Parameter(m0.clone())
    opt = torch.optim.Adam([param], lr=cfg.lr) if cfg.optimizer == "adam" else torch.optim.SGD([param], lr=cfg.lr)

    with torch.no_grad():
        init_loss = float(objective(param))
    best_loss, best = init_loss, param.detach().clone()
    history: List[float] = [init_loss]

    for step in range(1, cfg.iterations + 1):
        opt.zero_grad(set_to_none=True)
        try:
            loss = objective(param)
        except RuntimeError as exc:
            if log:
                log(f"  paired transform: stopping at step {step}: {exc}")
            break
        if not torch.isfinite(loss):
            if log:
                log(f"  paired transform: non-finite loss at step {step}; stopping")
            break
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        if cfg.keep_best and lv < best_loss:
            best_loss, best = lv, param.detach().clone()
        if cfg.log_every and step % cfg.log_every == 0:
            history.append(lv)
            if log:
                log(f"  paired transform step {step}: loss={lv:.5f}")

    m = best if cfg.keep_best else param.detach().clone()
    m_inv = torch.linalg.inv(m.double()).float()
    with torch.no_grad():
        final_loss = float(objective(m))
    sv_m = torch.linalg.svdvals(m.double())
    info = {
        "pseudo_init": init_loss,
        "pseudo_final": final_loss,
        "m_sigma_min": float(sv_m.min()),
        "m_sigma_max": float(sv_m.max()),
    }
    if pqe_fn is not None:
        info["pqe_init"] = pqe_fn(w1 @ m0, torch.linalg.solve(m0, w2))
        info["pqe_final"] = pqe_fn(w1 @ m, m_inv @ w2)
    return PairedTransformResult(m=m, m_inv=m_inv, info=info, history=history)
