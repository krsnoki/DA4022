"""Closed-form Frobenius norm of the softmax Jacobian.

For one softmax row ``p = softmax(z)`` the Jacobian is ``J = diag(p) - p p^T``.
Expanding ``||J||_F^2 = sum_ij J_ij^2`` and using ``sum_i p_i = 1`` gives

    ||J||_F^2 = sum_i p_i^2 - 2 * sum_i p_i^3 + (sum_i p_i^2)^2

(Pandey et al., "Jacobian-guided Noise Injection for Quantization Robustness in
Large Language Models", Eq. 12 / Appendix A).  The value lies in ``[0, 1/4]``:
it is 0 for a one-hot row and maximal (1/4) when the mass is split evenly over
two entries.  Cost is O(N) per row instead of the O(N^2) of materialising J,
which is why we never call ``torch.autograd.functional.jacobian`` in training
(it is used only in the unit test that checks this formula).
"""

from __future__ import annotations

import torch


def softmax_jacobian_fro_sq(probs: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm of the softmax Jacobian for every row.

    Args:
        probs: tensor ``[..., N]`` whose last dimension is a probability
            distribution (output of ``softmax(dim=-1)``).  Masked entries must
            already be zero, which is what an additive ``-inf``/``finfo.min``
            mask followed by softmax produces.

    Returns:
        Tensor ``[...]`` (last dim reduced), computed in float32.
    """
    p = probs.float()
    s2 = (p * p).sum(dim=-1)
    s3 = (p * p * p).sum(dim=-1)
    return (s2 - 2.0 * s3 + s2 * s2).clamp_min(0.0)


def softmax_jacobian_fro(probs: torch.Tensor) -> torch.Tensor:
    """Frobenius norm (not squared) of the softmax Jacobian for every row."""
    return softmax_jacobian_fro_sq(probs).sqrt()


def softmax_jacobian_explicit(probs: torch.Tensor) -> torch.Tensor:
    """Explicit ``diag(p) - p p^T`` for a single row ``[N]`` (reference/testing)."""
    p = probs.float()
    return torch.diag(p) - torch.outer(p, p)
