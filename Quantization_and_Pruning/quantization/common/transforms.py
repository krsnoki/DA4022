"""Structured (block-diagonal) linear transforms used before weight quantization.

Convention (paper / row-vector layout)
--------------------------------------
A block-diagonal matrix ``M = diag(B_1, ..., B_{d/k})`` is stored as a tensor ``blocks`` of shape
``(d // k, k, k)``. Multiplying a row vector ``x`` (last dim ``d``) on the right, ``x @ M``, is
:func:`block_diag_matmul`. It costs ``O(d * k)`` per vector instead of ``O(d^2)``.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "hadamard_matrix",
    "random_hadamard",
    "random_orthogonal",
    "block_diag_matmul",
    "block_diag_dense",
    "pick_block_size",
    "random_block_diagonal",
]


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Unnormalised Sylvester Hadamard matrix of size ``n`` (``n`` must be a power of two)."""
    if not _is_pow2(n):
        raise ValueError(f"Sylvester Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def random_hadamard(k: int, generator: Optional[torch.Generator] = None, dtype=torch.float32, device=None) -> torch.Tensor:
    """Orthogonal ``H diag(s) / sqrt(k)`` with random signs ``s``."""
    signs = torch.randint(0, 2, (k,), generator=generator).to(dtype=dtype) * 2 - 1
    h = hadamard_matrix(k, dtype=dtype) * signs.unsqueeze(0) / k ** 0.5
    return h.to(device) if device is not None else h


def random_orthogonal(k: int, generator: Optional[torch.Generator] = None, dtype=torch.float32, device=None) -> torch.Tensor:
    """Haar-random orthogonal matrix (QR of a Gaussian with sign correction)."""
    a = torch.randn(k, k, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    q = q.to(dtype)
    return q.to(device) if device is not None else q


def block_diag_matmul(x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ blockdiag(blocks)`` for ``x`` of shape ``(..., d)``."""
    nb, k, k2 = blocks.shape
    if k != k2:
        raise ValueError(f"blocks must be square, got {tuple(blocks.shape)}")
    d = x.shape[-1]
    if d != nb * k:
        raise ValueError(f"last dim {d} != n_blocks*block_size = {nb}*{k}")
    xs = x.reshape(*x.shape[:-1], nb, k)
    y = torch.einsum("...bi,bij->...bj", xs, blocks)
    return y.reshape(x.shape)


def block_diag_dense(blocks: torch.Tensor) -> torch.Tensor:
    """Dense ``(d, d)`` matrix for a block tensor (tests / debugging only)."""
    return torch.block_diag(*blocks.unbind(0))


def pick_block_size(d: int, requested: int, power_of_two: bool = False) -> int:
    """Largest block size ``<= requested`` that divides ``d`` (optionally restricted to powers of two)."""
    if requested <= 0:
        raise ValueError("requested block size must be positive")
    for k in range(min(requested, d), 0, -1):
        if d % k == 0 and (not power_of_two or _is_pow2(k)):
            return k
    return 1


def random_block_diagonal(
    d: int,
    block_size: int,
    kind: str = "rotation",
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Stack of ``d // block_size`` random orthogonal blocks: ``kind`` in {rotation, hadamard, identity}."""
    if d % block_size != 0:
        raise ValueError(f"block_size={block_size} does not divide d={d}")
    nb = d // block_size
    if kind == "identity":
        return torch.eye(block_size, dtype=dtype).expand(nb, block_size, block_size).clone()
    if kind == "hadamard":
        return torch.stack([random_hadamard(block_size, generator, dtype) for _ in range(nb)])
    if kind == "rotation":
        return torch.stack([random_orthogonal(block_size, generator, dtype) for _ in range(nb)])
    raise ValueError(f"unknown block kind {kind!r}")
