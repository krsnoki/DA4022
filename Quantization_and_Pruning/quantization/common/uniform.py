"""Scalar uniform (min-max, asymmetric) weight quantization.

Shared primitive for weight-only PTQ methods in this module.

Layout convention
-----------------
All functions take weights in the ``torch.nn.Linear`` layout ``(out_features, in_features)``.
Quantization groups run along ``in_features`` (the contraction dimension):

* ``group_size=None``  -> per-channel: one group per output row, spanning all inputs.
* ``group_size=g``     -> sub-channel: each output row is split into ``in_features // g`` groups.

This matches the CafeQ paper's "per-channel quantization along the contraction dimension" and
its sub-channel block sizes 256 / 128.

The quantizer is Eq. (1) of the CafeQ paper::

    s = (w_max - w_min) / (2^N - 1)
    w_hat = s * round((w - w_min) / s) + w_min

Important: these functions return *dequantized* floating-point tensors ("fake quantization").
They do not produce or run low-bit kernels. ``quantize_to_codes`` is provided so storage size can be
reported honestly, but inference in this repository uses the dequantized weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "QuantSpec",
    "quantize_dequantize",
    "quantize_to_codes",
    "dequantize_codes",
    "group_ranges",
    "relative_l2",
    "quantize_paper_layout",
]


@dataclass(frozen=True)
class QuantSpec:
    """Bit-width and grouping for uniform asymmetric weight quantization."""

    bits: int = 4
    group_size: Optional[int] = None  # None == per-channel

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or not 1 <= self.bits <= 16:
            raise ValueError(f"bits must be an int in [1, 16], got {self.bits!r}")
        if self.group_size is not None and (not isinstance(self.group_size, int) or self.group_size <= 0):
            raise ValueError(f"group_size must be a positive int or None, got {self.group_size!r}")

    @property
    def levels(self) -> int:
        """Largest integer code, ``2^N - 1``."""
        return 2 ** self.bits - 1

    def check(self, in_features: int) -> int:
        """Return the effective group size for ``in_features`` or raise if it does not divide."""
        g = in_features if self.group_size is None else self.group_size
        if in_features % g != 0:
            raise ValueError(f"group_size={g} does not divide in_features={in_features}")
        return g


def _grouped(w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D (out, in) weight, got shape {tuple(w.shape)}")
    out_f, in_f = w.shape
    g = spec.check(in_f)
    return w.reshape(out_f, in_f // g, g)


def _scale_and_min(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    scale = (hi - lo) / spec.levels
    # Constant groups: any positive scale reproduces them exactly (code 0 -> lo).
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale, lo


def quantize_dequantize(
    w: torch.Tensor,
    spec: QuantSpec,
    stochastic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Round-to-nearest (or stochastic) uniform quantization; returns dequantized weights.

    Computation is done in float32 and cast back to ``w.dtype``.
    """
    orig_dtype = w.dtype
    x = _grouped(w.detach().float(), spec)
    scale, lo = _scale_and_min(x, spec)
    z = (x - lo) / scale
    if stochastic:
        f = torch.floor(z)
        u = torch.rand(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        z = f + (u < (z - f)).to(z.dtype)
    else:
        z = torch.round(z)
    z = z.clamp_(0, spec.levels)
    return (z * scale + lo).reshape(w.shape).to(orig_dtype)


def quantize_to_codes(w: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(codes, scale, w_min)``: integer codes plus per-group scale and offset.

    ``codes`` has shape ``(out, n_groups, group)`` and dtype ``torch.int32`` (not bit-packed).
    """
    x = _grouped(w.detach().float(), spec)
    scale, lo = _scale_and_min(x, spec)
    codes = torch.round((x - lo) / scale).clamp_(0, spec.levels).to(torch.int32)
    return codes, scale, lo


def dequantize_codes(codes: torch.Tensor, scale: torch.Tensor, w_min: torch.Tensor) -> torch.Tensor:
    out_f, n_groups, g = codes.shape
    return (codes.float() * scale + w_min).reshape(out_f, n_groups * g)


def group_ranges(w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Differentiable per-group range ``max - min``; shape ``(out, n_groups)``."""
    x = _grouped(w, spec)
    return x.amax(dim=-1) - x.amin(dim=-1)


def relative_l2(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """``||approx - ref||_F / ||ref||_F`` as a Python float."""
    ref64 = ref.detach().double()
    return float(torch.linalg.norm(approx.detach().double() - ref64) / torch.linalg.norm(ref64).clamp_min(1e-30))


def quantize_paper_layout(w_paper: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Quantize a matrix in the paper's ``Y = X W`` layout ``(in, out)``.

    Groups still run along the contraction dimension (here: rows), so this is just a transpose
    around :func:`quantize_dequantize`.
    """
    return quantize_dequantize(w_paper.t(), spec).t()
