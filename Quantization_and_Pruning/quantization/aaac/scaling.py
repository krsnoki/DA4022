"""Row-wise group scaling: split every weight row into contiguous groups of size g.

Scale rule (shared with RTN and the packed codec):

    scale[o, j] = max_{k in group j} |W[o, k]|      (absmax of the group)

rounded to ``scale_dtype``. The scale is always **strictly positive**:

* an all-zero group (absmax == 0) gets ``torch.finfo(scale_dtype).tiny``
  (6.1e-5 for FP16). Its normalised values are exactly 0, and it decodes to
  ``tiny * table[idx]``, i.e. at most ~6e-5 away from zero;
* values are clamped to ``[tiny, finfo(scale_dtype).max]`` before rounding, so
  rounding can never produce 0 or inf.

Positivity matters because, for S = g, the packed format stores the table
selector in the sign bit of the scale (decode uses ``abs(scale)``).

Normalisation divides by the *rounded* scale, so ``normalized * scale``
reproduces W up to float32 rounding. Because the scale is rounded to nearest,
``|normalized|`` can exceed 1 by at most half an ulp of ``scale_dtype``
(about 1 + 2^-11 for FP16); the learned tables absorb this.
"""

from __future__ import annotations

import torch


def _check_weight(weight: torch.Tensor, group_size: int) -> tuple[int, int]:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a 2-D tensor with layout [out_features, in_features]")
    if not weight.is_floating_point():
        raise ValueError(f"weight must be floating point, got {weight.dtype}")
    out_features, in_features = weight.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("weight must not be empty")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size}); "
            "padding is not supported"
        )
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains non-finite values")
    return out_features, in_features


def compute_group_scales(
    weight: torch.Tensor,
    group_size: int,
    scale_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Return strictly positive absmax scales, shape ``[out, in // group_size]``, in ``scale_dtype``."""
    out_features, in_features = _check_weight(weight, group_size)
    absmax = (
        weight.detach()
        .to(torch.float32)
        .abs()
        .reshape(out_features, in_features // group_size, group_size)
        .amax(dim=-1)
    )
    finfo = torch.finfo(scale_dtype)
    return absmax.clamp(min=finfo.tiny, max=finfo.max).to(scale_dtype)


def normalize_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Divide each group by its scale. Returns float32 ``[out, in]`` (values ~ in [-1, 1])."""
    out_features, in_features = _check_weight(weight, group_size)
    num_groups = in_features // group_size
    if scales.shape != (out_features, num_groups):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} != expected {(out_features, num_groups)}"
        )
    if not (scales > 0).all():
        raise ValueError("scales must be strictly positive")
    grouped = weight.detach().to(torch.float32).reshape(out_features, num_groups, group_size)
    scale = scales.to(device=weight.device, dtype=torch.float32).unsqueeze(-1)
    return (grouped / scale).reshape(out_features, in_features)


def expand_group_scales(scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat each scale over its group: ``[out, in // g]`` -> float32 ``[out, in]``."""
    return scales.to(torch.float32).repeat_interleave(group_size, dim=1)
