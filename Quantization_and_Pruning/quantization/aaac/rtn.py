from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import pack_int4_indices, unpack_int4_indices
from .linear import set_module_by_name
from .scaling import compute_group_scales, expand_group_scales, normalize_weight

RTN_QMIN = -8
RTN_QMAX = 7
RTN_NORM_DENOM = 7.0


def quantize_weight_rtn_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = compute_group_scales(weight, group_size=group_size, scale_dtype=scale_dtype)
    normalized = normalize_weight(weight, scales, group_size=group_size)
    q = torch.round(normalized * RTN_NORM_DENOM).clamp(RTN_QMIN, RTN_QMAX).to(torch.int64)
    indices = (q - RTN_QMIN).to(torch.uint8)
    packed_indices = pack_int4_indices(indices)
    return packed_indices, scales


def reconstruct_rtn_weight(
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    group_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    indices = unpack_int4_indices(packed_indices, in_features).to(torch.float32)
    q = (indices + RTN_QMIN) / RTN_NORM_DENOM
    expanded_scales = expand_group_scales(scales, group_size).to(q.device)
    return (q * expanded_scales).to(out_dtype)


RTN_ASYM_LEVELS = 15  # codes 0..15


def quantize_weight_rtn_int4_asym(
    weight: torch.Tensor,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric (min/max, zero-point) INT4 RTN, the usual GPTQ/AWQ-style baseline.

    Per group of ``group_size`` weights along ``in``:
        scale = (max - min) / 15          (rounded to ``scale_dtype``, always > 0)
        zero  = clamp(round(-min / scale), 0, 15)
        q     = clamp(round(w / scale) + zero, 0, 15)
        w_hat = (q - zero) * scale
    All 16 codes are usable. Returns (packed indices uint8 [out, in/2],
    scales [out, in/g], zero points uint8 [out, in/g]).
    """
    if weight.ndim != 2 or weight.shape[1] % group_size != 0:
        raise ValueError("weight must be [out, in] with in divisible by group_size")
    w = weight.detach().to(torch.float32)
    out_f, in_f = w.shape
    groups = w.reshape(out_f, in_f // group_size, group_size)
    w_min = groups.amin(dim=-1).clamp(max=0.0)
    w_max = groups.amax(dim=-1).clamp(min=0.0)
    finfo = torch.finfo(scale_dtype)
    scales = ((w_max - w_min) / RTN_ASYM_LEVELS).clamp(min=finfo.tiny, max=finfo.max).to(scale_dtype)
    s32 = scales.to(torch.float32)
    zeros = torch.round(-w_min / s32).clamp(0, RTN_ASYM_LEVELS)
    q = (torch.round(groups / s32.unsqueeze(-1)) + zeros.unsqueeze(-1)).clamp(0, RTN_ASYM_LEVELS)
    indices = q.reshape(out_f, in_f).to(torch.uint8)
    return pack_int4_indices(indices), scales, zeros.to(torch.uint8)


def reconstruct_rtn_weight_asym(
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    in_features: int,
    group_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    q = unpack_int4_indices(packed_indices, in_features).to(torch.float32)
    zeros = expand_group_scales(zero_points.to(torch.float32), group_size).to(q.device)
    expanded_scales = expand_group_scales(scales, group_size).to(q.device)
    return ((q - zeros) * expanded_scales).to(out_dtype)


class RTNLinear(nn.Module):
    """Dense reference evaluation wrapper for a packed fixed-grid INT4 RTN Linear layer."""

    def __init__(
        self,
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
        zero_points: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_indices", packed_indices)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)  # None for the symmetric grid
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float16,
        asymmetric: bool = False,
    ) -> "RTNLinear":
        zero_points = None
        if asymmetric:
            packed, scales, zero_points = quantize_weight_rtn_int4_asym(
                linear.weight, group_size=group_size, scale_dtype=scale_dtype
            )
        else:
            packed, scales = quantize_weight_rtn_int4(
                linear.weight,
                group_size=group_size,
                scale_dtype=scale_dtype,
            )
        return cls(
            packed_indices=packed,
            scales=scales,
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias,
            zero_points=zero_points,
        ).to(device=linear.weight.device)

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype or self.scales.dtype
        if self.zero_points is not None:
            return reconstruct_rtn_weight_asym(
                packed_indices=self.packed_indices,
                scales=self.scales,
                zero_points=self.zero_points,
                in_features=self.in_features,
                group_size=self.group_size,
                out_dtype=target_dtype,
            )
        return reconstruct_rtn_weight(
            packed_indices=self.packed_indices,
            scales=self.scales,
            in_features=self.in_features,
            group_size=self.group_size,
            out_dtype=target_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def quantize_model_rtn(
    model: nn.Module,
    target_layer_names: Iterable[str],
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
    asymmetric: bool = False,
) -> nn.Module:
    modules_by_name = dict(model.named_modules())
    for name in target_layer_names:
        mod = modules_by_name[name]
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Target layer '{name}' is {type(mod).__name__}, expected nn.Linear")
        rtn_mod = RTNLinear.from_linear(
            mod, group_size=group_size, scale_dtype=scale_dtype, asymmetric=asymmetric
        )
        set_module_by_name(model, name, rtn_mod)
    return model