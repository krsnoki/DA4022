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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_indices", packed_indices)
        self.register_buffer("scales", scales)
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
    ) -> "RTNLinear":
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
        ).to(device=linear.weight.device)

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype or self.scales.dtype
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
) -> nn.Module:
    modules_by_name = dict(model.named_modules())
    for name in target_layer_names:
        mod = modules_by_name[name]
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Target layer '{name}' is {type(mod).__name__}, expected nn.Linear")
        rtn_mod = RTNLinear.from_linear(mod, group_size=group_size, scale_dtype=scale_dtype)
        set_module_by_name(model, name, rtn_mod)
    return model