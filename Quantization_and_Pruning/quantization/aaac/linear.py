from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import PackedAAACLayer, pack_aaac_fit, pack_aaac_layer, reconstruct_aaac_weight
from .fit import AAACLayerFit


class AAACLinear(nn.Module):
    """Dense reference evaluation wrapper for a packed 4-bit AAAC Linear layer."""

    def __init__(
        self,
        packed: PackedAAACLayer,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.group_size = packed.group_size
        self.selection_size = packed.selection_size

        self.register_buffer("packed_indices", packed.packed_indices)
        self.register_buffer("codebooks", packed.codebooks)
        self.register_buffer("scales", packed.scales)
        if packed.selector_bitset is not None:
            self.register_buffer("selector_bitset", packed.selector_bitset)
        else:
            self.register_buffer("selector_bitset", None)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_layer_fit(
        cls,
        layer_fit: AAACLayerFit,
        bias: Optional[torch.Tensor] = None,
    ) -> "AAACLinear":
        return cls(packed=pack_aaac_fit(layer_fit), bias=bias)

    @classmethod
    def from_quantized_tensors(
        cls,
        indices: torch.Tensor,
        codebooks: torch.Tensor,
        scales: torch.Tensor,
        selectors: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
        selection_size: int = 128,
    ) -> "AAACLinear":
        packed = pack_aaac_layer(
            indices=indices,
            codebooks=codebooks,
            scales=scales,
            selectors=selectors,
            group_size=group_size,
            selection_size=selection_size,
        )
        return cls(packed=packed, bias=bias)

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype or self.scales.dtype
        return reconstruct_aaac_weight(
            packed_indices=self.packed_indices,
            codebooks=self.codebooks,
            scales=self.scales,
            selector_bitset=self.selector_bitset,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
            selection_size=self.selection_size,
            out_dtype=target_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def packed_weight_bytes(self) -> int:
        total = (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.codebooks.numel() * self.codebooks.element_size()
            + self.scales.numel() * self.scales.element_size()
        )
        if self.selector_bitset is not None:
            total += self.selector_bitset.numel() * self.selector_bitset.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, g={self.group_size}, S={self.selection_size}"
        )


def set_module_by_name(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def replace_linear_layers_with_aaac(
    model: nn.Module,
    fits: Mapping[str, AAACLayerFit],
) -> nn.Module:
    """
    Replace only the named nn.Linear layers in `fits` with packed AAACLinear modules.
    Preserves all biases and unquantized parameters untouched.
    """
    modules_by_name = dict(model.named_modules())
    for name, layer_fit in fits.items():
        if name not in modules_by_name:
            raise KeyError(f"Target layer '{name}' not found in model")
        mod = modules_by_name[name]
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Target layer '{name}' is {type(mod).__name__}, expected nn.Linear")

        aaac_mod = AAACLinear.from_layer_fit(layer_fit, bias=mod.bias).to(device=mod.weight.device)
        set_module_by_name(model, name, aaac_mod)
    return model


def measure_model_storage_bytes(model: nn.Module) -> dict[str, int]:
    """
    Exact serialized byte accounting, completely separate from dense runtime memory.
    Handles shared/tied parameters (e.g., Qwen2.5 embed_tokens and lm_head) without double-counting.
    """
    seen_data_ptrs: set[tuple[int, int]] = set()
    quantized_weight_bytes = 0
    quantized_codebook_bytes = 0
    quantized_scale_bytes = 0
    quantized_selector_bytes = 0
    unquantized_param_bytes = 0
    quantized_layer_count = 0

    for mod in model.modules():
        if isinstance(mod, AAACLinear):
            quantized_layer_count += 1
            quantized_weight_bytes += mod.packed_indices.numel() * mod.packed_indices.element_size()
            quantized_codebook_bytes += mod.codebooks.numel() * mod.codebooks.element_size()
            quantized_scale_bytes += mod.scales.numel() * mod.scales.element_size()
            if mod.selector_bitset is not None:
                quantized_selector_bytes += (
                    mod.selector_bitset.numel() * mod.selector_bitset.element_size()
                )
        elif hasattr(mod, "packed_indices") and hasattr(mod, "scales"):
            quantized_layer_count += 1
            quantized_weight_bytes += mod.packed_indices.numel() * mod.packed_indices.element_size()
            quantized_scale_bytes += mod.scales.numel() * mod.scales.element_size()

    for param in model.parameters():
        key = (param.data_ptr(), param.numel() * param.element_size())
        if key not in seen_data_ptrs:
            seen_data_ptrs.add(key)
            unquantized_param_bytes += param.numel() * param.element_size()

    total_quantized_bytes = (
        quantized_weight_bytes
        + quantized_codebook_bytes
        + quantized_scale_bytes
        + quantized_selector_bytes
    )
    return {
        "quantized_layer_count": quantized_layer_count,
        "packed_indices_bytes": quantized_weight_bytes,
        "codebooks_bytes": quantized_codebook_bytes,
        "scales_bytes": quantized_scale_bytes,
        "selector_bitset_bytes": quantized_selector_bytes,
        "total_quantized_bytes": total_quantized_bytes,
        "unquantized_param_bytes": unquantized_param_bytes,
        "total_checkpoint_bytes": total_quantized_bytes + unquantized_param_bytes,
    }