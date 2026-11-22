"""Inference module for CafeQ-style weight-only quantized linears, plus save/load helpers.

``CafeQLinear`` stores *dequantized* weights (fake quantization) and, optionally, the inverse
block-diagonal transform that must be applied to the input online:

    y = F.linear(x @ M^{-1}, Q(W M^T), b)

It does not run low-bit kernels; reported speed numbers must say so.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.transforms import block_diag_matmul

__all__ = ["CafeQLinear", "quantized_layout", "save_quantized", "load_quantized"]


class CafeQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        inv_block_shape: Optional[Tuple[int, int]] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kw = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **kw), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kw), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        if inv_block_shape is not None:
            nb, k = inv_block_shape
            if nb * k != in_features:
                raise ValueError(f"inv blocks {nb}x{k} do not cover in_features={in_features}")
            self.register_buffer("inv_blocks", torch.empty(nb, k, k, **kw))
        else:
            self.register_buffer("inv_blocks", None)
        self.quant_meta: Dict[str, object] = {}

    @classmethod
    def build(
        cls,
        weight_q: torch.Tensor,
        bias: Optional[torch.Tensor],
        inv_blocks: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> "CafeQLinear":
        out_f, in_f = weight_q.shape
        shape = None if inv_blocks is None else (inv_blocks.shape[0], inv_blocks.shape[1])
        mod = cls(in_f, out_f, bias is not None, shape, dtype=weight_q.dtype, device=weight_q.device)
        with torch.no_grad():
            mod.weight.copy_(weight_q)
            if bias is not None:
                mod.bias.copy_(bias.to(weight_q.dtype))
            if inv_blocks is not None:
                mod.inv_blocks.copy_(inv_blocks.to(weight_q.dtype))
        mod.quant_meta = dict(meta or {})
        return mod

    @property
    def online_block_size(self) -> int:
        return 0 if self.inv_blocks is None else int(self.inv_blocks.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_blocks is not None:
            x = block_diag_matmul(x, self.inv_blocks.to(x.dtype))
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, online_block={self.online_block_size}, "
            f"bits={self.quant_meta.get('bits')}, method={self.quant_meta.get('method')}"
        )


# ---------------------------------------------------------------------------------- save / load

def quantized_layout(model: nn.Module) -> Dict[str, Dict[str, object]]:
    """Shapes needed to rebuild every ``CafeQLinear`` in ``model`` before loading a state dict."""
    layout = {}
    for name, mod in model.named_modules():
        if isinstance(mod, CafeQLinear):
            layout[name] = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": mod.bias is not None,
                "inv_block_shape": None if mod.inv_blocks is None else list(mod.inv_blocks.shape[:2]),
                "meta": {k: v for k, v in mod.quant_meta.items() if isinstance(v, (int, float, str, bool, type(None)))},
            }
    return layout


def save_quantized(model: nn.Module, path: Union[str, Path]) -> None:
    """Write ``<path>/quantized_state.pt`` and ``<path>/layout.json``. Do not commit these files."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "quantized_state.pt")
    (path / "layout.json").write_text(json.dumps(quantized_layout(model), indent=2))


def load_quantized(model: nn.Module, path: Union[str, Path], map_location="cpu") -> nn.Module:
    """Replace the layout's modules in a freshly built ``model`` with ``CafeQLinear`` and load weights."""
    path = Path(path)
    layout = json.loads((path / "layout.json").read_text())
    ref_param = next(model.parameters())
    for name, spec in layout.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        shape = spec["inv_block_shape"]
        mod = CafeQLinear(
            spec["in_features"],
            spec["out_features"],
            spec["bias"],
            None if shape is None else tuple(shape),
            dtype=ref_param.dtype,
            device=ref_param.device,
        )
        mod.quant_meta = dict(spec.get("meta", {}))
        setattr(parent, child, mod)
    state = torch.load(path / "quantized_state.pt", map_location=map_location)
    model.load_state_dict(state)
    return model
