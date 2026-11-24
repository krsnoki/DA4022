"""Simulated ("fake") uniform quantization of weights and activations.

This is a shared primitive for the Quantization_and_Pruning module.  Tensors
are quantized to ``bits`` integers and immediately de-quantized, so every
matmul still runs in floating point.  It therefore measures the *accuracy*
effect of low-bit formats (W4A4, W6A6, W8A8, ...) but gives **no speed-up and
no memory saving**; never report it as an INT4 kernel (see README_qp.md).

Conventions (match common LLM PTQ practice and Pandey et al. App. B.4):

* weights: per-output-channel (one scale per row of ``[out, in]``), optionally
  per group of ``group_size`` input columns; symmetric by default;
* activations: per-token dynamic (one scale per row of ``[..., hidden]``),
  asymmetric by default;
* optional query/key quantization inside attention (``qk_bits``), per token and
  head, which pushes rounding error straight into the softmax logits.  Requires
  the ``jacobian_noise`` attention implementation (it reads ``module._fq_qk_bits``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


# --------------------------------------------------------------------- core ops
def quant_params(
    x: torch.Tensor, bits: int, symmetric: bool = True, clip_ratio: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale and zero-point for every row of ``x`` (statistics over the last dim)."""
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    x = x.float()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
        amax = x.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scale = (amax / qmax).clamp_min(1e-8)
        zero = torch.zeros_like(scale)
    else:
        qmax = 2**bits - 1
        xmax = x.amax(dim=-1, keepdim=True).clamp_min(0) * clip_ratio
        xmin = x.amin(dim=-1, keepdim=True).clamp_max(0) * clip_ratio
        scale = ((xmax - xmin) / qmax).clamp_min(1e-8)
        zero = torch.round(-xmin / scale)
    return scale, zero


def quant_dequant(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, symmetric: bool = True
) -> torch.Tensor:
    """Round ``x`` to the integer grid defined by ``scale``/``zero`` and map back."""
    xf = x.float()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
        q = torch.clamp(torch.round(xf / scale), -qmax - 1, qmax)
        out = q * scale
    else:
        qmax = 2**bits - 1
        q = torch.clamp(torch.round(xf / scale) + zero, 0, qmax)
        out = (q - zero) * scale
    return out.to(x.dtype)


def fake_quantize(
    x: torch.Tensor,
    bits: Optional[int],
    symmetric: bool = True,
    group_size: Optional[int] = None,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize-dequantize with one scale per row of the last dim (or per group)."""
    if bits is None or bits >= 16:
        return x
    if group_size:
        *lead, n = x.shape
        if n % group_size != 0:
            raise ValueError(f"last dim {n} not divisible by group_size {group_size}")
        xg = x.reshape(*lead, n // group_size, group_size)
        s, z = quant_params(xg, bits, symmetric, clip_ratio)
        return quant_dequant(xg, s, z, bits, symmetric).reshape(x.shape)
    s, z = quant_params(x, bits, symmetric, clip_ratio)
    return quant_dequant(x, s, z, bits, symmetric)


# ------------------------------------------------------------------- config
@dataclass
class FakeQuantConfig:
    w_bits: Optional[int] = 4
    a_bits: Optional[int] = 4
    qk_bits: Optional[int] = None
    w_symmetric: bool = True
    a_symmetric: bool = False
    w_group_size: Optional[int] = None
    a_group_size: Optional[int] = None
    w_clip_ratio: float = 1.0
    a_clip_ratio: float = 1.0
    weight_method: str = "rtn"  # "rtn" | "gptq" (gptq is applied by quantize_model)
    include: str = r"\.layers\.\d+\."  # regex on module names; transformer blocks only
    exclude: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FakeQuantConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown FakeQuantConfig keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_tag(cls, tag: str, **overrides) -> "FakeQuantConfig":
        """Parse tags like ``W4A4``, ``W4A8``, ``W4A16``, ``W4A4QK4``, ``FP16``."""
        tag = tag.upper()
        if tag in ("FP16", "BF16", "FP32", "FP"):
            return cls(w_bits=None, a_bits=None, qk_bits=None, **overrides)
        m = re.fullmatch(r"W(\d+)A(\d+)(?:QK(\d+))?", tag)
        if not m:
            raise ValueError(f"cannot parse quantization tag {tag!r}")
        w, a, qk = (int(g) if g else None for g in m.groups())
        return cls(w_bits=w, a_bits=None if a >= 16 else a, qk_bits=qk, **overrides)

    @property
    def tag(self) -> str:
        if self.w_bits is None and self.a_bits is None and self.qk_bits is None:
            return "FP16"
        t = f"W{self.w_bits or 16}A{self.a_bits or 16}"
        return t + (f"QK{self.qk_bits}" if self.qk_bits else "")

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ modules
class FakeQuantLinear(nn.Module):
    """Wraps an ``nn.Linear``: weights quantized once, activations per call."""

    def __init__(self, linear: nn.Linear, cfg: FakeQuantConfig, quantize_weight: bool = True):
        super().__init__()
        self.linear = linear
        self.a_bits = cfg.a_bits
        self.a_symmetric = cfg.a_symmetric
        self.a_group_size = cfg.a_group_size
        self.a_clip_ratio = cfg.a_clip_ratio
        self.w_bits = cfg.w_bits
        if quantize_weight and cfg.w_bits is not None:
            with torch.no_grad():
                w = linear.weight.data
                linear.weight.data.copy_(
                    fake_quantize(w, cfg.w_bits, cfg.w_symmetric, cfg.w_group_size, cfg.w_clip_ratio)
                )

    @property
    def weight(self) -> torch.Tensor:  # keeps `module.weight` working for callers
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quantize(x, self.a_bits, self.a_symmetric, self.a_group_size, self.a_clip_ratio)
        return self.linear(x)

    def extra_repr(self) -> str:
        return f"w_bits={self.w_bits}, a_bits={self.a_bits}"


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def select_linears(model: nn.Module, cfg: FakeQuantConfig) -> Dict[str, nn.Linear]:
    inc = re.compile(cfg.include) if cfg.include else None
    exc = re.compile(cfg.exclude) if cfg.exclude else None
    out = {}
    for name, mod in model.named_modules():
        if type(mod) is not nn.Linear and not isinstance(mod, nn.Linear):
            continue
        if isinstance(mod, FakeQuantLinear):
            continue
        full = "." + name + "."
        if inc and not inc.search(full):
            continue
        if exc and exc.search(full):
            continue
        out[name] = mod
    return out


def apply_fake_quant(
    model: nn.Module, cfg: FakeQuantConfig, weights_already_quantized: bool = False
) -> dict:
    """In-place: wrap selected linears and enable q/k quantization if requested.

    ``weights_already_quantized=True`` is used after GPTQ so that weights are
    not re-quantized with RTN.
    Returns a small report dict.
    """
    linears = select_linears(model, cfg)
    for name, lin in linears.items():
        _set_module(model, name, FakeQuantLinear(lin, cfg, quantize_weight=not weights_already_quantized))
    n_attn = 0
    if cfg.qk_bits:
        inc = re.compile(cfg.include) if cfg.include else None
        for name, mod in model.named_modules():
            if hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and (not inc or inc.search("." + name + ".")):
                mod._fq_qk_bits = cfg.qk_bits
                n_attn += 1
    return {"quant_tag": cfg.tag, "num_linear_quantized": len(linears), "num_attention_qk_quantized": n_attn}


def fake_quant_qk(module: nn.Module, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Called from the attention function: per-token/per-head symmetric q/k quantization."""
    bits = getattr(module, "_fq_qk_bits", None)
    if not bits:
        return query, key
    return fake_quantize(query, bits, symmetric=True), fake_quantize(key, bits, symmetric=True)


def model_size_bytes(model: nn.Module, w_bits: Optional[int] = None, cfg: Optional[FakeQuantConfig] = None) -> dict:
    """Parameter count and *theoretical* packed weight size (not measured on disk)."""
    n_total = sum(p.numel() for p in model.parameters())
    if cfg is None or cfg.w_bits is None:
        return {"num_parameters": n_total}
    n_q = sum(m.linear.weight.numel() for m in model.modules() if isinstance(m, FakeQuantLinear))
    bytes_theory = (n_total - n_q) * 2 + n_q * cfg.w_bits / 8
    return {"num_parameters": n_total, "num_quantized_weights": n_q, "theoretical_weight_mb": bytes_theory / 2**20}
