"""Hand-off from calibration (Member 1) to the AAAC fit (Member 2): one layer at a time."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .activation_energy import eligible_linear_modules


@dataclass(frozen=True)
class QuantizationTarget:
    """One eligible linear layer, in the layout the AAAC fit expects.

    ``weight`` is a float32 *copy* of ``module.weight`` with shape ``[out, in]``;
    ``importance`` is the float32 column energy ``I[k] = sum_t X[t, k]^2`` with
    shape ``[in]``. ``module`` is the live layer (use it to write back results and
    to read ``bias``, which is never quantized).
    """

    name: str
    module: nn.Linear
    weight: torch.Tensor
    importance: torch.Tensor


def iter_quantization_targets(
    model: nn.Module,
    column_energy: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> Iterator[QuantizationTarget]:
    """Yield every eligible layer with its FP32 weight copy and column importance.

    Layers are produced lazily, so only one FP32 weight copy exists at a time. The
    energy dict must cover exactly the eligible layer set; any mismatch (for example
    energies collected from a different model) raises instead of silently skipping.
    """
    layers = eligible_linear_modules(model)
    if set(layers) != set(column_energy):
        missing = sorted(set(layers) - set(column_energy))
        extra = sorted(set(column_energy) - set(layers))
        raise ValueError(
            f"column_energy does not match eligible layers; missing={missing[:3]}, extra={extra[:3]}"
        )

    for name, module in layers.items():
        importance = column_energy[name].detach().to(device=device, dtype=torch.float32).clone()
        if importance.shape != (module.in_features,):
            raise ValueError(
                f"{name}: importance shape {tuple(importance.shape)} != ({module.in_features},)"
            )
        if not torch.isfinite(importance).all():
            raise ValueError(f"{name}: importance contains non-finite values")
        weight = module.weight.detach().to(device=device, dtype=torch.float32).clone()
        yield QuantizationTarget(name=name, module=module, weight=weight, importance=importance)
