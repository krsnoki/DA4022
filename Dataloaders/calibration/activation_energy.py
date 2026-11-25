"""Collect per-input-column activation energy for decoder linear layers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn


def eligible_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    """Return 2-D linear weights inside decoder layers, keyed by module path.

    Embeddings, the language-model head, and linear modules outside decoder
    layers are excluded. For Qwen2.5 this includes attention and MLP projections.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if name
        and "layers" in name.split(".")
        and isinstance(module, nn.Linear)
        and module.weight.ndim == 2
    }


def collect_activation_column_energy(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Sum valid-token input squares for each eligible linear layer.

    Each batch must provide ``input_ids`` and a 2-D ``attention_mask``. The
    mask excludes padding. Hooks retain only FP32 column sums on CPU; activation
    tensors are used during the forward pass and are not kept after each hook.
    The model is put in evaluation mode and cache generation is disabled.
    """
    layers = eligible_linear_modules(model)
    if not layers:
        raise ValueError("model has no eligible decoder-layer nn.Linear modules")

    model.eval()
    column_energy = {
        name: torch.zeros(module.in_features, dtype=torch.float32, device="cpu")
        for name, module in layers.items()
    }
    current_mask: list[torch.Tensor | None] = [None]

    def make_hook(name: str):
        def collect_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{name} did not receive a tensor as its first input")
            activations = inputs[0].detach()
            mask = current_mask[0]
            if mask is None:
                raise RuntimeError("set an attention mask before each model forward")
            if activations.shape[-1] != layers[name].in_features:
                raise ValueError(f"{name} input width does not match its linear weight")

            rows = activations.reshape(-1, activations.shape[-1])
            valid = mask.reshape(-1).to(device=activations.device, dtype=torch.bool)
            if valid.numel() != rows.shape[0]:
                raise ValueError(
                    f"{name} input rows ({rows.shape[0]}) do not match mask tokens "
                    f"({valid.numel()})"
                )
            sums = rows[valid].to(dtype=torch.float32).square().sum(dim=0)
            column_energy[name].add_(sums.to(device="cpu"))

        return collect_input

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in layers.items()
    ]
    try:
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = torch.device("cpu")

        with torch.inference_mode():
            for batch in batches:
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                if input_ids is None or attention_mask is None:
                    raise ValueError("each batch must include input_ids and attention_mask")
                if attention_mask.ndim != 2 or input_ids.shape != attention_mask.shape:
                    raise ValueError("input_ids and attention_mask must have matching [batch, seq] shapes")

                input_ids = input_ids.to(device=input_device)
                attention_mask = attention_mask.to(device=input_device)
                current_mask[0] = attention_mask
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return column_energy
