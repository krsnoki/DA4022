"""Apply XShare routing to a Hugging Face MoE model.

The selection logic lives in ``Mixture_of_Experts.routing.xshare`` and knows
nothing about any model. This file is the thin layer that connects it to
Hugging Face models whose MoE block calls a router module returning
``(router_logits, top_k_weights, top_k_indices)``. That is the layout of
``OlmoeSparseMoeBlock`` in transformers 5.x, where ``layer.mlp.gate`` is the
router.

:func:`install_xshare` replaces each layer's router with :class:`XShareRouter`.
The wrapper keeps the original router and calls it unchanged whenever no
pruning applies, so the unpruned baseline runs the model's own code.

Pruning applies only at decode steps (one new token per sequence). Prefill
passes, where each sequence contributes many tokens, are left unpruned, as in
the paper.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from Mixture_of_Experts.routing.xshare import XShareConfig, xshare_route


class XShareController:
    """Shared state between the decode loop and the router wrappers.

    The decode loop sets :attr:`valid_rows` before each decode step. The
    wrappers read it, and add their counts to the statistics.

    Attributes:
        config: the selection settings, or ``None`` for the unpruned baseline.
        valid_rows: ``[batch]`` bool, true for sequences that are still
            generating. Finished sequences are fed padding and must not vote.
        decode_only: prune only when the sequence length of the pass is 1.
        capture: when set to a list, every decode-step call appends its raw
            router logits (CPU, float16), for offline simulations.
    """

    def __init__(self, config: Optional[XShareConfig], n_layers: int, decode_only: bool = True):
        self.config = config
        self.decode_only = decode_only
        self.n_layers = n_layers
        self.valid_rows: Optional[Tensor] = None
        self.sequence_length: int = 0
        self.capture: Optional[list] = None
        self.reset_stats()

    def reset_stats(self) -> None:
        # Sums are kept as tensors on the model's device, so that counting
        # does not force a GPU synchronisation in the middle of a step.
        self._calls = [0] * self.n_layers
        self._kept = [None] * self.n_layers
        self._activated = [None] * self.n_layers
        self._tokens = [None] * self.n_layers

    @property
    def is_decode_step(self) -> bool:
        return self.sequence_length == 1

    @property
    def should_prune(self) -> bool:
        if self.config is None or not self.config.prunes:
            return False
        return self.is_decode_step or not self.decode_only

    def record(self, layer: int, kept: Tensor, activated: Tensor, tokens: Tensor) -> None:
        self._calls[layer] += 1
        if self._kept[layer] is None:
            self._kept[layer], self._activated[layer], self._tokens[layer] = kept, activated, tokens
        else:
            self._kept[layer] = self._kept[layer] + kept
            self._activated[layer] = self._activated[layer] + activated
            self._tokens[layer] = self._tokens[layer] + tokens

    def stats(self) -> dict:
        """Means over all recorded decode-step calls, overall and per layer."""
        per_layer = []
        for layer in range(self.n_layers):
            calls = self._calls[layer]
            if calls == 0:
                continue
            per_layer.append(
                {
                    "layer": layer,
                    "calls": calls,
                    "mean_kept_experts": float(self._kept[layer]) / calls,
                    "mean_activated_experts": float(self._activated[layer]) / calls,
                    "mean_tokens": float(self._tokens[layer]) / calls,
                }
            )
        if not per_layer:
            return {"layer_calls": 0, "per_layer": []}
        n = len(per_layer)
        return {
            "layer_calls": sum(p["calls"] for p in per_layer),
            "mean_kept_experts": sum(p["mean_kept_experts"] for p in per_layer) / n,
            "mean_activated_experts": sum(p["mean_activated_experts"] for p in per_layer) / n,
            "mean_tokens_per_step": sum(p["mean_tokens"] for p in per_layer) / n,
            "per_layer": per_layer,
        }


class XShareRouter(nn.Module):
    """Drop-in replacement for a top-k router module.

    ``forward`` returns the same triple as the wrapped router:
    ``(router_logits, top_k_weights, top_k_indices)``.
    """

    def __init__(self, router: nn.Module, controller: XShareController, layer: int):
        super().__init__()
        self.router = router
        self.controller = controller
        self.layer = layer
        self.num_experts = router.num_experts
        self.top_k = router.top_k
        norm_topk_prob = getattr(router, "norm_topk_prob", False)
        # The wrapped router takes softmax over all experts, then top-k.
        self.gate_norm = "softmax_topk_renorm" if norm_topk_prob else "softmax_topk"

    def forward(self, hidden_states: Tensor):
        controller = self.controller
        if not controller.should_prune:
            router_logits, weights, indices = self.router(hidden_states)
            keep = None
        else:
            config = controller.config
            if config.top_k != self.top_k or config.gate_norm != self.gate_norm:
                raise ValueError(
                    f"config (top_k={config.top_k}, gate_norm={config.gate_norm}) does not match "
                    f"the router (top_k={self.top_k}, gate_norm={self.gate_norm})"
                )
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            router_logits = nn.functional.linear(flat, self.router.weight)
            valid = controller.valid_rows if controller.is_decode_step else None
            result = xshare_route(
                router_logits.float(), config, valid_mask=valid, with_diagnostics=False
            )
            keep = result.keep_mask
            weights = result.weights.to(router_logits.dtype)
            indices = result.indices

        if controller.is_decode_step:
            self._record(router_logits, indices, keep)
        return router_logits, weights, indices

    def _record(self, router_logits: Tensor, indices: Tensor, keep: Optional[Tensor]) -> None:
        controller = self.controller
        valid = controller.valid_rows
        if valid is None:
            valid = torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device)
        # Count per expert without mask indexing, which would stall the GPU.
        votes = valid.unsqueeze(-1).expand_as(indices).reshape(-1).to(torch.int32)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=indices.device)
        used = counts.index_add_(0, indices.reshape(-1), votes) > 0
        kept = used.new_tensor(self.num_experts, dtype=torch.long) if keep is None else keep.sum()
        controller.record(self.layer, kept, used.sum(), valid.sum())
        if controller.capture is not None:
            controller.capture.append(
                (self.layer, router_logits.detach().to("cpu", torch.float16), valid.cpu())
            )


def install_xshare(
    model: nn.Module, config: Optional[XShareConfig], decode_only: bool = True
) -> XShareController:
    """Wrap every MoE router of ``model`` and return the controller.

    Works for models that expose ``model.model.layers[i].mlp.gate`` as a
    top-k router (OLMoE in transformers 5.x).
    """
    layers = model.model.layers
    controller = XShareController(config, n_layers=len(layers), decode_only=decode_only)

    def remember_shape(module, args):
        controller.sequence_length = args[0].shape[1]

    for index, layer in enumerate(layers):
        block = layer.mlp
        if isinstance(block.gate, XShareRouter):
            raise RuntimeError("XShare is already installed on this model")
        block.gate = XShareRouter(block.gate, controller, index)
        block.register_forward_pre_hook(remember_shape)
    return controller
