"""
KVgradVariantsPress: kvpress press that evicts KV pairs with any of the four KVgrad score variants.

It reuses the replay, hook and eviction machinery of kvpress's KVgradPress (kvpress 0.5.5, Apache-2.0)
and replaces only the scoring step with `scores.score_variants`. With `variant="bound_max"` it computes the
same scores as kvpress's KVgradPress (checked in Attention/tests/test_kvgrad_variants.py).

One replay (forward + backward) produces all four variants, so the press can optionally keep them in a
shared `score_cache`: a later call on the same context, with any variant or compression ratio, skips the
replay and only redoes the cheap eviction step. This makes the score ablation affordable on a T4, but the
latency of a call that hits the cache does not include scoring; `last_scoring_cached` reports which case
the last call was.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import torch
from kvpress import KVgradPress
from transformers import Cache, PreTrainedModel, PreTrainedTokenizer

from .scores import VARIANTS, score_variants

logger = logging.getLogger(__name__)


@dataclass
class KVgradVariantsPress(KVgradPress):
    """
    Parameters
    ----------
    variant : str, default="bound_max"
        One of "taylor_sum" (Eq. 6), "taylor_max", "bound_sum" (Eq. 7 summed), "bound_max" (Eq. 8, KVgrad).
    score_cache : dict or None
        Optional dict shared between presses. Maps a hash of the context token ids to the CPU scores of all
        variants, so the same context is replayed only once.
    compression_ratio, chunk_size, layerwise, n_sink
        See kvpress's KVgradPress / KVzipPress.
    """

    variant: str = "bound_max"
    score_cache: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{self.variant}'; choose from {VARIANTS}")
        super().__post_init__()
        self.last_scoring_cached: bool | None = None

    def _cache_key(self, model: PreTrainedModel) -> str | None:
        if self.score_cache is None:
            return None
        digest = hashlib.sha1(self._context_ids.detach().cpu().numpy().tobytes()).hexdigest()
        return f"{model.config.name_or_path}|{self.chunk_size}|{self.n_sink}|{digest}"

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        key = self._cache_key(model)
        if key is not None and key in self.score_cache:
            self.context_length = self._context_ids.shape[1]
            cached = self.score_cache[key][self.variant]
            self.score_val = cached.to(device=model.device, dtype=model.dtype).clone()
            self.last_scoring_cached = True
            self.compress_post(model)
            return

        self.last_scoring_cached = False
        self._variant_scores: dict[str, torch.Tensor] = {}
        super()._perform_kvzip_compression(model, tokenizer)  # replay, backward, _update_scores, compress_post

    def _update_scores(self, model: PreTrainedModel, cache: Cache, layer_idx: int, captures: dict):
        module = model.model.layers[layer_idx].self_attn
        attention_output = captures["attention_outputs"][layer_idx]
        assert attention_output.grad is not None, f"No gradient captured for layer {layer_idx}"

        variants = VARIANTS if self.score_cache is not None else (self.variant,)
        scores = score_variants(
            attn_weights=captures["attn_weights"][layer_idx],
            values=cache.layers[layer_idx].values[:, :, self.start_idx : self.end_idx],
            grad=attention_output.grad,
            o_proj_weight=module.o_proj.weight,
            variants=variants,
        )
        for name, value in scores.items():
            if name not in self._variant_scores:
                self._variant_scores[name] = torch.zeros_like(self.score_val)
            self._variant_scores[name][layer_idx][..., self.start_idx : self.end_idx] = value
        self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores[self.variant]

    def compress_post(self, model: PreTrainedModel):
        # Store the raw scores before compress_post overwrites the attention-sink scores in place
        if self.last_scoring_cached is False and self.score_cache is not None:
            key = self._cache_key(model)
            self.score_cache[key] = {name: s.detach().cpu().clone() for name, s in self._variant_scores.items()}
        super().compress_post(model)
