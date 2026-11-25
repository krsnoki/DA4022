"""TERMINATOR probe: one transformer layer plus a prediction head.

Matches the released checkpoint layout (acnagle/Terminator-Qwen3-8B):
- input: the output of the base model's last decoder layer, before the final
  RMSNorm (captured with a forward hook on model.layers[-1]);
- one extra decoder layer, initialised as a copy of that last layer, run over
  all positions (prompt and thinking) with its own rotary embedding;
- head: RMSNorm + Linear(hidden, 1) giving one logit per position.

Run the probe in float32 even when the base model is in fp16. In fp16 the
copied layer's MLP overflows at some positions (the pre-norm residual stream
of Qwen3 is large), which gives inf, then NaN in the head's RMSNorm.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class TerminatorHead(nn.Module):
    """RMSNorm followed by a linear layer to one logit (the paper's 1-layer head)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.network = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(self.norm(h)).squeeze(-1)


class TerminatorProbe(nn.Module):
    """Probe over the base model's final (pre-norm) hidden states.

    forward(hidden[B, T, H]) -> logits[B, T]. Causal: the logit at position t
    depends only on hidden states at positions <= t.
    """

    def __init__(self, base_model: nn.Module, copy_last_layer: bool = True):
        super().__init__()
        inner = base_model.model if hasattr(base_model, "model") else base_model
        config = inner.config
        layer = copy.deepcopy(inner.layers[-1])
        if not copy_last_layer:
            for m in layer.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        # Own cache slot 0, and sdpa so a None mask means causal attention.
        layer.self_attn.layer_idx = 0
        layer.self_attn.config = copy.deepcopy(layer.self_attn.config)
        layer.self_attn.config._attn_implementation = "sdpa"
        if hasattr(layer, "attention_type"):
            layer.attention_type = "full_attention"
        self.layer = layer
        self.rotary_emb = copy.deepcopy(inner.rotary_emb)
        self.head = TerminatorHead(config.hidden_size, getattr(config, "rms_norm_eps", 1e-6))
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_values=None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """hidden: [B, T, H]. With past_key_values (a transformers Cache), the
        layer attends to cached positions and appends the new ones."""
        B, T, _ = hidden.shape
        pos = torch.arange(position_offset, position_offset + T, device=hidden.device)
        position_ids = pos.unsqueeze(0).expand(B, -1)
        pe = self.rotary_emb(hidden, position_ids)
        out = self.layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            cache_position=pos,
            position_embeddings=pe,
        )
        if isinstance(out, tuple):
            out = out[0]
        return self.head(out)

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}


class LastLayerHook:
    """Captures the output of the base model's last decoder layer."""

    def __init__(self, base_model: nn.Module):
        inner = base_model.model if hasattr(base_model, "model") else base_model
        self.hidden: Optional[torch.Tensor] = None
        self._h = inner.layers[-1].register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        self.hidden = (output[0] if isinstance(output, tuple) else output).detach()

    def remove(self):
        self._h.remove()


def probe_input(hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cast base hidden states to float32 for the probe. Non-finite values
    (possible when the base runs in fp16) are replaced by large finite ones;
    returns the cast tensor and the number of positions that had to be fixed."""
    h = hidden.float()
    bad = ~torch.isfinite(h)
    n_bad = int(bad.any(-1).sum()) if bad.any() else 0
    if n_bad:
        h = torch.nan_to_num(h, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return h, n_bad


@torch.no_grad()
def base_hidden_states(base_model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Pre-norm output of the last decoder layer for input_ids [B, T]. Runs the
    decoder stack only (no LM head) to save memory on long traces."""
    inner = base_model.model if hasattr(base_model, "model") else base_model
    hook = LastLayerHook(base_model)
    try:
        inner(input_ids=input_ids, use_cache=False)
        return hook.hidden
    finally:
        hook.remove()
