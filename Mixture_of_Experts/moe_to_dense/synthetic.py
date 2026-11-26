"""Tiny routed language model for honest, hardware-feasible end-to-end validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .concatenation import DenseSwiGLU
from .merging import ExpertWeights


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def export_weights(self) -> ExpertWeights:
        return ExpertWeights(
            self.gate_proj.weight.detach().clone(),
            self.up_proj.weight.detach().clone(),
            self.down_proj.weight.detach().clone(),
        )


class TinyMoELayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("invalid top_k")
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.top_k = top_k

    def all_expert_outputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(hidden_states) for expert in self.experts], dim=-2)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(self.router(hidden_states), dim=-1)
        selected = torch.topk(probabilities, self.top_k, dim=-1, sorted=True).indices
        return probabilities, selected

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        probabilities, selected = self.route(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        selected = selected.reshape(-1, self.top_k)
        weights = probabilities.reshape(flat.shape[0], -1).gather(1, selected)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = (selected == expert_id).nonzero(as_tuple=True)
            if tokens.numel():
                contribution = expert(flat[tokens]) * weights[tokens, slots, None]
                output = output.index_add(0, tokens, contribution)
        return output.reshape_as(hidden_states)


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TinyMoELM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        intermediate_size: int = 16,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = TinyMoELayer(hidden_size, intermediate_size, num_experts, top_k)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> LMOutput:
        hidden = self.embed(input_ids)
        hidden = hidden + self.ffn(self.norm1(hidden))
        logits = self.lm_head(self.norm2(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1))
        return LMOutput(logits, loss)

    @torch.no_grad()
    def calibration_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.norm1(self.embed(input_ids))
        probabilities, selected = self.ffn.route(hidden)
        outputs = self.ffn.all_expert_outputs(hidden)
        return probabilities, outputs, selected


class TinyDenseLM(nn.Module):
    def __init__(self, teacher: TinyMoELM, dense_ffn: DenseSwiGLU):
        super().__init__()
        self.vocab_size = teacher.vocab_size
        self.hidden_size = teacher.hidden_size
        self.embed = deepcopy(teacher.embed)
        self.norm1 = deepcopy(teacher.norm1)
        self.ffn = dense_ffn
        self.norm2 = deepcopy(teacher.norm2)
        self.lm_head = deepcopy(teacher.lm_head)
        # A teacher may already have been frozen by a prior controlled run.
        # Every newly constructed student must nevertheless be fully trainable.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> LMOutput:
        hidden = self.embed(input_ids)
        hidden = hidden + self.ffn(self.norm1(hidden))
        logits = self.lm_head(self.norm2(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1))
        return LMOutput(logits, loss)


def patterned_tokens(samples: int, length: int, vocab_size: int, seed: int) -> torch.Tensor:
    """Deterministic synthetic next-token task with varied offsets."""
    generator = torch.Generator().manual_seed(seed)
    start = torch.randint(0, vocab_size, (samples, 1), generator=generator)
    step = torch.randint(1, 5, (samples, 1), generator=generator)
    positions = torch.arange(length).unsqueeze(0)
    return (start + step * positions) % vocab_size
