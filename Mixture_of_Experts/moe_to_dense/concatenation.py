"""Block-concatenate SwiGLU experts into a standard dense FFN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .merging import ExpertWeights


class DenseSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def scaling_coefficients(group_scores: torch.Tensor, method: str) -> torch.Tensor:
    count = group_scores.numel()
    if count == 0:
        raise ValueError("at least one group is required")
    if method == "uniform":
        return torch.full_like(group_scores, 1.0 / count)
    if method == "proportional":
        scores = group_scores.clamp_min(0)
        return scores / scores.sum() if scores.sum() > 0 else torch.full_like(scores, 1.0 / count)
    raise ValueError(f"unknown scaling method: {method}")


def concatenate_experts(
    experts: list[ExpertWeights],
    group_scores: torch.Tensor,
    scaling: str = "uniform",
) -> DenseSwiGLU:
    if not experts:
        raise ValueError("at least one expert is required")
    for expert in experts:
        expert.validate()
    hidden = experts[0].gate.shape[1]
    intermediate = sum(expert.gate.shape[0] for expert in experts)
    dense = DenseSwiGLU(hidden, intermediate).to(device=experts[0].gate.device, dtype=experts[0].gate.dtype)
    alpha = scaling_coefficients(group_scores.to(experts[0].gate), scaling)
    with torch.no_grad():
        dense.gate_proj.weight.copy_(torch.cat([expert.gate for expert in experts], dim=0))
        dense.up_proj.weight.copy_(torch.cat([expert.up for expert in experts], dim=0))
        dense.down_proj.weight.copy_(
            torch.cat([coefficient * expert.down for coefficient, expert in zip(alpha, experts)], dim=1)
        )
    return dense
