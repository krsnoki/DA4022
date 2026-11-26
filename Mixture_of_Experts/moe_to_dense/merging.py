"""Pure pruning, round-robin grouping, and score-weighted expert merging."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def validate(self) -> None:
        if self.gate.shape != self.up.shape or self.gate.ndim != 2:
            raise ValueError("gate and up projections must share [intermediate, hidden] shape")
        if self.down.shape != (self.gate.shape[1], self.gate.shape[0]):
            raise ValueError("down projection must have [hidden, intermediate] shape")

    def clone(self) -> "ExpertWeights":
        return ExpertWeights(self.gate.clone(), self.up.clone(), self.down.clone())


def round_robin_groups(selected: list[int], group_count: int) -> list[list[int]]:
    if not 1 <= group_count <= len(selected):
        raise ValueError("group_count must be non-zero and no larger than selected experts")
    groups = [[] for _ in range(group_count)]
    for position, expert_index in enumerate(selected):
        groups[position % group_count].append(expert_index)
    return groups


def merge_groups(
    experts: list[ExpertWeights],
    groups: list[list[int]],
    scores: torch.Tensor,
) -> tuple[list[ExpertWeights], torch.Tensor]:
    """Merge every group using within-group normalized importance weights."""
    if any(not group for group in groups):
        raise ValueError("groups must not be empty")
    for expert in experts:
        expert.validate()
    merged: list[ExpertWeights] = []
    group_scores: list[torch.Tensor] = []
    for group in groups:
        weights = scores[group].to(dtype=torch.float64).clamp_min(0)
        weights = weights / weights.sum() if weights.sum() > 0 else torch.full_like(weights, 1 / len(group))
        fields: dict[str, torch.Tensor] = {}
        for name in ("gate", "up", "down"):
            tensors = torch.stack([getattr(experts[index], name) for index in group])
            view = (len(group),) + (1,) * (tensors.ndim - 1)
            fields[name] = (tensors * weights.to(tensors).view(view)).sum(dim=0)
        merged.append(ExpertWeights(**fields))
        group_scores.append(scores[group].sum())
    return merged, torch.stack(group_scores)
