"""Streaming calibration statistics for routed experts.

The collector intentionally stores O(E^2) statistics, not per-token activations.
Router probabilities are measured after softmax and only for selected experts.
The Gram matrix is accumulated from expert outputs before router weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import torch


@dataclass
class CalibrationStats:
    """Sufficient statistics for SF, CP, ACP and D-optimal selection."""

    token_count: int
    selection_count: torch.Tensor
    selected_probability_sum: torch.Tensor
    selected_activation_norm_sum: torch.Tensor
    output_gram: torch.Tensor

    @classmethod
    def empty(cls, num_experts: int, *, dtype: torch.dtype = torch.float64) -> "CalibrationStats":
        return cls(
            token_count=0,
            selection_count=torch.zeros(num_experts, dtype=dtype),
            selected_probability_sum=torch.zeros(num_experts, dtype=dtype),
            selected_activation_norm_sum=torch.zeros(num_experts, dtype=dtype),
            output_gram=torch.zeros((num_experts, num_experts), dtype=dtype),
        )

    @property
    def num_experts(self) -> int:
        return int(self.selection_count.numel())

    def update(
        self,
        router_probabilities: torch.Tensor,
        expert_outputs: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> None:
        """Update from flattened tokens.

        Args:
            router_probabilities: ``[tokens, experts]`` softmax probabilities.
            expert_outputs: ``[tokens, experts, hidden]`` expert outputs.
            selected_indices: ``[tokens, top_k]`` selected expert IDs.
        """
        if router_probabilities.ndim != 2 or expert_outputs.ndim != 3:
            raise ValueError("expected probabilities [T,E] and outputs [T,E,H]")
        tokens, experts = router_probabilities.shape
        if expert_outputs.shape[:2] != (tokens, experts):
            raise ValueError("expert output dimensions do not match router probabilities")
        if selected_indices.ndim != 2 or selected_indices.shape[0] != tokens:
            raise ValueError("selected_indices must have shape [T, top_k]")
        if experts != self.num_experts:
            raise ValueError("expert count does not match accumulator")

        probs = router_probabilities.detach().to(device="cpu", dtype=torch.float64)
        outputs = expert_outputs.detach().to(device="cpu", dtype=torch.float64)
        indices = selected_indices.detach().to(device="cpu", dtype=torch.long)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, indices, 1.0)
        norms = torch.linalg.vector_norm(outputs, dim=-1)

        self.token_count += tokens
        self.selection_count += mask.sum(dim=0)
        self.selected_probability_sum += (probs * mask).sum(dim=0)
        self.selected_activation_norm_sum += (norms * mask).sum(dim=0)
        # Each expert's calibration output is a vector over token and hidden axes.
        flattened = outputs.permute(1, 0, 2).reshape(experts, -1)
        self.output_gram += flattened @ flattened.T

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationStats":
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**data)


class CalibrationModel(Protocol):
    num_experts: int

    def calibration_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@torch.no_grad()
def collect_calibration(model: CalibrationModel, batches: Iterable[torch.Tensor]) -> CalibrationStats:
    """Collect streaming statistics from a model exposing ``calibration_forward``."""
    stats = CalibrationStats.empty(model.num_experts)
    for input_ids in batches:
        probabilities, expert_outputs, selected = model.calibration_forward(input_ids)
        stats.update(
            probabilities.reshape(-1, probabilities.shape[-1]),
            expert_outputs.reshape(-1, expert_outputs.shape[-2], expert_outputs.shape[-1]),
            selected.reshape(-1, selected.shape[-1]),
        )
    if stats.token_count == 0:
        raise ValueError("calibration received no tokens")
    return stats
