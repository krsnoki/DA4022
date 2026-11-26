"""Selection, optional merging, and dense construction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .calibration import CalibrationStats
from .concatenation import DenseSwiGLU, concatenate_experts
from .merging import ExpertWeights, merge_groups, round_robin_groups
from .scoring import ExpertScorer


@dataclass
class ConversionResult:
    dense_ffn: DenseSwiGLU
    selected_experts: list[int]
    groups: list[list[int]]
    scores: torch.Tensor


def convert_experts(
    experts: list[ExpertWeights],
    stats: CalibrationStats,
    *,
    scoring: str,
    selected_count: int,
    group_count: int,
    scaling: str = "uniform",
    lambda_reg: float | None = None,
) -> ConversionResult:
    if len(experts) != stats.num_experts:
        raise ValueError("expert weights and calibration statistics disagree")
    scorer = ExpertScorer(stats)
    selected = scorer.select(scoring, selected_count, lambda_reg)
    scores = scorer.scores(scoring)
    groups = round_robin_groups(selected, group_count)
    merged, group_scores = merge_groups(experts, groups, scores)
    dense = concatenate_experts(merged, group_scores, scaling)
    return ConversionResult(dense, selected, groups, scores)
