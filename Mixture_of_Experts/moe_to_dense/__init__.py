"""Modular MoE-to-dense conversion and distillation utilities."""

from .calibration import CalibrationStats, collect_calibration
from .concatenation import DenseSwiGLU, concatenate_experts
from .distillation import DistillationConfig, forward_kl, train_distillation
from .merging import ExpertWeights, merge_groups, round_robin_groups
from .scoring import ExpertScorer, greedy_logdet_select

__all__ = [
    "CalibrationStats",
    "DenseSwiGLU",
    "DistillationConfig",
    "ExpertScorer",
    "ExpertWeights",
    "collect_calibration",
    "concatenate_experts",
    "forward_kl",
    "greedy_logdet_select",
    "merge_groups",
    "round_robin_groups",
    "train_distillation",
]
