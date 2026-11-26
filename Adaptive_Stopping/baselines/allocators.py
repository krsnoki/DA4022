"""Deterministic baseline budget allocators (Uniform, Easy2Hard, Hard2Easy)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from Adaptive_Stopping.methods.dipa.allocator import BudgetAllocator
from Adaptive_Stopping.methods.dipa.types import AllocationConfig, ArmState


class UniformAllocator(BudgetAllocator):
    """Round-robin uniform allocator without early stopping by default."""

    method_name: str = "uniform"
    deterministic: bool = True

    def __init__(
        self,
        config: AllocationConfig,
        eliminate_solved: bool = False,
    ) -> None:
        super().__init__(config=config, eliminate_solved=eliminate_solved)
        self._step: int = 0

    def reset(self, request_metadata: dict[str, Any] | None = None) -> None:
        super().reset(request_metadata)
        self._step = 0

    def select_arm(
        self,
        active_arms: Sequence[ArmState],
        rng: np.random.Generator,
    ) -> ArmState:
        arm = active_arms[self._step % len(active_arms)]
        self._step += 1
        return arm


class Easy2HardAllocator(BudgetAllocator):
    """Greedy allocator selecting the active arm with lowest estimated difficulty."""

    method_name: str = "easy2hard"
    deterministic: bool = True

    def select_arm(
        self,
        active_arms: Sequence[ArmState],
        rng: np.random.Generator,
    ) -> ArmState:
        return min(active_arms, key=lambda arm: arm.difficulty)


class Hard2EasyAllocator(BudgetAllocator):
    """Greedy allocator selecting the active arm with highest estimated difficulty."""

    method_name: str = "hard2easy"
    deterministic: bool = True

    def select_arm(
        self,
        active_arms: Sequence[ArmState],
        rng: np.random.Generator,
    ) -> ArmState:
        return max(active_arms, key=lambda arm: arm.difficulty)
