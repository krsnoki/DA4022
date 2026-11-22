"""Core data structures, configs, and contracts for DIPA budget allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol

import numpy as np


@dataclass
class ProblemInstance:
    """Single verifiable reasoning instance in an allocation batch."""

    instance_id: str
    prompt: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationAttempt:
    """Single candidate response produced for a problem instance."""

    text: str
    extracted_answer: str | None = None
    num_tokens: int = 0
    is_correct: bool = False
    latency_ms: float = 0.0
    token_ids: list[int] | None = None


@dataclass
class ArmState:
    """Bandit arm state tracking difficulty and accumulated attempts."""

    instance: ProblemInstance
    difficulty: float
    solved: bool = False
    attempts: list[GenerationAttempt] = field(default_factory=list)


@dataclass(frozen=True)
class AllocationConfig:
    """Hyperparameters controlling DIPA and baseline budget allocators."""

    total_budget: int
    input_proxy: str = "mql"
    gen_proxy: str = "mgl"
    lambda_mode: str = "dynamic"
    lambda_value: float = 1.0
    min_difficulty: float = 1e-6
    seed: int = 42

    def __post_init__(self) -> None:
        if self.total_budget < 0:
            raise ValueError(f"total_budget must be >= 0, got {self.total_budget}")
        if self.lambda_mode not in {"fixed", "dynamic"}:
            raise ValueError(
                f"lambda_mode must be 'fixed' or 'dynamic', got {self.lambda_mode!r}"
            )
        if not math.isfinite(self.lambda_value) or self.lambda_value <= 0:
            raise ValueError(f"lambda_value must be > 0, got {self.lambda_value}")
        if not math.isfinite(self.min_difficulty) or self.min_difficulty <= 0:
            raise ValueError(f"min_difficulty must be > 0, got {self.min_difficulty}")


@dataclass
class AllocationReport:
    """Summary of an allocation run over a batch of problem instances."""

    method: str
    total_budget: int
    budget_used: int
    num_instances: int
    num_solved: int
    coverage: float
    per_instance_allocations: dict[str, int]
    solved_ids: list[str]
    unsolved_ids: list[str]
    total_tokens: int
    total_latency_ms: float
    config: dict[str, Any]


class StoppingPolicy:
    """Shared stopping-policy contract for Adaptive_Stopping methods."""

    def reset(self, request_metadata: dict[str, Any] | None = None) -> None:
        """Reset internal policy state for a new run or request batch."""

    def should_stop(self, state: Any, metrics: dict[str, Any], budget: int) -> bool:
        """Return whether computation should stop given current state and budget."""
        raise NotImplementedError

    def report(self) -> dict[str, Any]:
        """Return summary statistics from the most recent execution."""
        raise NotImplementedError


class GeneratorFn(Protocol):
    """Protocol for single-attempt candidate generation callbacks."""

    def __call__(
        self,
        instance: ProblemInstance,
        attempt_idx: int,
        rng: np.random.Generator,
    ) -> GenerationAttempt: ...


class VerifierFn(Protocol):
    """Protocol for answer verification callbacks returning (is_correct, extracted)."""

    def __call__(
        self,
        generation_text: str,
        ground_truth: str,
    ) -> tuple[bool, str | None]: ...
