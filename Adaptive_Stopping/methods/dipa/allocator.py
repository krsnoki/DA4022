"""DIPA and base MAB budget allocators implementing the StoppingPolicy contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import math
from typing import Any

import numpy as np

from Adaptive_Stopping.methods.dipa.proxies import (
    compute_gen_difficulty,
    compute_input_difficulty,
)
from Adaptive_Stopping.methods.dipa.types import (
    AllocationConfig,
    AllocationReport,
    ArmState,
    GeneratorFn,
    ProblemInstance,
    StoppingPolicy,
    VerifierFn,
)
from Adaptive_Stopping.methods.dipa.verifier import verify_math_answer


class BudgetAllocator(StoppingPolicy):
    """Base multi-armed bandit budget allocator with arm elimination upon success."""

    method_name: str = "budget_allocator"
    deterministic: bool = False

    def __init__(
        self,
        config: AllocationConfig,
        eliminate_solved: bool = True,
        method_name: str | None = None,
        deterministic: bool | None = None,
    ) -> None:
        self.config = config
        self.eliminate_solved = eliminate_solved
        if method_name is not None:
            self.method_name = method_name
        if deterministic is not None:
            self.deterministic = deterministic
        self._last_report: AllocationReport | None = None
        self._budget_used: int = 0
        self._active_arms: int = 0
        self._num_instances: int = 0
        self._request_metadata: dict[str, Any] = {}

    def reset(self, request_metadata: dict[str, Any] | None = None) -> None:
        """Reset allocator tracking state before a batch run."""
        self._last_report = None
        self._budget_used = 0
        self._active_arms = 0
        self._num_instances = 0
        self._request_metadata = dict(request_metadata or {})

    def should_stop(self, state: Any, metrics: dict[str, Any], budget: int) -> bool:
        """Stop when budget is exhausted or no eligible unsolved arms remain."""
        if budget <= 0:
            return True
        budget_used = int(metrics.get("budget_used", self._budget_used))
        if budget_used >= budget:
            return True
        if "active_arms" in metrics:
            return int(metrics["active_arms"]) <= 0
        if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
            return not any(not getattr(arm, "solved", False) for arm in state)
        if isinstance(state, dict) and "active_arms" in state:
            return int(state["active_arms"]) <= 0
        if self._last_report is not None:
            return self._active_arms <= 0
        return False

    def report(self) -> dict[str, Any]:
        """Return the last allocation report as a dictionary."""
        if self._last_report is not None:
            data = asdict(self._last_report)
            data["deterministic"] = self.deterministic
            return data
        return {
            "method": self.method_name,
            "deterministic": self.deterministic,
            "total_budget": self.config.total_budget,
            "budget_used": self._budget_used,
        }

    def select_arm(
        self,
        active_arms: Sequence[ArmState],
        rng: np.random.Generator,
    ) -> ArmState:
        """Select the next arm to pull from ``active_arms``."""
        raise NotImplementedError

    def _build_report(
        self,
        arms: Sequence[ArmState],
        budget_used: int,
    ) -> AllocationReport:
        num_instances = len(arms)
        solved_ids = [arm.instance.instance_id for arm in arms if arm.solved]
        unsolved_ids = [arm.instance.instance_id for arm in arms if not arm.solved]
        num_solved = len(solved_ids)
        coverage = num_solved / num_instances if num_instances else 0.0
        return AllocationReport(
            method=self.method_name,
            total_budget=self.config.total_budget,
            budget_used=budget_used,
            num_instances=num_instances,
            num_solved=num_solved,
            coverage=coverage,
            per_instance_allocations={
                arm.instance.instance_id: len(arm.attempts) for arm in arms
            },
            solved_ids=solved_ids,
            unsolved_ids=unsolved_ids,
            total_tokens=sum(a.num_tokens for arm in arms for a in arm.attempts),
            total_latency_ms=sum(a.latency_ms for arm in arms for a in arm.attempts),
            config=asdict(self.config),
        )

    def allocate(
        self,
        instances: Sequence[ProblemInstance],
        generate_fn: GeneratorFn,
        verifier_fn: VerifierFn | None = verify_math_answer,
        model: Any = None,
        tokenizer: Any = None,
    ) -> AllocationReport:
        """Allocate ``config.total_budget`` attempts across ``instances``."""
        self.reset()
        self._num_instances = len(instances)
        total_budget = self.config.total_budget
        if not instances or total_budget == 0:
            empty_arms = [ArmState(instance=inst, difficulty=1.0) for inst in instances]
            report = self._build_report(empty_arms, budget_used=0)
            self._last_report = report
            self._active_arms = len(empty_arms)
            return report

        arms = [
            ArmState(
                instance=inst,
                difficulty=compute_input_difficulty(
                    self.config.input_proxy,
                    inst,
                    min_difficulty=self.config.min_difficulty,
                    model=model,
                    tokenizer=tokenizer,
                ),
            )
            for inst in instances
        ]
        rng = np.random.default_rng(self.config.seed)
        budget_used = 0

        while True:
            active_arms = [
                arm for arm in arms if not (self.eliminate_solved and arm.solved)
            ]
            if self.should_stop(
                arms,
                {"budget_used": budget_used, "active_arms": len(active_arms)},
                total_budget,
            ):
                break

            arm = self.select_arm(active_arms, rng)
            attempt = generate_fn(arm.instance, len(arm.attempts), rng)
            if verifier_fn is not None:
                is_correct, extracted = verifier_fn(
                    attempt.text, arm.instance.ground_truth
                )
                attempt.is_correct = is_correct
                if extracted is not None:
                    attempt.extracted_answer = extracted

            arm.attempts.append(attempt)
            budget_used += 1
            self._budget_used = budget_used

            if attempt.is_correct:
                arm.solved = True
            if not arm.solved:
                arm.difficulty = compute_gen_difficulty(
                    self.config.gen_proxy,
                    arm.instance,
                    arm.attempts,
                    min_difficulty=self.config.min_difficulty,
                    model=model,
                    tokenizer=tokenizer,
                )

        report = self._build_report(arms, budget_used=budget_used)
        self._last_report = report
        self._active_arms = len(report.unsolved_ids)
        return report


class DIPABudgetAllocator(BudgetAllocator):
    """Difficulty-Informed Probabilistic Allocation (Algorithm 1)."""

    method_name: str = "dipa"
    deterministic: bool = False

    def compute_sampling_probabilities(
        self,
        active_arms: Sequence[ArmState],
        total_instances: int | None = None,
    ) -> np.ndarray:
        """Compute P_k proportional to 1 / M_k^lambda_t in stabilized log-space.

        In ``dynamic`` mode, ``lambda_t = lambda_value * (|X| / N)`` remains
        strictly proportional to ``|X|`` while staying batch-size invariant
        across ``N = 20, 100, 500``.
        """
        n = len(active_arms)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        if self.config.lambda_mode == "fixed":
            lambda_t = self.config.lambda_value
        else:
            # Normalize by total batch size N so lambda_t stays O(1) across N = 20, 100, 500.
            denom = (
                total_instances
                if total_instances is not None
                else (self._num_instances if self._num_instances > 0 else n)
            )
            denom = max(1, int(denom))
            lambda_t = self.config.lambda_value * (n / denom)
        raw_diffs = np.array(
            [float(arm.difficulty) for arm in active_arms], dtype=np.float64
        )
        raw_diffs = np.where(np.isfinite(raw_diffs), raw_diffs, 1.0)
        difficulties = np.maximum(raw_diffs, self.config.min_difficulty)

        log_w = -lambda_t * np.log(difficulties)
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float64)

        probs = weights / total
        if not np.all(np.isfinite(probs)):
            return np.full(n, 1.0 / n, dtype=np.float64)
        return probs

    def select_arm(
        self,
        active_arms: Sequence[ArmState],
        rng: np.random.Generator,
    ) -> ArmState:
        """Sample an active arm according to difficulty-inverse probabilities."""
        probs = self.compute_sampling_probabilities(active_arms)
        idx = int(rng.choice(len(active_arms), p=probs))
        return active_arms[idx]
