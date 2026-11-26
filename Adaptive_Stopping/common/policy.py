"""Shared stopping-policy contract for ``Adaptive_Stopping`` (see README_adaptiveStopping.md)."""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass(frozen=True)
class StepState:
    """What a step-level policy sees when a reasoning step completes.

    ``text`` is the step without its delimiter; ``None`` means the step text
    is unavailable (missing telemetry). ``n_tokens`` counts generated tokens up
    to and including the token that closed the step. ``in_thinking`` is False
    once the model has emitted its end-of-thinking marker.
    """

    step_index: int
    text: str | None
    n_tokens: int
    in_thinking: bool = True


class StoppingPolicy(abc.ABC):
    """Decides, at a well-defined point of the generation loop, whether to stop.

    Step-level policies are evaluated once per completed reasoning step (at the
    step delimiter), using only the generated text: no extra forward pass of
    the backbone and no access to its internals.
    """

    name: str = "policy"
    requires_extra_forward_pass: bool = False
    deterministic: bool = True

    def reset(self, request_metadata: dict | None = None) -> None:
        """Clear per-request state."""

    @abc.abstractmethod
    def should_stop(self, state: StepState, metrics: dict | None = None, budget: dict | None = None) -> bool:
        """Return True to stop generation after the step in ``state``."""

    def report(self) -> dict:
        return {"policy": self.name}


def budget_exceeded(state: StepState, n_steps_seen: int, budget: dict | None) -> bool:
    """Hard caps shared by step-level policies.

    ``budget`` may contain ``max_steps`` (reasoning steps) and/or ``max_tokens``
    (generated tokens). A zero budget stops at the first completed step.
    Missing keys mean "no cap".
    """
    if not budget:
        return False
    max_steps = budget.get("max_steps")
    max_tokens = budget.get("max_tokens")
    if max_steps is not None and max_steps < 0 or max_tokens is not None and max_tokens < 0:
        raise ValueError(f"budget values must be >= 0, got {budget}")
    if max_steps is not None and n_steps_seen >= max_steps:
        return True
    if max_tokens is not None and state.n_tokens >= max_tokens:
        return True
    return False
