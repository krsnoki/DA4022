"""Step-count baseline: stop after a fixed number of reasoning steps of any type.

This is the paper's "Token-count baseline" (Sec. 4, after Pu et al., 2025),
which "early-stops the generation for a given count of step (regardless of
their step-type)". After stopping, the answer is forced exactly as for ST-ES.
"""

from __future__ import annotations

from Adaptive_Stopping.common.policy import StepState, StoppingPolicy, budget_exceeded


class StepCountStopping(StoppingPolicy):
    requires_extra_forward_pass = False
    deterministic = True

    def __init__(self, max_steps: int):
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be an int >= 1")
        self.max_steps = max_steps
        self.name = f"step_count[k={max_steps}]"
        self.reset()

    def reset(self, request_metadata: dict | None = None) -> None:
        self.n_steps = 0
        self.stopped_at_step: int | None = None

    def should_stop(self, state: StepState, metrics: dict | None = None, budget: dict | None = None) -> bool:
        if self.stopped_at_step is not None or not state.in_thinking:
            return False
        self.n_steps += 1
        if self.n_steps >= self.max_steps or budget_exceeded(state, self.n_steps, budget):
            self.stopped_at_step = state.step_index
            return True
        return False

    def report(self) -> dict:
        return {"policy": "step_count", "max_steps": self.max_steps, "reasoning_steps_seen": self.n_steps,
                "stopped": self.stopped_at_step is not None, "stopped_at_step": self.stopped_at_step}
