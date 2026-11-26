"""Full-compute baseline: never stops (standard generation up to EOS or the token cap)."""

from __future__ import annotations

from Adaptive_Stopping.common.policy import StepState, StoppingPolicy


class NoStopping(StoppingPolicy):
    name = "standard"

    def reset(self, request_metadata: dict | None = None) -> None:
        self.n_steps = 0

    def should_stop(self, state: StepState, metrics: dict | None = None, budget: dict | None = None) -> bool:
        if state.in_thinking:
            self.n_steps += 1
        return False

    def report(self) -> dict:
        return {"policy": "standard", "reasoning_steps_seen": getattr(self, "n_steps", 0), "stopped": False}
