"""Step-Tagging Early-Stopping (ST-ES) policy.

Paper, Eq. 3 and Algorithm 1: for a target step type tau* and threshold delta,
generation continues while ``f_freq(S_running, tau*) <= delta`` and stops at
the first step that makes the count exceed ``delta``. That step is kept, and
the model is then asked for its current best answer (see ``forced_answer``).
``delta = 0`` therefore stops at the first step of type tau*.

The policy reads only the generated step text (black-box), needs no extra
forward pass of the backbone, and is deterministic given the tagger.
"""

from __future__ import annotations

from collections import Counter

from Adaptive_Stopping.common.policy import StepState, StoppingPolicy, budget_exceeded
from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger, StepTagger
from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS


class StepTagEarlyStopping(StoppingPolicy):
    requires_extra_forward_pass = False
    deterministic = True

    def __init__(self, target_tag: str, delta: int | None, tagger: StepTagger | None = None):
        if target_tag not in TAG_IDS:
            raise ValueError(f"unknown step type {target_tag!r}")
        if delta is not None and (not isinstance(delta, int) or delta < 0):
            raise ValueError("delta must be a non-negative int, or None to only monitor")
        self.target_tag = target_tag
        self.delta = delta
        self.tagger = tagger or KeywordStepTagger()
        self.name = f"st_es[{target_tag},delta={delta}]"
        self.reset()

    def reset(self, request_metadata: dict | None = None) -> None:
        self.count = 0
        self.n_steps = 0
        self.n_missing = 0
        self.tag_counts: Counter = Counter()
        self.stopped_at_step: int | None = None
        self.stop_reason: str | None = None

    def should_stop(self, state: StepState, metrics: dict | None = None, budget: dict | None = None) -> bool:
        if self.stopped_at_step is not None or not state.in_thinking:
            return False
        self.n_steps += 1
        if state.text is None and not (metrics and "tag" in metrics):
            self.n_missing += 1  # missing telemetry: never counts toward tau*
            tag = None
        else:
            tag = metrics["tag"] if metrics and "tag" in metrics else self.tagger.tag(state.text)
            self.tag_counts[tag] += 1
        if tag == self.target_tag:
            self.count += 1
        if self.delta is not None and self.count > self.delta:
            self.stopped_at_step, self.stop_reason = state.step_index, "constraint"
            return True
        if budget_exceeded(state, self.n_steps, budget):
            self.stopped_at_step, self.stop_reason = state.step_index, "budget"
            return True
        return False

    def report(self) -> dict:
        return {
            "policy": "st_es", "target_tag": self.target_tag, "delta": self.delta, "tagger": self.tagger.name,
            "count": self.count, "reasoning_steps_seen": self.n_steps, "missing_steps": self.n_missing,
            "stopped": self.stopped_at_step is not None, "stopped_at_step": self.stopped_at_step,
            "stop_reason": self.stop_reason, "tag_counts": dict(self.tag_counts),
        }
