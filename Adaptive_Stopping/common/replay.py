"""Offline replay of step-level stopping policies over cached traces.

The paper evaluates ST-ES this way (App. D, "Inference setting"): run standard
inference once, then apply step tagging and the stopping rule offline. Replay
gives the same stop decision as the live loop, because both feed the same
``Step`` objects (``common.steps``) to the same ``StoppingPolicy``. Only closed
reasoning steps are eligible: the trailing piece at EOS/cap and anything after
``</think>`` never trigger a stop.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Iterable, Sequence

from Adaptive_Stopping.common.policy import StepState, StoppingPolicy
from Adaptive_Stopping.common.steps import Step


@dataclasses.dataclass(frozen=True)
class ReplayOutcome:
    stopped: bool
    stop_step: int | None
    cut_tokens: int  # generated tokens kept: up to the closing token of the stop step, or the full trace
    report: dict


def eligible_steps(steps: Sequence[Step]) -> list[Step]:
    return [s for s in steps if s.closed and s.in_thinking]


def replay_policy(policy: StoppingPolicy, steps: Sequence[Step], n_gen_tokens: int,
                  tags: Sequence[str] | None = None, budget: dict | None = None) -> ReplayOutcome:
    """Run ``policy`` over the steps of one cached trace. ``tags`` (aligned with ``steps``) skips re-tagging."""
    policy.reset()
    for i, s in enumerate(steps):
        if not s.closed:
            break
        metrics = {"tag": tags[i]} if tags is not None else None
        if policy.should_stop(StepState(s.index, s.text, s.end_token, s.in_thinking), metrics, budget):
            return ReplayOutcome(True, s.index, s.end_token, policy.report())
    return ReplayOutcome(False, None, n_gen_tokens, policy.report())


def st_es_cut_points(steps: Sequence[Step], tags: Sequence[str], target_tag: str,
                     deltas: Iterable[int]) -> dict[int, int | None]:
    """Fast path for a delta sweep: ``{delta: cut_tokens or None (no stop)}``.

    Equivalent to ``replay_policy(StepTagEarlyStopping(target_tag, delta), ...)``
    for every delta (checked in the unit tests).
    """
    hits = [s.end_token for s, t in zip(steps, tags) if s.closed and s.in_thinking and t == target_tag]
    return {d: (hits[d] if d < len(hits) else None) for d in deltas}


def step_count_cut_points(steps: Sequence[Step], ks: Iterable[int]) -> dict[int, int | None]:
    """Fast path for the step-count baseline: ``{k: cut_tokens or None}``."""
    ends = [s.end_token for s in eligible_steps(steps)]
    return {k: (ends[k - 1] if 1 <= k <= len(ends) else None) for k in ks}


def token_budget_cut_point(steps: Sequence[Step], max_tokens: int) -> int | None:
    """Last eligible step boundary at or below ``max_tokens`` (helper for matched-budget comparisons)."""
    ends = [s.end_token for s in eligible_steps(steps)]
    i = bisect.bisect_right(ends, max_tokens)
    return ends[i - 1] if i > 0 else None
