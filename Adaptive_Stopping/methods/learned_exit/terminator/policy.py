"""Stopping rule for TERMINATOR and a fixed thinking-budget baseline.

Both classes follow the module contract in Adaptive_Stopping/README:

    reset(request_metadata=None)
    should_stop(state, metrics, budget) -> bool
    report() -> dict

`should_stop` is called once per generated thinking token, after the token is
appended. Inputs:

- state:   dict, optional key "thinking_tokens" (tokens generated so far).
           If missing, the policy counts its own calls.
- metrics: dict with key "terminator_prob", the probe probability for the
           newest thinking token. It comes from the probe that runs on the
           base model's last hidden state, so it needs one extra transformer
           layer per token but no extra pass of the base model. If the key is
           missing or not finite, the step counts as missing telemetry and
           only the budget can stop generation.
- budget:  int or None, maximum number of thinking tokens. None means no cap.

Stopping is deterministic given the probability sequence.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence


def window_fires(window_probs: Sequence[float], threshold: float, window: int) -> bool:
    """True when the window is full and a strict majority is above threshold."""
    if len(window_probs) < window:
        return False
    above = sum(1 for p in list(window_probs)[-window:] if p > threshold)
    return above / window > 0.5


def exit_index(probs: Sequence[float], threshold: float = 0.7, window: int = 10) -> Optional[int]:
    """Number of thinking tokens kept by TERMINATOR, or None if it never fires.

    probs[i] is the probe probability after thinking token i. When the rule
    fires at token i, tokens 0..i are kept, so the return value is i + 1.
    This is what online generation would keep for the same sampled trace,
    because the probe is causal.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    recent: deque = deque(maxlen=window)
    for i, p in enumerate(probs):
        recent.append(float(p))
        if window_fires(recent, threshold, window):
            return i + 1
    return None


def _count(state, calls: int) -> int:
    if isinstance(state, dict) and state.get("thinking_tokens") is not None:
        return int(state["thinking_tokens"])
    return calls


def _budget_hit(n_tokens: int, budget) -> bool:
    return budget is not None and n_tokens >= int(budget)


class TerminatorPolicy:
    """Sliding-window majority vote over probe probabilities."""

    def __init__(self, threshold: float = 0.7, window: int = 10):
        if window <= 0:
            raise ValueError("window must be positive")
        self.threshold = float(threshold)
        self.window = int(window)
        self.reset()

    def reset(self, request_metadata=None):
        self.request_metadata = request_metadata
        self._recent: deque = deque(maxlen=self.window)
        self._calls = 0
        self._n_pred = 0
        self._n_missing = 0
        self._max_prob = None
        self._stopped = False
        self._reason = None
        self._exit_pos = None

    def should_stop(self, state=None, metrics=None, budget=None) -> bool:
        if self._stopped:
            return True
        self._calls += 1
        n = _count(state, self._calls)
        prob = None if not isinstance(metrics, dict) else metrics.get("terminator_prob")
        if prob is None or not math.isfinite(float(prob)):
            self._n_missing += 1
        else:
            prob = float(prob)
            self._n_pred += 1
            self._max_prob = prob if self._max_prob is None else max(self._max_prob, prob)
            self._recent.append(prob)
            if window_fires(self._recent, self.threshold, self.window):
                return self._stop("probe", n)
        if _budget_hit(n, budget):
            return self._stop("budget", n)
        return False

    def _stop(self, reason: str, n: int) -> bool:
        self._stopped = True
        self._reason = reason
        self._exit_pos = n
        return True

    def report(self) -> dict:
        return {
            "method": "terminator",
            "threshold": self.threshold,
            "window": self.window,
            "stopped": self._stopped,
            "reason": self._reason,
            "exit_pos": self._exit_pos,
            "n_calls": self._calls,
            "n_predictions": self._n_pred,
            "n_missing": self._n_missing,
            "max_prob": self._max_prob,
        }


class FixedBudgetPolicy:
    """Baseline: stop thinking after a fixed number of tokens."""

    def __init__(self, budget_tokens: Optional[int] = None):
        self.budget_tokens = budget_tokens
        self.reset()

    def reset(self, request_metadata=None):
        self.request_metadata = request_metadata
        self._calls = 0
        self._stopped = False
        self._exit_pos = None

    def should_stop(self, state=None, metrics=None, budget=None) -> bool:
        if self._stopped:
            return True
        self._calls += 1
        n = _count(state, self._calls)
        b = budget if budget is not None else self.budget_tokens
        if _budget_hit(n, b):
            self._stopped = True
            self._exit_pos = n
        return self._stopped

    def report(self) -> dict:
        return {
            "method": "fixed_budget",
            "budget_tokens": self.budget_tokens,
            "stopped": self._stopped,
            "reason": "budget" if self._stopped else None,
            "exit_pos": self._exit_pos,
            "n_calls": self._calls,
        }
