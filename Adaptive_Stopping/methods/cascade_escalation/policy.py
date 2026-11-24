"""Threshold cascade policy (Bouchard 2026, Sec. 3.1).

A cascade is an ordered sequence of models ``(m_1, ..., m_d)`` (cheapest first)
and thresholds ``(tau_1, ..., tau_{d-1})``. Each non-terminal model emits a
confidence score ``s_i(x)`` from the same generation call. The query escalates
from stage i to i+1 when ``s_i(x) < tau_i`` (strict), so

    I(x; tau) = min({j < d : s_j(x) >= tau_j} U {d})
    U(x; tau) = U_{I(x)}(x)
    C(x; tau) = sum_{j <= I(x)} C_j(x)

``tau = -inf`` never escalates and ``tau = +inf`` always escalates.

Two views of the same policy are provided:

* :func:`evaluate_policy` - vectorised offline evaluation on stored per-query
  outcomes (what the paper does on calibration/test splits).
* :class:`CascadeStoppingPolicy` - the repository's ``StoppingPolicy`` contract
  (``reset`` / ``should_stop`` / ``report``) for running a cascade live, one
  stage at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# A score provider returns the confidence score of `model` for all N queries.
# `next_model` is the model the query would escalate to (needed only by
# benefit-trained scores that predict U_next - U_model).
ScoreFn = Callable[[int, Optional[int]], np.ndarray]


@dataclass(frozen=True)
class CascadePolicy:
    """An ordered model sequence with one threshold per non-terminal stage."""

    sequence: Tuple[int, ...]
    thresholds: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.sequence) < 1:
            raise ValueError("a cascade needs at least one model")
        if len(set(self.sequence)) != len(self.sequence):
            raise ValueError("a model may appear at most once in a cascade")
        if len(self.thresholds) != len(self.sequence) - 1:
            raise ValueError(
                f"need {len(self.sequence) - 1} thresholds for a depth-{len(self.sequence)} cascade, "
                f"got {len(self.thresholds)}"
            )
        for t in self.thresholds:
            if np.isnan(t):
                raise ValueError("thresholds must not be NaN")

    @property
    def depth(self) -> int:
        return len(self.sequence)

    @property
    def effective_depth(self) -> int:
        """Number of stages a query can actually reach (stops at the first tau=-inf)."""
        d = 1
        for t in self.thresholds:
            if t == -np.inf:
                break
            d += 1
        return d

    def to_dict(self, model_names: Optional[Sequence[str]] = None) -> Dict:
        seq = [model_names[m] for m in self.sequence] if model_names is not None else list(self.sequence)
        return {"sequence": seq, "thresholds": [float(t) for t in self.thresholds]}


def stop_index(policy: CascadePolicy, score_fn: ScoreFn, n: int) -> np.ndarray:
    """Stage (0-based position in ``policy.sequence``) where each query stops."""
    stop = np.full(n, policy.depth - 1, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    for i, tau in enumerate(policy.thresholds):
        nxt = policy.sequence[i + 1]
        s = np.asarray(score_fn(policy.sequence[i], nxt), dtype=np.float64)
        accept = active & ~(s < tau)
        stop[accept] = i
        active &= ~accept
        if not active.any():
            break
    return stop


def evaluate_policy(
    policy: CascadePolicy,
    U: np.ndarray,
    C: np.ndarray,
    score_fn: ScoreFn,
    idx: Optional[np.ndarray] = None,
    latency: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate a policy on stored outcomes.

    Args:
        U: [N, M] binary (or real) quality of every model on every query.
        C: [N, M] cost of every model on every query.
        score_fn: score provider over all N queries.
        idx: optional subset of query indices (e.g. a test split).
        latency: optional [N, M] per-query latency of every model; cascade latency
            is the sum over the stages a query visits (sequential execution).

    Returns:
        accuracy, mean cost, per-stage reach/escalation rates, and latency stats.
    """
    U = np.asarray(U, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    n_all = U.shape[0]
    stop = stop_index(policy, score_fn, n_all)
    if idx is None:
        idx = np.arange(n_all)
    stop = stop[idx]
    seq = np.asarray(policy.sequence)
    final_model = seq[stop]
    quality = U[idx, final_model]
    # cumulative cost along the sequence
    cum_cost = np.cumsum(C[idx][:, seq], axis=1)
    cost = cum_cost[np.arange(len(idx)), stop]
    out = {
        "accuracy": float(quality.mean()) if len(idx) else float("nan"),
        "mean_cost": float(cost.mean()) if len(idx) else float("nan"),
        "escalation_rate": float((stop > 0).mean()) if len(idx) else float("nan"),
        "mean_models_called": float((stop + 1).mean()) if len(idx) else float("nan"),
    }
    for j in range(policy.depth):
        out[f"reach_stage{j + 1}"] = float((stop >= j).mean()) if len(idx) else float("nan")
        out[f"answer_stage{j + 1}"] = float((stop == j).mean()) if len(idx) else float("nan")
    if latency is not None:
        lat = np.cumsum(np.asarray(latency, dtype=np.float64)[idx][:, seq], axis=1)[np.arange(len(idx)), stop]
        out["latency_mean"] = float(lat.mean())
        out["latency_median"] = float(np.median(lat))
        out["latency_p95"] = float(np.percentile(lat, 95))
    return out


@dataclass
class CascadeStoppingPolicy:
    """Live cascade policy following the repository's StoppingPolicy contract.

    ``state`` is the 0-based index of the stage that just produced an answer and
    ``metrics`` must contain ``"score"`` (the confidence of that stage's answer,
    computed from the same generation). ``budget`` optionally caps the number of
    stages (``budget["max_stages"]``). Stopping is deterministic given the score.
    """

    policy: CascadePolicy
    _trace: List[Dict] = field(default_factory=list)
    _n_requests: int = 0
    _n_escalations: int = 0

    def reset(self, request_metadata: Optional[Dict] = None) -> None:
        self._trace = []
        self._n_requests += 1

    def should_stop(self, state: int, metrics: Dict, budget: Optional[Dict] = None) -> bool:
        stage = int(state)
        if stage < 0 or stage >= self.policy.depth:
            raise ValueError(f"stage {stage} out of range for depth {self.policy.depth}")
        max_stages = None if not budget else budget.get("max_stages")
        last = stage == self.policy.depth - 1 or (max_stages is not None and stage + 1 >= int(max_stages))
        score = metrics.get("score") if metrics else None
        if last:
            stop = True
        elif score is None or (isinstance(score, float) and np.isnan(score)):
            # missing telemetry: escalate (conservative: prefer quality)
            stop = False
        else:
            stop = not (float(score) < self.policy.thresholds[stage])
        if not stop:
            self._n_escalations += 1
        self._trace.append({"stage": stage, "score": score, "stop": stop})
        return stop

    def report(self) -> Dict:
        return {
            "policy": self.policy.to_dict(),
            "trace": list(self._trace),
            "requests": self._n_requests,
            "escalations": self._n_escalations,
        }
