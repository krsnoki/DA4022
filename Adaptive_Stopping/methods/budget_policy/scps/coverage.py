"""Semantic-coverage objects from the SCPS paper, for finite target sets.

Notation (paper Sections 3 and 4):

* ``profile``: dict ``target -> k(y, target)`` in [0, 1] for one completed artifact y.
* ``baseline``: dict ``target -> b(target)`` in [0, 1] (coverage before the portfolio).
* ``mu``: dict ``target -> importance weight`` (finite measure).
* ``m_S(target) = max(b(target), max_{y in S} k(y, target))``.

These functions are exact for finite targets and finite outcome
distributions. They are used to test the paper's claims (Theorem 1,
Theorem 2, Equation 5 and Proposition P3) and are not on the hot path of
the controller.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Profile = Mapping[str, float]
# One joint realization of a batch: probability and, per action, the list of
# completed-artifact profiles that action returned.
Realization = Tuple[float, Mapping[str, Sequence[Profile]]]


def _check_unit(x: float, what: str) -> None:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{what} must be in [0, 1], got {x}")


def coverage_level(portfolio: Iterable[Profile], baseline: Mapping[str, float], targets: Iterable[str]) -> Dict[str, float]:
    """``m_S`` for every target."""
    portfolio = list(portfolio)
    out = {}
    for t in targets:
        b = baseline.get(t, 0.0)
        _check_unit(b, f"baseline[{t}]")
        best = 0.0  # sup over the empty set is 0
        for y in portfolio:
            k = y.get(t, 0.0)
            _check_unit(k, f"k(y, {t})")
            best = max(best, k)
        out[t] = max(b, best)
    return out


def semantic_value(portfolio: Iterable[Profile], baseline: Mapping[str, float], mu: Mapping[str, float]) -> float:
    """Equation (1): ``F_sem(S) = sum_t mu(t) * (m_S(t) - b(t))``."""
    m = coverage_level(portfolio, baseline, mu.keys())
    return sum(w * (m[t] - baseline.get(t, 0.0)) for t, w in mu.items())


def answer_value(answers: Iterable[str], posterior: Mapping[str, float]) -> float:
    """Equation (2): posterior mass of the answer classes present in the portfolio."""
    present = set(a for a in answers if a is not None)
    return sum(p for a, p in posterior.items() if a in present)


def answer_profile(answer: str | None) -> Dict[str, float]:
    """Exact-answer kernel ``k(z, a) = 1{canon(z) = a}``."""
    return {} if answer is None else {answer: 1.0}


# ---------------------------------------------------------------- action lift

def _union_outputs(real: Mapping[str, Sequence[Profile]], batch: Iterable[str]) -> List[Profile]:
    out: List[Profile] = []
    for e in batch:
        out.extend(real.get(e, ()))
    return out


def _check_distribution(realizations: Sequence[Realization]) -> None:
    total = sum(p for p, _ in realizations)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"realization probabilities must sum to 1, got {total}")


def expected_batch_gain(
    realizations: Sequence[Realization],
    batch: Iterable[str],
    portfolio: Sequence[Profile],
    baseline: Mapping[str, float],
    mu: Mapping[str, float],
) -> float:
    """Equation (3): ``G(B | S) = E[F(S u U_B) - F(S)]``."""
    _check_distribution(realizations)
    batch = list(batch)
    base = semantic_value(portfolio, baseline, mu)
    return sum(
        p * (semantic_value(list(portfolio) + _union_outputs(real, batch), baseline, mu) - base)
        for p, real in realizations
    )


def _action_max(real: Mapping[str, Sequence[Profile]], e: str, t: str) -> float:
    return max((y.get(t, 0.0) for y in real.get(e, ())), default=0.0)


def _hit_length(m: float, k_max: float) -> float:
    """Lebesgue measure of residual thresholds r in (m, k_max] hit by one action."""
    return max(0.0, k_max - m)


def residual_hit_terms(
    realizations: Sequence[Realization],
    batch: Sequence[str],
    portfolio: Sequence[Profile],
    baseline: Mapping[str, float],
    mu: Mapping[str, float],
) -> Dict[str, object]:
    """Exact ``v(e)``, ``omega(e, f)``, additive score and duplicate overcount.

    Uses the residual-threshold representation: action e hits u = (t, r)
    iff ``m_S(t) < r <= max_{y in Z_e} k(y, t)``.
    """
    _check_distribution(realizations)
    batch = list(batch)
    m = coverage_level(portfolio, baseline, mu.keys())
    v = {e: 0.0 for e in batch}
    omega = {}
    overcount = 0.0
    for p, real in realizations:
        for t, w in mu.items():
            kmax = {e: _action_max(real, e, t) for e in batch}
            for e in batch:
                v[e] += p * w * _hit_length(m[t], kmax[e])
            for i, e in enumerate(batch):
                for f in batch[i + 1:]:
                    both = _hit_length(m[t], min(kmax[e], kmax[f]))
                    omega[(e, f)] = omega.get((e, f), 0.0) + p * w * both
            # integral over r in (m, 1] of (N(r) - 1)_+, N(r) = #{e : kmax_e >= r}
            levels = sorted((x for x in kmax.values() if x > m[t]), reverse=True)
            for j in range(1, len(levels)):
                # on (levels[j+1] or m, levels[j]] at least j+1 actions hit -> excess j ... accumulate per layer
                lower = levels[j + 1] if j + 1 < len(levels) else m[t]
                overcount += p * w * j * (levels[j] - lower)
    additive = sum(v.values())
    return {"v": v, "omega": omega, "additive": additive, "overcount": overcount}


def bonferroni_lower_bound(terms: Mapping[str, object]) -> float:
    """Equation (5): ``sum_e v(e) - sum_{e<f} omega(e, f)``."""
    return terms["additive"] - sum(terms["omega"].values())


__all__ = [
    "coverage_level",
    "semantic_value",
    "answer_value",
    "answer_profile",
    "expected_batch_gain",
    "residual_hit_terms",
    "bonferroni_lower_bound",
]
