"""Formal statements of Appendix D ("Formal Statements and Proofs"), executable.

Every quantity is in nats. Functions mirror the paper's numbering:

* Definition D.1  (ε, δ)-fork token            -> :func:`fork_token_report`
* Eq. (5)         sandwich bound              -> :func:`sandwich_bounds`
* Assumption D.2  anchor-fork degeneracy      -> :func:`check_anchor_fork_degeneracy`
* Lemma D.3       pairwise ordering bound     -> :func:`pairwise_order_probability`,
                                                 :func:`pairwise_order_bounds`
* Proposition D.4 T_pos raises fork entropy   -> :func:`expected_fork_entropy`,
                                                 :func:`proposition_d4_threshold`
* Remark D.5      LC is the minimum            -> :func:`lc_fork_entropy`
* Assumption D.6 + Corollary D.7              -> :func:`corollary_d7_applies`
* Proposition D.8 T_token invariance under LC -> :func:`confidence_at_temperature`,
                                                 :func:`strengthened_gap_holds`
* Proposition D.9 AR beats LC                 -> :func:`ar_fork_entropy`
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

__all__ = [
    "entropy",
    "binary_entropy",
    "inverse_binary_entropy",
    "joint_entropies",
    "fork_token_report",
    "sandwich_bounds",
    "check_anchor_fork_degeneracy",
    "sigmoid",
    "pairwise_order_probability",
    "pairwise_order_bounds",
    "expected_fork_entropy",
    "lc_fork_entropy",
    "ar_fork_entropy",
    "proposition_d4_threshold",
    "proposition_d4_holds",
    "corollary_d7_applies",
    "confidence_at_temperature",
    "strengthened_gap_holds",
]


# --------------------------------------------------------------------------- #
# Entropy utilities
# --------------------------------------------------------------------------- #
def entropy(probs: Iterable[float]) -> float:
    """Shannon entropy (nats) of a probability vector (zeros allowed)."""
    probs = list(probs)
    total = sum(probs)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


def binary_entropy(p: float) -> float:
    """``H(p) = -p ln p - (1-p) ln(1-p)``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return entropy([p, 1.0 - p]) if 0.0 < p < 1.0 else 0.0


def inverse_binary_entropy(h: float, tol: float = 1e-13) -> float:
    """The ``p ∈ [1/2, 1]`` with ``binary_entropy(p) = h`` (bisection)."""
    if not 0.0 <= h <= math.log(2) + 1e-12:
        raise ValueError(f"binary entropy must lie in [0, ln 2], got {h}")
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binary_entropy(mid) > h:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def joint_entropies(joint: Mapping[Tuple[object, object], float]) -> Dict[str, float]:
    """Entropies of a joint table ``p(x^ℓ, s)`` (conditioned on a fixed ``x_t``).

    Returns ``H(X)``, ``H(S)``, ``H(X, S)``, ``H(X | S)`` and ``H(S | X)`` where
    ``X`` is the fork token's value and ``S = ⟦x_0⟧`` the semantic outcome.
    """
    total = sum(joint.values())
    if total <= 0:
        raise ValueError("joint must have positive mass")
    px: Dict[object, float] = {}
    ps: Dict[object, float] = {}
    for (xv, sv), p in joint.items():
        px[xv] = px.get(xv, 0.0) + p / total
        ps[sv] = ps.get(sv, 0.0) + p / total
    h_xs = entropy([p / total for p in joint.values()])
    h_x = entropy(px.values())
    h_s = entropy(ps.values())
    return {"H_X": h_x, "H_S": h_s, "H_XS": h_xs, "H_X_given_S": h_xs - h_s, "H_S_given_X": h_xs - h_x}


def fork_token_report(joint: Mapping[Tuple[object, object], float], eps: float, delta: float) -> Dict[str, object]:
    """Definition D.1: is position ℓ an (ε, δ)-fork token?

    Conditions: (1) ``H(x^ℓ | x_t, ⟦x_0⟧) <= ε`` and (2) ``H(⟦x_0⟧ | x_t, x^ℓ) <= δ``.
    Also returns the sandwich bound of Eq. (5) and whether it holds (it must,
    by the chain rule, whenever both conditions hold).
    """
    h = joint_entropies(joint)
    cond1 = h["H_X_given_S"] <= eps + 1e-12
    cond2 = h["H_S_given_X"] <= delta + 1e-12
    lo, hi = sandwich_bounds(h["H_X"], eps, delta)
    return {
        **h,
        "condition_1": cond1,
        "condition_2": cond2,
        "is_fork": cond1 and cond2,
        "sandwich_lower": lo,
        "sandwich_upper": hi,
        "sandwich_holds": lo - 1e-12 <= h["H_S"] <= hi + 1e-12,
    }


def sandwich_bounds(fork_entropy: float, eps: float, delta: float) -> Tuple[float, float]:
    """Eq. (5): ``H(x^ℓ) − ε <= H(⟦x_0⟧) <= H(x^ℓ) + δ``."""
    return fork_entropy - eps, fork_entropy + delta


# --------------------------------------------------------------------------- #
# Assumption D.2 and Lemma D.3
# --------------------------------------------------------------------------- #
def check_anchor_fork_degeneracy(
    fork_confidences: Mapping[frozenset, float],
    anchor_confidences: Mapping[object, float],
    h0: float,
    etas: Mapping[object, float],
) -> Dict[str, bool]:
    """Check both conditions of Assumption D.2 on an explicit state table.

    Args:
        fork_confidences: ``c^ℓ`` for every reachable set ``S`` of revealed anchors.
        anchor_confidences: ``c^a`` per anchor (constant while ℓ is masked).
        h0: fork entropy at ``x_t``.
        etas: ``η_a > 0`` per anchor.
    """
    gap = all(
        anchor_confidences[a] > c_fork
        for S, c_fork in fork_confidences.items()
        for a in anchor_confidences
        if a not in S
    )
    eta_ok = all(e > 0 for e in etas.values()) and h0 > sum(etas.values())
    return {"confidence_gap": gap, "linear_degradation_well_posed": eta_ok}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def pairwise_order_probability(c_anchor: float, c_fork: float, tpos: float) -> float:
    """``P(T(a) < T(ℓ))`` for TLC, K = 1, with *constant* confidences.

    Equals ``c_a^{1/T} / (c_a^{1/T} + c_ℓ^{1/T}) = σ(ln(c_a / c_ℓ) / T_pos)`` (proof
    of Lemma D.3). Limits: ``T_pos = 0`` -> LC (1 if ``c_a > c_ℓ``);
    ``T_pos = ∞`` -> random (1/2).
    """
    if c_anchor <= 0 or c_fork <= 0:
        raise ValueError("confidences must be positive")
    r = math.log(c_anchor / c_fork)
    if tpos == 0:
        return 1.0 if r > 0 else 0.0 if r < 0 else 0.5
    if math.isinf(tpos):
        return 0.5
    return sigmoid(r / tpos)


def pairwise_order_bounds(delta_a: float, Delta_a: float, tpos: float) -> Tuple[float, float]:
    """Lemma D.3: ``σ(δ_a / T_pos) <= P(T(a) < T(ℓ)) <= σ(Δ_a / T_pos)``.

    ``δ_a = inf ln(c_a / c_ℓ)`` and ``Δ_a = sup ln(c_a / c_ℓ)`` over reachable
    states with both positions masked; ``0 < δ_a <= Δ_a < ∞``.
    """
    if not 0 < delta_a <= Delta_a < math.inf:
        raise ValueError("need 0 < δ_a <= Δ_a < ∞")
    if not tpos > 0:
        raise ValueError("Lemma D.3 requires T_pos > 0")
    if math.isinf(tpos):
        return 0.5, 0.5
    return sigmoid(delta_a / tpos), sigmoid(Delta_a / tpos)


# --------------------------------------------------------------------------- #
# Proposition D.4, Remark D.5, Proposition D.9
# --------------------------------------------------------------------------- #
def expected_fork_entropy(h0: float, etas: Mapping[object, float], p_anchor_first: Mapping[object, float]) -> float:
    """Tower-rule identity from the proof of Prop. D.4:

    ``H_P(x^ℓ | x_t) = H_q(x^ℓ | x_t) − Σ_a P(T(a) < T(ℓ)) η_a``.

    Note: as in the paper, ``H_P`` here is the *expected* model entropy at the
    step the fork is unmasked, ``E[H_q(x^ℓ | x_{T(ℓ)-1})]``.
    """
    return h0 - sum(p_anchor_first[a] * etas[a] for a in etas)


def lc_fork_entropy(h0: float, etas: Mapping[object, float]) -> float:
    """Remark D.5: under LC every anchor precedes the fork -> ``H_0 − Σ η_a`` (the minimum)."""
    return h0 - sum(etas.values())


def ar_fork_entropy(h0: float, etas: Mapping[int, float], fork_position: int) -> float:
    """Prop. D.9: under left-to-right order only anchors left of ℓ are revealed first."""
    return h0 - sum(e for a, e in etas.items() if a < fork_position)


def proposition_d4_threshold(deltas: Sequence[float], Deltas: Sequence[float]) -> float:
    """The ratio ``Δ / δ`` with ``δ = min_a δ_a``, ``Δ = max_a Δ_a``.

    Prop. D.4: any ``T'_pos > T_pos · Δ/δ`` strictly increases expected fork entropy.
    """
    d, D = min(deltas), max(Deltas)
    if not 0 < d <= D:
        raise ValueError("need 0 < δ <= Δ")
    return D / d


def proposition_d4_holds(tpos: float, tpos_prime: float, deltas: Sequence[float], Deltas: Sequence[float]) -> bool:
    """Whether the sufficient condition ``T'_pos > T_pos · Δ/δ`` is met."""
    return tpos_prime > tpos * proposition_d4_threshold(deltas, Deltas)


def corollary_d7_applies(fork_entropy_hi: float, fork_entropy_lo: float, eps: float, delta: float) -> bool:
    """Corollary D.7: if the fork-entropy gain exceeds ``ε + δ`` (and Assumption
    D.6 holds) then the semantic entropy strictly increases as well."""
    return fork_entropy_hi - fork_entropy_lo > eps + delta


# --------------------------------------------------------------------------- #
# Proposition D.8
# --------------------------------------------------------------------------- #
def confidence_at_temperature(logits: Sequence[float], temperature: float) -> float:
    """``c(T) = max_v softmax(z / T)_v = 1 / (1 + Σ_{i>=2} exp((z_(i) − z_(1)) / T))``."""
    z = sorted((float(v) for v in logits), reverse=True)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return 1.0 / (1.0 + sum(math.exp((zi - z[0]) / temperature) for zi in z[1:]))


def strengthened_gap_holds(anchor_logits: Sequence[float], fork_logits: Sequence[float]) -> bool:
    """Prop. D.8 condition: ``z^a_(1) − z^a_(i) >= z^ℓ_(1) − z^ℓ_(i)`` for all ``i >= 2``,
    at least one strictly (sorted logits, equal vocabulary sizes)."""
    za = sorted((float(v) for v in anchor_logits), reverse=True)
    zl = sorted((float(v) for v in fork_logits), reverse=True)
    if len(za) != len(zl):
        raise ValueError("logit vectors must have the same length")
    gaps_a = [za[0] - v for v in za[1:]]
    gaps_l = [zl[0] - v for v in zl[1:]]
    return all(ga >= gl for ga, gl in zip(gaps_a, gaps_l)) and any(ga > gl for ga, gl in zip(gaps_a, gaps_l))
