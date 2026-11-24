"""Decision-theoretic diagnostics from Bouchard (2026), Sec. 3 and App. A.

* :func:`auroc` - Mann-Whitney AUROC with mid-ranks for ties.
* :func:`gamma_stat` - Gamma(Y) = pi (1 - pi) (AUROC(s, Y) - 1/2).
* :func:`escalation_gain_over_random` - left side of Prop. 3.2 computed
  *directly* by sweeping the escalated fraction t and integrating
  a_i(t) - abar_i(t).
* :func:`prop32_rhs` - right side of Prop. 3.2:
  p_i [Gamma(U_i) - Gamma(U_{>i})] = p_i [Gamma(U_i (1-U_{>i})) - Gamma((1-U_i) U_{>i})].
* :func:`local_benefit_cost` - two-model FOC ingredients m_H(tau) - m_L(tau) and
  gamma(tau) = E[C_H | s_L = tau], estimated with a nearest-neighbour window on
  calibration scores (the paper uses windows of <= 100 observations).
* :func:`frontier_slopes` - finite-difference slope dU*/dC of a frontier, which
  by Prop. A.3 should equal (m_H - m_L)/gamma at interior optima (Eq. 15).
* :func:`stagewise_dominance` - fraction of score levels where m_H > m_L
  (Prop. A.1 condition for monotone threshold trade-off).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of ``scores`` for binary ``labels`` (NaN if one class is absent)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _midranks(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gamma_stat(scores: np.ndarray, labels: np.ndarray) -> float:
    """Gamma(Y) = pi_Y (1 - pi_Y) (AUROC(s, Y) - 1/2); 0 if Y is constant."""
    y = np.asarray(labels, dtype=np.float64)
    pi = float(y.mean()) if len(y) else 0.0
    if pi <= 0.0 or pi >= 1.0:
        return 0.0
    return pi * (1.0 - pi) * (auroc(scores, y) - 0.5)


def escalation_gain_over_random(
    s_i: np.ndarray,
    U_i: np.ndarray,
    U_after: np.ndarray,
    reached: np.ndarray,
    U_other: Optional[np.ndarray] = None,
) -> float:
    """Integral over t in [0,1] of a_i(t) - abar_i(t), computed directly.

    Args:
        s_i: stage-i score for all N queries.
        U_i: quality if the query stops at stage i.
        U_after: quality if the query escalates past stage i (U_{I_{>i}},
            i.e. the downstream cascade outcome with its thresholds fixed).
        reached: boolean mask of queries that reach stage i.
        U_other: quality of queries that never reach stage i (they are fixed
            and cancel in a - abar; only used for overall accuracy levels).

    The bottom fraction t (by increasing s_i, ties broken by index) of reached
    queries escalates; a_i(t) is overall accuracy over all N queries and abar_i
    the straight line between a_i(0) and a_i(1). Uses t_r = r/n and the
    trapezoid rule, for which Prop. 3.2 holds exactly with percentile ranks r/n.
    """
    reached = np.asarray(reached, dtype=bool)
    N = len(reached)
    n = int(reached.sum())
    if n == 0:
        return 0.0
    s = np.asarray(s_i, dtype=np.float64)[reached]
    order = np.argsort(s, kind="mergesort")
    u = np.asarray(U_i, dtype=np.float64)[reached][order]
    v = np.asarray(U_after, dtype=np.float64)[reached][order]
    base = 0.0 if U_other is None else float(np.asarray(U_other, dtype=np.float64)[~reached].sum())
    # a(r) for r = 0..n escalated
    a = (base + u.sum() + np.concatenate([[0.0], np.cumsum(v - u)])) / N
    t = np.arange(n + 1) / n
    abar = (1 - t) * a[0] + t * a[-1]
    diff = a - abar
    return float(np.sum(0.5 * (diff[1:] + diff[:-1])) / n)


def prop32_rhs(s_i: np.ndarray, U_i: np.ndarray, U_after: np.ndarray, reached: np.ndarray) -> Dict[str, float]:
    """Both forms of the right side of Prop. 3.2 (Eq. 8)."""
    reached = np.asarray(reached, dtype=bool)
    p = float(reached.mean())
    s = np.asarray(s_i, dtype=np.float64)[reached]
    u = np.asarray(U_i, dtype=np.float64)[reached]
    v = np.asarray(U_after, dtype=np.float64)[reached]
    form1 = p * (gamma_stat(s, u) - gamma_stat(s, v))
    form2 = p * (gamma_stat(s, u * (1 - v)) - gamma_stat(s, (1 - u) * v))
    return {
        "p_reach": p,
        "rhs_form1": form1,
        "rhs_form2": form2,
        "auroc_own": auroc(s, u),
        "auroc_downstream": auroc(s, v),
    }


def local_benefit_cost(
    s_L: np.ndarray,
    U_L: np.ndarray,
    U_H: np.ndarray,
    C_H: np.ndarray,
    taus: np.ndarray,
    window: int = 100,
) -> Dict[str, np.ndarray]:
    """Nearest-neighbour estimates of m_H(tau) - m_L(tau) and gamma(tau)."""
    s = np.asarray(s_L, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    d = (np.asarray(U_H, dtype=np.float64) - np.asarray(U_L, dtype=np.float64))[order]
    c = np.asarray(C_H, dtype=np.float64)[order]
    w = int(min(window, len(s)))
    benefit, gamma = [], []
    for tau in np.asarray(taus, dtype=np.float64):
        if not np.isfinite(tau):
            benefit.append(np.nan)
            gamma.append(np.nan)
            continue
        centre = int(np.searchsorted(ss, tau))
        lo = max(0, min(centre - w // 2, len(ss) - w))
        benefit.append(d[lo : lo + w].mean())
        gamma.append(c[lo : lo + w].mean())
    benefit = np.asarray(benefit)
    gamma = np.asarray(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = benefit / gamma
    return {"benefit": benefit, "gamma": gamma, "ratio": ratio}


def frontier_slopes(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Central finite-difference slope dU/dC along a frontier (NaN at ends)."""
    cost = np.asarray(cost, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    out = np.full(len(cost), np.nan)
    if len(cost) >= 3:
        dc = cost[2:] - cost[:-2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:-1] = (acc[2:] - acc[:-2]) / dc
    return out


def concavity_violations(cost: np.ndarray, acc: np.ndarray, tol: float = 1e-12) -> float:
    """Fraction of interior frontier points where the upper envelope is locally
    non-concave (slope increases), i.e. where Prop. A.3's decreasing-ratio
    condition fails."""
    cost = np.asarray(cost, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    if len(cost) < 3:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = np.diff(acc) / np.diff(cost)
    sl = sl[np.isfinite(sl)]
    if len(sl) < 2:
        return 0.0
    return float(np.mean(np.diff(sl) > tol))


def stagewise_dominance(s_L: np.ndarray, U_L: np.ndarray, U_H: np.ndarray, window: int = 100) -> float:
    """Share of calibration score levels where the local m_H - m_L > 0."""
    s = np.asarray(s_L, dtype=np.float64)
    taus = np.quantile(s, np.linspace(0.02, 0.98, 49))
    est = local_benefit_cost(s, U_L, U_H, np.ones_like(s), taus, window)["benefit"]
    return float(np.mean(est > 0))
