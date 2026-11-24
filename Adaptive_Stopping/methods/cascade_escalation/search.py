"""Calibration-time threshold search and budget-constrained policy selection.

Implements the paper's protocol (Bouchard 2026, Sec. 4.1 / App. B.4):

1. For every ordered model subsequence (ordered by calibration mean cost)
   enumerate threshold vectors on calibration score partitions.
2. Policy class ``S_d`` = all subsequences of length <= d (``S_1`` = single
   models). Classes are nested, so calibration accuracy is non-decreasing in d.
3. For each budget B, select the calibration policy with the highest accuracy
   among those with mean cost <= B (ties -> lower cost), freeze it, and
   evaluate it on the held-out split.

Exhaustiveness. Depth-2 search uses *every* calibration partition (exact, as in
the paper). Deeper cascades use a quantile grid of ``grid_sizes[d]`` thresholds
per stage (plus the never/always sentinels), which is an approximation of the
paper's exhaustive search with pruning kernels; it is recorded in results.
Because the sentinels are always in the grid, the nesting S_{d-1} within S_d
is preserved exactly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .policy import CascadePolicy, ScoreFn, evaluate_policy

DEFAULT_GRID_SIZES = {2: None, 3: 128, 4: 40, 5: 16}  # None = all distinct partitions


def threshold_grid(scores: np.ndarray, size: Optional[int]) -> np.ndarray:
    """Candidate thresholds for one stage from calibration scores.

    ``tau = v`` escalates queries with ``s < v``; the grid contains -inf
    (never escalate), the distinct calibration values (or ``size`` quantiles of
    them), and +inf (always escalate). Tied scores therefore move together.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    distinct = np.unique(s)
    if size is not None and len(distinct) > size:
        qs = np.linspace(0.0, 1.0, size)
        pos = np.unique(np.round(qs * (len(distinct) - 1)).astype(int))
        distinct = distinct[pos]
    # threshold = v means escalate s < v; v = min value escalates nothing, so
    # start with -inf and then use each distinct value above the minimum.
    inner = distinct[1:] if len(distinct) else distinct
    return np.concatenate([[-np.inf], inner, [np.inf]])


@dataclass
class CandidateSet:
    """All calibration-evaluated policies of one sequence."""

    sequence: Tuple[int, ...]
    grids: List[np.ndarray]
    acc: np.ndarray  # flattened over the threshold grid
    cost: np.ndarray

    def policy(self, flat_index: int) -> CascadePolicy:
        if not self.grids:
            return CascadePolicy(self.sequence, ())
        shape = tuple(len(g) for g in self.grids)
        multi = np.unravel_index(int(flat_index), shape)
        return CascadePolicy(self.sequence, tuple(float(g[i]) for g, i in zip(self.grids, multi)))


def enumerate_sequence(
    sequence: Sequence[int],
    U: np.ndarray,
    C: np.ndarray,
    score_fn: ScoreFn,
    cal_idx: np.ndarray,
    grid_sizes: Optional[Dict[int, Optional[int]]] = None,
) -> CandidateSet:
    """Evaluate every grid threshold vector of one sequence on calibration data.

    Uses the telescoping identity (masks E_i = 1[s_i < tau_i])
        U = U_1 + E_1 (U_2 - U_1) + E_1 E_2 (U_3 - U_2) + ...
        C = C_1 + E_1 C_2 + E_1 E_2 C_3 + ...
    so all grid points are evaluated with a few matrix products.
    """
    grid_sizes = {**DEFAULT_GRID_SIZES, **(grid_sizes or {})}
    seq = tuple(int(m) for m in sequence)
    d = len(seq)
    Uc = np.asarray(U, dtype=np.float64)[cal_idx][:, list(seq)]
    Cc = np.asarray(C, dtype=np.float64)[cal_idx][:, list(seq)]
    n = len(cal_idx)
    if d == 1:
        return CandidateSet(seq, [], np.array([Uc[:, 0].mean()]), np.array([Cc[:, 0].mean()]))

    size = grid_sizes.get(d, 16)
    grids, masks = [], []
    for i in range(d - 1):
        s = np.asarray(score_fn(seq[i], seq[i + 1]), dtype=np.float64)[cal_idx]
        g = threshold_grid(s, size)
        grids.append(g)
        masks.append((s[None, :] < g[:, None]).astype(np.float64))  # [G_i, n]

    dU = [Uc[:, j + 1] - Uc[:, j] for j in range(d - 1)]
    shape = tuple(len(g) for g in grids)
    acc = np.full(shape, Uc[:, 0].sum())
    cost = np.full(shape, Cc[:, 0].sum())
    letters = "abcdefgh"
    for j in range(1, d):
        # term_j = sum_n prod_{i<j} E_i[g_i, n] * w_j[n]
        sub = ",".join(f"{letters[i]}n" for i in range(j)) + ",n->" + letters[:j]
        for w, target in ((dU[j - 1], acc), (Cc[:, j], cost)):
            term = np.einsum(sub, *masks[:j], w, optimize=True)
            target += term.reshape(term.shape + (1,) * (d - 1 - j))
    return CandidateSet(seq, grids, (acc / n).ravel(), (cost / n).ravel())


def ordered_subsequences(order: Sequence[int], max_depth: int) -> Iterable[Tuple[int, ...]]:
    """All subsequences (preserving cost order) of length 1..max_depth."""
    for d in range(1, max_depth + 1):
        for combo in itertools.combinations(order, d):
            yield tuple(combo)


@dataclass
class Frontier:
    """Calibration Pareto frontier: increasing cost, strictly increasing accuracy."""

    cost: np.ndarray
    acc: np.ndarray
    policies: List[CascadePolicy]

    def select(self, budget: float, tol: float = 1e-12) -> Optional[int]:
        """Index of the best feasible frontier point for a budget (None if infeasible)."""
        k = int(np.searchsorted(self.cost, budget + tol, side="right")) - 1
        return k if k >= 0 else None


def pareto_frontier(cands: Sequence[CandidateSet], tie_tol: float = 1e-15) -> Frontier:
    """Max-accuracy-at-each-cost frontier over many candidate sets.

    Selection rule: highest calibration accuracy, ties broken by lower cost
    (and then by shallower effective depth, to avoid unused stages).
    """
    accs = np.concatenate([c.acc for c in cands])
    costs = np.concatenate([c.cost for c in cands])
    owner = np.concatenate([np.full(len(c.acc), i) for i, c in enumerate(cands)])
    local = np.concatenate([np.arange(len(c.acc)) for c in cands])
    depth = np.concatenate([np.full(len(c.acc), len(c.sequence)) for c in cands])
    # sort by cost asc, then acc desc, then depth asc
    order = np.lexsort((depth, -accs, costs))
    best = -np.inf
    keep = []
    for k in order:
        if accs[k] > best + tie_tol:
            keep.append(k)
            best = accs[k]
    keep = np.asarray(keep, dtype=np.int64)
    pols = [cands[owner[k]].policy(local[k]) for k in keep]
    return Frontier(costs[keep], accs[keep], pols)


def budget_grid(cal_acc: np.ndarray, cal_cost: np.ndarray, n_budgets: int = 500) -> np.ndarray:
    """Paper's budget grid: from the cheapest model's cost to the cost of the
    highest-accuracy model (both on calibration), ``n_budgets`` equally spaced."""
    lo = float(np.min(cal_cost))
    best = np.flatnonzero(cal_acc == np.max(cal_acc))
    hi = float(np.min(cal_cost[best]))
    if hi <= lo:
        hi = float(np.max(cal_cost))
    return lo + (np.arange(n_budgets) / max(n_budgets - 1, 1)) * (hi - lo)


def select_and_evaluate(
    frontier: Frontier,
    budgets: np.ndarray,
    U: np.ndarray,
    C: np.ndarray,
    score_fn: ScoreFn,
    test_idx: np.ndarray,
    latency: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Freeze the calibration-selected policy at each budget and evaluate on test."""
    cache: Dict[int, Dict] = {}
    rows = []
    for b in budgets:
        k = frontier.select(float(b))
        if k is None:
            rows.append({"budget": float(b), "feasible": False})
            continue
        if k not in cache:
            res = evaluate_policy(frontier.policies[k], U, C, score_fn, test_idx, latency)
            res["cal_accuracy"] = float(frontier.acc[k])
            res["cal_cost"] = float(frontier.cost[k])
            res["policy"] = frontier.policies[k]
            cache[k] = res
        rows.append({"budget": float(b), "feasible": True, **cache[k]})
    return rows


def cost_order(C: np.ndarray, cal_idx: np.ndarray) -> List[int]:
    """Models ordered by calibration mean cost (cheapest first)."""
    means = np.asarray(C, dtype=np.float64)[cal_idx].mean(axis=0)
    return [int(i) for i in np.argsort(means, kind="stable")]


def build_frontiers(
    U: np.ndarray,
    C: np.ndarray,
    score_fn: ScoreFn,
    cal_idx: np.ndarray,
    max_depth: int,
    grid_sizes: Optional[Dict[int, Optional[int]]] = None,
    models: Optional[Sequence[int]] = None,
) -> Dict[int, Frontier]:
    """Calibration frontiers for the nested classes S_1 ... S_max_depth."""
    order = cost_order(C, cal_idx)
    if models is not None:
        allowed = set(int(m) for m in models)
        order = [m for m in order if m in allowed]
    by_depth: Dict[int, List[CandidateSet]] = {d: [] for d in range(1, max_depth + 1)}
    for seq in ordered_subsequences(order, max_depth):
        by_depth[len(seq)].append(enumerate_sequence(seq, U, C, score_fn, cal_idx, grid_sizes))
    out = {}
    pool: List[CandidateSet] = []
    for d in range(1, max_depth + 1):
        pool = pool + by_depth[d]
        out[d] = pareto_frontier(pool)
    return out
