"""Evaluation metrics used in the paper (Sec. 4 and Appendix E).

* pass@k — unbiased estimator of Chen et al. (2021) / Kulal et al. (2019).
* pass@NFE — pass rate as a function of the cumulative NFEs spent on k
  samples (Sec. 4.1, after Shen et al. 2026). "Both rows plot the same data;
  only the metric changes" (Fig. 3): the x-coordinate for k samples is the
  expected cumulative NFE of k samples.
* Best@k / Best@NFE — accuracy of a single selected answer out of k samples,
  selected by self-consistency (majority vote; Wang et al., 2023) or by an
  outcome reward model (ORM) (Sec. 4.2, Fig. 4 and 9).
* Entropy@k — mean empirical entropy (nats) of the per-question answer
  distribution over k samples (Fig. 10).
* GRPO group reward statistics (Sec. 4.3, Fig. 5).

Pure Python (no NumPy dependency). Monte Carlo estimates use a seeded
``random.Random`` so results are reproducible.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Dict, Hashable, List, Optional, Sequence

__all__ = [
    "pass_at_k",
    "pass_at_k_curve",
    "pass_at_nfe_curve",
    "pass_within_budget",
    "majority_vote",
    "best_at_k",
    "best_at_nfe_curve",
    "answer_entropy",
    "entropy_at_k",
    "group_reward_stats",
]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: ``1 − C(n−c, k) / C(n, k)`` for ``n`` samples, ``c`` correct."""
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    if n - c < k:
        return 1.0
    # numerically stable product form
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_at_k_curve(correct: Sequence[Sequence[bool]], ks: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """Mean pass@k over problems. ``correct[i]`` lists sample outcomes of problem i."""
    n_min = min(len(c) for c in correct)
    ks = list(ks) if ks is not None else _powers_of_two(n_min)
    return {k: sum(pass_at_k(len(c), sum(map(bool, c)), k) for c in correct) / len(correct) for k in ks}


def pass_at_nfe_curve(
    correct: Sequence[Sequence[bool]],
    nfes: Sequence[Sequence[float]],
    ks: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    """pass@NFE: pairs (expected cumulative NFE of k samples, pass@k).

    For problem ``i`` with per-sample NFEs ``nfes[i]``, k samples drawn without
    replacement cost ``k · mean(nfes[i])`` NFEs in expectation; the curve
    reports the mean over problems. Returns a list of dicts with keys ``k``,
    ``nfe`` (mean cumulative NFEs) and ``pass``.
    """
    if len(correct) != len(nfes):
        raise ValueError("correct and nfes must have the same number of problems")
    curve = pass_at_k_curve(correct, ks)
    out = []
    for k, p in curve.items():
        cum = sum(k * (sum(f) / len(f)) for f in nfes) / len(nfes)
        out.append({"k": k, "nfe": cum, "pass": p})
    return out


def pass_within_budget(
    correct: Sequence[Sequence[bool]],
    nfes: Sequence[Sequence[float]],
    budgets: Sequence[float],
    num_trials: int = 1000,
    seed: int = 0,
) -> Dict[float, float]:
    """Alternative budgeted view: draw samples (random order, w/o replacement)
    until the next one would exceed the NFE budget; pass if any drawn sample
    is correct. Monte Carlo over orderings."""
    rng = random.Random(seed)
    out = {}
    for B in budgets:
        tot = 0.0
        for c, f in zip(correct, nfes):
            hits = 0
            idx = list(range(len(c)))
            for _ in range(num_trials):
                rng.shuffle(idx)
                spent, ok = 0.0, False
                for j in idx:
                    if spent + f[j] > B:
                        break
                    spent += f[j]
                    if c[j]:
                        ok = True
                        break
                hits += ok
            tot += hits / num_trials
        out[B] = tot / len(correct)
    return out


def majority_vote(answers: Sequence[Optional[Hashable]], rng: Optional[random.Random] = None) -> Optional[Hashable]:
    """Most common non-``None`` answer; ties broken uniformly at random."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    top = max(counts.values())
    winners = [a for a, n in counts.items() if n == top]
    return winners[0] if len(winners) == 1 else (rng or random).choice(winners)


def best_at_k(
    answers: Sequence[Sequence[Optional[Hashable]]],
    references: Sequence[Hashable],
    k: int,
    method: str = "self_consistency",
    scores: Optional[Sequence[Sequence[float]]] = None,
    correct: Optional[Sequence[Sequence[bool]]] = None,
    num_trials: int = 200,
    seed: int = 0,
) -> float:
    """Best@k: accuracy of the answer selected from k random samples.

    Args:
        answers: extracted final answers per problem and sample (``None`` = no answer).
        references: ground-truth answer per problem (compared with ``==`` unless
            ``correct`` is given).
        k: subset size (drawn without replacement, Monte Carlo over subsets).
        method: ``"self_consistency"`` (majority vote) or ``"orm"`` (highest score).
        scores: ORM scores, required for ``method="orm"``.
        correct: optional precomputed per-sample correctness (overrides ``==``).
    """
    if method not in ("self_consistency", "orm"):
        raise ValueError("method must be 'self_consistency' or 'orm'")
    if method == "orm" and scores is None:
        raise ValueError("ORM selection needs scores")
    rng = random.Random(seed)
    total = 0.0
    for i, ans in enumerate(answers):
        n = len(ans)
        if not 1 <= k <= n:
            raise ValueError(f"k={k} must be in [1, {n}]")
        exhaustive = math.comb(n, k) <= num_trials
        subsets = (
            [list(s) for s in itertools.combinations(range(n), k)]
            if exhaustive
            else [rng.sample(range(n), k) for _ in range(num_trials)]
        )
        hits = 0.0
        for sub in subsets:
            if method == "orm":
                best = max(sub, key=lambda j: (scores[i][j], -j))
                ok = correct[i][best] if correct is not None else ans[best] == references[i]
            else:
                choice = majority_vote([ans[j] for j in sub], rng)
                if choice is None:
                    ok = False
                elif correct is not None:
                    ok = any(correct[i][j] for j in sub if ans[j] == choice)
                else:
                    ok = choice == references[i]
            hits += bool(ok)
        total += hits / len(subsets)
    return total / len(answers)


def best_at_nfe_curve(
    answers: Sequence[Sequence[Optional[Hashable]]],
    references: Sequence[Hashable],
    nfes: Sequence[Sequence[float]],
    ks: Optional[Sequence[int]] = None,
    **kwargs,
) -> List[Dict[str, float]]:
    """Best@NFE: (expected cumulative NFE of k samples, Best@k) pairs."""
    n_min = min(len(a) for a in answers)
    ks = list(ks) if ks is not None else _powers_of_two(n_min)
    out = []
    for k in ks:
        cum = sum(k * (sum(f) / len(f)) for f in nfes) / len(nfes)
        out.append({"k": k, "nfe": cum, "best": best_at_k(answers, references, k, **kwargs)})
    return out


def answer_entropy(answers: Sequence[Optional[Hashable]]) -> float:
    """Empirical entropy (nats) of an answer multiset (``None`` counted as "invalid")."""
    counts = Counter("<invalid>" if a is None else a for a in answers)
    n = sum(counts.values())
    return -sum((m / n) * math.log(m / n) for m in counts.values())


def entropy_at_k(
    answers: Sequence[Sequence[Optional[Hashable]]], k: int, num_trials: int = 200, seed: int = 0
) -> float:
    """Entropy@k (Fig. 10): mean over questions of the answer entropy of k random samples."""
    rng = random.Random(seed)
    tot = 0.0
    for ans in answers:
        if k == len(ans):
            tot += answer_entropy(ans)
            continue
        tot += sum(answer_entropy([ans[j] for j in rng.sample(range(len(ans)), k)]) for _ in range(num_trials)) / num_trials
    return tot / len(answers)


def group_reward_stats(rewards: Sequence[Sequence[float]]) -> Dict[str, float]:
    """GRPO diagnostics (Fig. 5): mean reward, mean within-group std (population),
    and the fraction of groups with zero variance (no learning signal)."""
    means, stds, zero = [], [], 0
    for g in rewards:
        m = sum(g) / len(g)
        s = math.sqrt(sum((r - m) ** 2 for r in g) / len(g))
        means.append(m)
        stds.append(s)
        zero += s == 0
    return {
        "reward_mean": sum(means) / len(means),
        "reward_std": sum(stds) / len(stds),
        "frac_zero_std_groups": zero / len(rewards),
    }


def _powers_of_two(n: int) -> List[int]:
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if ks[-1] != n:
        ks.append(n)
    return ks
