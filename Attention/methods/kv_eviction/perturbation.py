"""Attention-output perturbation for a single query (Lemma 1 of the paper).

These helpers work on one head and one query:
    p : [N]     attention weights, sum to 1
    v : [N, D]  value vectors
They are used by the tests and by the approximation-ratio check, not on the
inference path.
"""

from itertools import combinations

import torch


def shift_closed_form(p, v, evict):
    """Delta(J) = sum_{i in J} p_i (a - v_i) / (1 - P_J)."""
    evict = torch.as_tensor(evict, dtype=torch.long)
    a = p @ v
    num = (p[evict, None] * (a[None, :] - v[evict])).sum(0)
    return num / (1.0 - p[evict].sum())


def shift_recomputed(p, v, evict):
    """Remove the tokens, renormalise the attention, and take the difference."""
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[torch.as_tensor(evict, dtype=torch.long)] = False
    p_kept = p[mask] / p[mask].sum()
    return p_kept @ v[mask] - p @ v


def eviction_loss(p, v, evict):
    """F(J) = ||Delta(J)||_2, the objective the paper wants to minimise."""
    return shift_closed_form(p, v, evict).norm()


def singleton_cost(p, v):
    """||Delta({j})||_2 for every j. This is the DropKV proxy for one query."""
    a = p @ v
    return p / (1.0 - p) * (a[None, :] - v).norm(dim=-1)


def proxy_set(p, v, k, pool=None):
    """The k tokens with the smallest individual cost, restricted to `pool`."""
    cost = singleton_cost(p, v)
    pool = torch.arange(len(p)) if pool is None else torch.as_tensor(pool)
    order = cost[pool].argsort()[:k]
    return pool[order].tolist()


def optimal_set(p, v, k, pool=None):
    """Exact minimiser of F(J) by trying every subset. Exponential, small N only."""
    pool = list(range(len(p))) if pool is None else list(pool)
    best, best_loss = None, float("inf")
    for subset in combinations(pool, k):
        loss = eviction_loss(p, v, list(subset)).item()
        if loss < best_loss:
            best, best_loss = list(subset), loss
    return best, best_loss


def cone_constant(p, v, pool):
    """Lower bound on the cone constant c of Assumption 2 for the given pool,
    using the mean residual direction as u. The true c* (best u) is at least
    this value."""
    a = p @ v
    r = a[None, :] - v[torch.as_tensor(pool)]
    r_unit = r / r.norm(dim=-1, keepdim=True)
    u = r_unit.mean(0)
    u = u / u.norm()
    return (r_unit @ u).min().item()
