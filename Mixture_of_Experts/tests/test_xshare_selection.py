"""Unit tests for Mixture_of_Experts/routing/xshare/selection.py (CPU only)."""

import itertools
import math

import pytest
import torch

from Mixture_of_Experts.routing.xshare import (
    XShareConfig,
    aggregate_scores,
    expert_scores,
    gpu_aware_greedy,
    greedy_select,
    route_within,
    select_batch,
    select_expert_parallel,
    select_per_request,
    select_speculative,
    warmup_set,
    xshare_route,
)

SEEDS = [0, 1, 2, 3]
MODES = ["batch", "speculative", "gpu"]
BUDGET_MODES = ["additional", "total"]
GATE_NORMS = ["topk_softmax", "softmax_topk", "softmax_topk_renorm"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def gen(seed):
    return torch.Generator().manual_seed(seed)


def make_logits(n_tokens, n_experts, seed):
    return torch.randn(n_tokens, n_experts, generator=gen(seed))


def make_valid(n_tokens, seed, p_invalid=0.25):
    """Random valid mask with at least one valid token."""
    valid = torch.rand(n_tokens, generator=gen(seed + 1000)) >= p_invalid
    valid[0] = True
    return valid


def size(mask):
    return int(mask.sum())


def idx_set(mask):
    return set(torch.nonzero(mask).flatten().tolist())


def make_request_ids(n_tokens, seed):
    """Non-contiguous, unsorted request ids."""
    choices = torch.tensor([42, 3, 100, 7])
    return choices[torch.randint(0, 4, (n_tokens,), generator=gen(seed + 2000))]


def make_groups(n_experts, n_groups=4):
    return torch.arange(n_experts) % n_groups


def build(mode, logits, top_k, k0=0, valid=None, request_ids=None, **overrides):
    """Config + xshare_route kwargs for one of the three pruning algorithms."""
    n_tokens, n_experts = logits.shape
    params = dict(top_k=top_k, warmup_k0=k0)
    if mode == "batch":
        params.update(budget=2)
    elif mode == "speculative":
        params.update(per_request_budget=1, budget=1)
    elif mode == "gpu":
        params.update(per_gpu_budget=2)
    else:
        raise AssertionError(mode)
    params.update(overrides)
    if request_ids is None:
        request_ids = (torch.arange(n_tokens) % 3) * 5 + 2
    kwargs = dict(
        valid_mask=valid,
        request_ids=request_ids,
        expert_groups=make_groups(n_experts),
    )
    return XShareConfig(**params), kwargs


def route(mode, logits, top_k, k0=0, valid=None, **overrides):
    cfg, kwargs = build(mode, logits, top_k, k0, valid, **overrides)
    return xshare_route(logits, cfg, **kwargs)


def expected_size(mode, n_init, budget, n_experts):
    if mode == "additional":
        return min(n_init + budget, n_experts)
    return max(n_init, min(budget, n_experts))


def topk_reference(logits, k, gate_norm):
    """Straightforward unpruned top-k routing."""
    if gate_norm == "topk_softmax":
        values, indices = torch.topk(logits, k, dim=-1)
        return indices, torch.softmax(values, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    _, indices = torch.topk(logits, k, dim=-1)
    weights = torch.gather(probs, -1, indices)
    if gate_norm == "softmax_topk_renorm":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights


def best_superset_sum(agg, init, target_size):
    """Brute-force max of sum(agg) over all supersets of init of given size."""
    n = agg.numel()
    free = [i for i in range(n) if not bool(init[i])]
    base = sum(float(agg[i]) for i in range(n) if bool(init[i]))
    need = target_size - size(init)
    return max(base + sum(float(agg[i]) for i in c) for c in itertools.combinations(free, need))


# --------------------------------------------------------------------------
# 1. Greedy optimality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
def test_greedy_select_is_optimal_over_all_supersets(seed, budget_mode):
    """Catches greedy picking non-best experts, or dropping/ignoring S0."""
    g = gen(seed)
    for n in (4, 6, 8):
        agg = torch.randn(n, generator=g)
        init = torch.rand(n, generator=g) < 0.4
        for b in range(0, n + 2):
            out = greedy_select(agg, b, init, budget_mode)
            target = expected_size(budget_mode, size(init), b, n)
            assert size(out) == target
            assert bool((out & init).equal(init)), "init set must be preserved"
            best = best_superset_sum(agg, init, target)
            assert float(agg[out].sum()) == pytest.approx(best, abs=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
def test_select_batch_is_optimal_on_real_scores(seed, budget_mode):
    """Catches select_batch aggregating the wrong axis or counting padded tokens."""
    n_experts, n_tokens = 7, 4
    scores = expert_scores(make_logits(n_tokens, n_experts, seed))
    valid = torch.tensor([True, True, False, True])
    agg = aggregate_scores(scores, valid)
    init = warmup_set(scores, 1, valid)
    for b in range(0, n_experts + 1):
        out = select_batch(scores, b, 1, valid, budget_mode)
        target = expected_size(budget_mode, size(init), b, n_experts)
        assert size(out) == target
        assert float(agg[out].sum()) == pytest.approx(best_superset_sum(agg, init, target), abs=1e-6)


# --------------------------------------------------------------------------
# 2. Warm-up experts always kept
# --------------------------------------------------------------------------


def warmup_experts(logits, k0, valid):
    return torch.topk(logits[valid], k0, dim=-1).indices.flatten()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("k0", [1, 2, 3])
def test_warmup_kept_by_direct_selectors(seed, k0):
    """Catches warm-up experts being evicted or invalid tokens leaking into S0."""
    logits = make_logits(10, 16, seed)
    valid = make_valid(10, seed)
    scores = expert_scores(logits)
    must = warmup_experts(logits, k0, valid)
    rids = make_request_ids(10, seed)
    for mask in (
        select_batch(scores, 2, k0, valid),
        select_batch(scores, 2, k0, valid, "total"),
        select_speculative(scores, rids, 1, 0, k0, valid),
        select_speculative(scores, rids, 0, 2, k0, valid),
        select_expert_parallel(scores, make_groups(16), 1, k0, valid),
    ):
        assert bool(mask[must].all())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k0", [1, 2])
def test_warmup_kept_by_xshare_route(seed, mode, k0):
    """Catches xshare_route dropping warm-up experts (e.g. via min_keep or wiring)."""
    logits = make_logits(10, 16, seed)
    valid = make_valid(10, seed)
    res = route(mode, logits, 2, k0, valid)
    assert bool(res.keep_mask[warmup_experts(logits, k0, valid)].all())


# --------------------------------------------------------------------------
# 3. Budget respected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
@pytest.mark.parametrize("k0,b", [(0, 0), (0, 3), (1, 0), (1, 4), (2, 3), (2, 30)])
def test_batch_budget_exact_size(seed, budget_mode, k0, b):
    """Catches off-by-one in budget handling for both budget modes."""
    n_experts = 16
    logits = make_logits(8, n_experts, seed)
    valid = make_valid(8, seed)
    scores = expert_scores(logits)
    s0 = size(warmup_set(scores, k0, valid))
    expected = expected_size(budget_mode, s0, b, n_experts)
    assert size(select_batch(scores, b, k0, valid, budget_mode)) == expected
    cfg = XShareConfig(top_k=2, warmup_k0=k0, budget=b, budget_mode=budget_mode, min_keep=0)
    res = xshare_route(logits, cfg, valid_mask=valid)
    assert size(res.keep_mask) == expected


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
@pytest.mark.parametrize("k0,g", [(0, 1), (1, 0), (1, 2), (2, 1)])
def test_speculative_global_zero_bounded_by_per_request(seed, budget_mode, k0, g):
    """Catches the per-request stage ignoring its budget or leaking padded tokens."""
    logits = make_logits(12, 16, seed)
    valid = make_valid(12, seed)
    scores = expert_scores(logits)
    rids = make_request_ids(12, seed)
    out = select_speculative(scores, rids, g, 0, k0, valid, budget_mode)
    bound = 0
    for r in torch.unique(rids[valid]).tolist():
        rows = valid & (rids == r)
        w = size(warmup_set(scores[rows], k0))
        bound += w + g if budget_mode == "additional" else max(w, g)
    assert size(out) <= bound


# --------------------------------------------------------------------------
# 4. Identity with the unpruned baseline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_experts,top_k", [(8, 2), (16, 4), (32, 1)])
@pytest.mark.parametrize("budget", [None, "all", "more"])
def test_identity_topk_softmax_exact(seed, n_experts, top_k, budget):
    """Catches any deviation from plain topk+softmax when nothing is pruned."""
    logits = make_logits(9, n_experts, seed)
    b = {None: None, "all": n_experts, "more": n_experts + 5}[budget]
    res = xshare_route(logits, XShareConfig(top_k=top_k, budget=b))
    values, indices = torch.topk(logits, top_k, dim=-1)
    assert torch.equal(res.indices, indices)
    assert torch.equal(res.weights, torch.softmax(values, dim=-1))
    assert bool(res.keep_mask.all())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gate_norm", GATE_NORMS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
def test_identity_all_gate_norms(seed, gate_norm, budget_mode):
    """Catches wrong gate normalisation in any of the three modes."""
    n_experts, top_k = 16, 3
    logits = make_logits(9, n_experts, seed)
    cfg = XShareConfig(top_k=top_k, budget=n_experts, budget_mode=budget_mode, gate_norm=gate_norm)
    res = xshare_route(logits, cfg)
    ref_idx, ref_w = topk_reference(logits, top_k, gate_norm)
    assert torch.equal(res.indices, ref_idx)
    assert torch.allclose(res.weights, ref_w, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("gate_norm", GATE_NORMS)
def test_pruned_softmax_gate_weights_not_renormalised_over_kept(gate_norm):
    """Catches softmax_topk weights changing when other experts are dropped."""
    logits = make_logits(6, 12, 5)
    keep = torch.zeros(12, dtype=torch.bool)
    keep[[0, 3, 5, 7, 9]] = True
    idx, w = route_within(logits, keep, 2, gate_norm)
    if gate_norm == "softmax_topk":
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(w, torch.gather(probs, -1, idx), atol=1e-6)


# --------------------------------------------------------------------------
# 5. Padding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gate_norm", GATE_NORMS)
def test_padding_does_not_change_mask_or_valid_routing(seed, mode, gate_norm):
    """Catches padded tokens influencing aggregation, warm-up, or per-request sets."""
    n, n_pad, n_experts = 8, 5, 16
    logits = make_logits(n, n_experts, seed)
    pad = torch.zeros(n_pad, n_experts)
    pad[torch.arange(n_pad), torch.randint(0, n_experts, (n_pad,), generator=gen(seed))] = 1e4
    pad[:, 0] -= 1e4 * (seed % 2)  # vary the extreme pattern
    padded = torch.cat([logits, pad])
    rids = make_request_ids(n, seed)
    rids_padded = torch.cat([rids, torch.full((n_pad,), 999)])
    valid_padded = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(n_pad, dtype=torch.bool)])

    base = route(mode, logits, 2, 1, None, request_ids=rids, gate_norm=gate_norm)
    with_pad = route(mode, padded, 2, 1, valid_padded, request_ids=rids_padded, gate_norm=gate_norm)
    assert torch.equal(base.keep_mask, with_pad.keep_mask)
    assert torch.equal(base.indices, with_pad.indices[:n])
    assert torch.allclose(base.weights, with_pad.weights[:n], atol=1e-6)
    assert with_pad.indices.shape == (n + n_pad, 2)  # padding is still routed


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_all_invalid_batch_keeps_every_expert(seed, mode):
    """Catches an empty/garbage keep set when there is nothing to select from."""
    logits = make_logits(6, 16, seed)
    valid = torch.zeros(6, dtype=torch.bool)
    res = route(mode, logits, 2, 1, valid)
    assert bool(res.keep_mask.all())
    assert res.diagnostics["n_tokens"] == 0
    assert res.diagnostics["activated_experts"] == 0


# --------------------------------------------------------------------------
# 6. Routing stays inside the set
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gate_norm", GATE_NORMS)
def test_routed_experts_inside_set_and_weights_valid(seed, mode, gate_norm):
    """Catches routing outside S, negative weights, or weights not summing to one."""
    top_k = 3
    logits = make_logits(10, 16, seed)
    res = route(mode, logits, top_k, 1, make_valid(10, seed), gate_norm=gate_norm)
    assert size(res.keep_mask) >= top_k
    assert bool((res.weights >= 0).all())
    inside = res.keep_mask[res.indices]
    assert bool(inside[res.weights > 0].all())
    sums = res.weights.sum(dim=-1)
    if gate_norm == "softmax_topk":
        assert bool((sums <= 1 + 1e-5).all())
    else:
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gate_norm", GATE_NORMS)
@pytest.mark.parametrize("n_kept", [1, 2, 3])
def test_route_within_fewer_kept_than_top_k_has_zero_surplus_weight(seed, gate_norm, n_kept):
    """Catches surplus slots carrying nonzero weight (or NaN) when |S| < top_k."""
    top_k, n_experts = 4, 12
    logits = make_logits(7, n_experts, seed)
    keep = torch.zeros(n_experts, dtype=torch.bool)
    keep[torch.randperm(n_experts, generator=gen(seed))[:n_kept]] = True
    idx, w = route_within(logits, keep, top_k, gate_norm)
    assert not bool(torch.isnan(w).any())
    assert bool((w[:, n_kept:] == 0).all())
    assert bool((w[:, :n_kept] > 0).all())
    assert bool(keep[idx[:, :n_kept]].all())
    if gate_norm != "softmax_topk":
        assert torch.allclose(w.sum(-1), torch.ones(7), atol=1e-5)


@pytest.mark.parametrize("gate_norm", GATE_NORMS)
def test_route_within_empty_keep_mask_gives_zero_weights(gate_norm):
    """Docstring promises zero weight on dropped experts; empty S must not give NaN."""
    logits = make_logits(4, 8, 0)
    idx, w = route_within(logits, torch.zeros(8, dtype=torch.bool), 2, gate_norm)
    assert not bool(torch.isnan(w).any())
    assert bool((w == 0).all())


# --------------------------------------------------------------------------
# 7. min_keep
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_min_keep_default_tops_up_to_top_k(seed):
    """Catches min_keep=None not defaulting to top_k."""
    logits = make_logits(8, 16, seed)
    cfg = XShareConfig(top_k=4, warmup_k0=0, budget=1)
    res = xshare_route(logits, cfg)
    assert size(res.keep_mask) >= 4
    agg = aggregate_scores(expert_scores(logits))
    assert idx_set(res.keep_mask) == set(torch.topk(agg, 4).indices.tolist())


@pytest.mark.parametrize("seed", SEEDS)
def test_min_keep_zero_disables_top_up(seed):
    """Catches min_keep=0 still padding the set."""
    logits = make_logits(8, 16, seed)
    cfg = XShareConfig(top_k=4, warmup_k0=0, budget=1, min_keep=0)
    res = xshare_route(logits, cfg)
    assert size(res.keep_mask) == 1
    only = int(torch.nonzero(res.keep_mask))
    assert bool((res.indices[:, 0] == only).all())
    assert torch.allclose(res.weights[:, 0], torch.ones(8))
    assert bool((res.weights[:, 1:] == 0).all())


@pytest.mark.parametrize("min_keep", [0, 3, 6, 100])
def test_min_keep_explicit_value(min_keep):
    """Catches min_keep ignored or not clipped to the number of experts."""
    logits = make_logits(8, 16, 1)
    cfg = XShareConfig(top_k=2, budget=1, min_keep=min_keep)
    assert size(xshare_route(logits, cfg).keep_mask) == min(max(min_keep, 1), 16)


# --------------------------------------------------------------------------
# 8. Algorithm 5: GPU-aware greedy
# --------------------------------------------------------------------------


def shuffled_equal_groups(n_experts, n_groups, seed):
    groups = torch.empty(n_experts, dtype=torch.long)
    perm = torch.randperm(n_experts, generator=gen(seed))
    groups[perm] = torch.arange(n_experts) % n_groups
    return groups


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n,G,m", [(12, 3, 2), (12, 4, 3), (16, 4, 1), (8, 2, 5), (16, 2, 0)])
def test_gpu_greedy_balanced_and_best_per_group(seed, n, G, m):
    """Catches unbalanced loads, wrong total size, or non-best experts in a group."""
    agg = torch.randn(n, generator=gen(seed))
    groups = shuffled_equal_groups(n, G, seed)
    keep = gpu_aware_greedy(agg, groups, m)
    total = min(m * G, n)
    assert size(keep) == total
    loads = torch.bincount(groups[keep], minlength=G)
    assert int(loads.max()) <= math.ceil(total / G)
    for gid in range(G):
        members = groups == gid
        chosen, rest = members & keep, members & ~keep
        if bool(chosen.any()) and bool(rest.any()):
            assert float(agg[chosen].min()) >= float(agg[rest].max())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "groups,m",
    [
        ([0] + [1] * 7, 3),
        ([0, 0, 1, 2, 2, 2, 2, 2], 3),
        ([0, 1, 1, 1, 1, 1, 1, 1], 100),
        ([5, 5, 9, 9, 9, 2], 2),
    ],
)
def test_gpu_greedy_unequal_groups_terminates(seed, groups, m):
    """Catches infinite loops / over-selection when a group runs out of experts."""
    groups = torch.tensor(groups)
    n = groups.numel()
    agg = torch.randn(n, generator=gen(seed))
    keep = gpu_aware_greedy(agg, groups, m)
    n_groups = len(torch.unique(groups))
    assert size(keep) == min(m * n_groups, n)
    assert size(keep) <= n


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m", [0, 1, 2, 4])
def test_gpu_greedy_preserves_init_mask(seed, m):
    """Catches init experts being dropped or the set shrinking below init."""
    n, G = 12, 3
    agg = torch.randn(n, generator=gen(seed))
    groups = make_groups(n, G)
    init = torch.rand(n, generator=gen(seed + 7)) < 0.5
    keep = gpu_aware_greedy(agg, groups, m, init)
    assert torch.equal(keep & init, init)
    assert size(keep) == max(size(init), min(m * G, n))


def test_gpu_greedy_does_not_mutate_init():
    """Catches in-place modification of the caller's init mask."""
    agg = torch.randn(8, generator=gen(0))
    init = torch.zeros(8, dtype=torch.bool)
    init[2] = True
    before = init.clone()
    gpu_aware_greedy(agg, make_groups(8, 2), 3, init)
    greedy_select(agg, 2, init)
    assert torch.equal(init, before)


# --------------------------------------------------------------------------
# 9. Algorithm 4
# --------------------------------------------------------------------------


def union_of_per_request(scores, rids, valid, g, k0, mode):
    keep = torch.zeros(scores.shape[1], dtype=torch.bool)
    for r in torch.unique(rids[valid]).tolist():
        rows = valid & (rids == r)
        keep |= select_per_request(scores[rows], g, k0, mode)
    return keep


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
@pytest.mark.parametrize("k0,g", [(0, 2), (1, 1), (2, 3), (1, 0)])
def test_speculative_equals_union_of_per_request(seed, budget_mode, k0, g):
    """Catches mixing tokens across requests or an unwanted global stage."""
    scores = expert_scores(make_logits(12, 16, seed))
    rids = make_request_ids(12, seed)
    valid = make_valid(12, seed)
    out = select_speculative(scores, rids, g, 0, k0, valid, budget_mode)
    assert torch.equal(out, union_of_per_request(scores, rids, valid, g, k0, budget_mode))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
def test_speculative_single_request_equals_per_request(seed, budget_mode):
    """Catches id handling that breaks for a lone (non-zero) request id."""
    scores = expert_scores(make_logits(5, 16, seed))
    rids = torch.full((5,), 37)
    out = select_speculative(scores, rids, 3, 0, 1, None, budget_mode)
    assert torch.equal(out, select_per_request(scores, 3, 1, budget_mode))


@pytest.mark.parametrize("seed", SEEDS)
def test_per_request_without_warmup_is_topg_of_summed_scores(seed):
    """Independent reference for Algorithm 3: top-g of the summed scores."""
    scores = expert_scores(make_logits(4, 16, seed))
    out = select_per_request(scores, 5, 0)
    assert idx_set(out) == set(torch.topk(scores.sum(0), 5).indices.tolist())


@pytest.mark.parametrize("seed", SEEDS)
def test_speculative_global_stage_adds_best_remaining(seed):
    """Catches the global stage not running after the per-request union."""
    scores = expert_scores(make_logits(9, 16, seed))
    rids = make_request_ids(9, seed)
    base = select_speculative(scores, rids, 1, 0, 1)
    more = select_speculative(scores, rids, 1, 3, 1)
    assert size(more) == min(size(base) + 3, 16)
    assert torch.equal(more & base, base)


# --------------------------------------------------------------------------
# 10. Determinism and ties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_xshare_route_deterministic(seed, mode):
    """Catches nondeterministic selection or routing."""
    logits = make_logits(10, 16, seed)
    a = route(mode, logits, 2, 1)
    b = route(mode, logits, 2, 1)
    assert torch.equal(a.keep_mask, b.keep_mask)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.weights, b.weights)


def test_ties_go_to_lower_index():
    """Catches unstable tie-breaking in greedy, warm-up, and GPU-aware selection."""
    agg = torch.ones(8)
    assert idx_set(greedy_select(agg, 3)) == {0, 1, 2}
    init = torch.zeros(8, dtype=torch.bool)
    init[5] = True
    assert idx_set(greedy_select(agg, 2, init)) == {5, 0, 1}
    assert idx_set(greedy_select(agg, 4, init, "total")) == {5, 0, 1, 2}

    scores = expert_scores(torch.zeros(3, 8))
    assert idx_set(warmup_set(scores, 2)) == {0, 1}
    assert idx_set(select_batch(scores, 3)) == {0, 1, 2}
    assert idx_set(select_batch(scores, 2, 2)) == {0, 1, 2, 3}

    keep = gpu_aware_greedy(agg, make_groups(8, 2), 2)
    assert idx_set(keep) == {0, 1, 2, 3}


# --------------------------------------------------------------------------
# 11. Diagnostics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_diagnostics_consistent(seed, mode):
    """Catches wrong counts, load, or mass in the diagnostics dict."""
    top_k, n_experts = 3, 16
    logits = make_logits(11, n_experts, seed)
    valid = make_valid(11, seed)
    res = route(mode, logits, top_k, 1, valid)
    d = res.diagnostics
    n_valid = size(valid)
    assert d["n_tokens"] == n_valid
    assert d["n_experts"] == n_experts
    assert d["kept_experts"] == size(res.keep_mask)
    assert size(res.keep_mask) >= top_k
    unique_routed = torch.unique(res.indices[valid]).numel()
    assert d["activated_experts"] == unique_routed
    assert d["activated_experts"] <= d["kept_experts"]
    assert int(d["expert_load"].sum()) == n_valid * top_k
    assert d["expert_load"].shape == (n_experts,)
    assert 0.0 <= d["score_mass_kept"] <= 1.0 + 1e-6

    agg = aggregate_scores(expert_scores(logits), valid)
    expected_mass = float(agg[res.keep_mask].sum() / agg.sum())
    assert d["score_mass_kept"] == pytest.approx(expected_mass, abs=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_diagnostics_mass_is_one_when_nothing_pruned(seed):
    """Catches score_mass_kept != 1 for the unpruned baseline."""
    res = xshare_route(make_logits(6, 16, seed), XShareConfig(top_k=2))
    assert res.diagnostics["score_mass_kept"] == pytest.approx(1.0, abs=1e-6)
    assert res.diagnostics["kept_experts"] == 16


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_diagnostics_per_group(seed, mode):
    """Catches per-group fields missing or inconsistent with global counts."""
    logits = make_logits(11, 16, seed)
    res = route(mode, logits, 2, 1, make_valid(11, seed))
    d = res.diagnostics
    groups = make_groups(16)
    for key in ("kept_per_group", "activated_per_group", "max_kept_per_group", "max_activated_per_group"):
        assert key in d
    assert int(d["kept_per_group"].sum()) == d["kept_experts"]
    assert int(d["activated_per_group"].sum()) == d["activated_experts"]
    assert d["max_kept_per_group"] == int(d["kept_per_group"].max())
    assert d["max_activated_per_group"] == int(d["activated_per_group"].max())
    expected = torch.bincount(groups[res.keep_mask], minlength=4)
    assert torch.equal(d["kept_per_group"], expected)
    assert bool((d["activated_per_group"] <= d["kept_per_group"]).all())


def test_diagnostics_without_groups_has_no_group_fields_and_can_be_disabled():
    """Catches group fields appearing without groups, or with_diagnostics ignored."""
    logits = make_logits(4, 8, 0)
    res = xshare_route(logits, XShareConfig(top_k=2, budget=2))
    assert "kept_per_group" not in res.diagnostics
    off = xshare_route(logits, XShareConfig(top_k=2, budget=2), with_diagnostics=False)
    assert off.diagnostics == {}


# --------------------------------------------------------------------------
# 12. Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(top_k=0),
        dict(top_k=-1),
        dict(top_k=2, warmup_k0=-1),
        dict(top_k=2, budget=-1),
        dict(top_k=2, per_request_budget=-1),
        dict(top_k=2, per_gpu_budget=-1),
        dict(top_k=2, min_keep=-1),
        dict(top_k=2, budget_mode="bogus"),
        dict(top_k=2, score="bogus"),
        dict(top_k=2, gate_norm="bogus"),
    ],
)
def test_config_rejects_bad_values(kwargs):
    """Catches missing validation in XShareConfig.__post_init__."""
    with pytest.raises(ValueError):
        XShareConfig(**kwargs)


def test_config_accepts_zero_budgets():
    """Catches over-strict validation rejecting zero."""
    cfg = XShareConfig(top_k=1, warmup_k0=0, budget=0, per_request_budget=0, per_gpu_budget=0, min_keep=0)
    assert cfg.prunes


def test_missing_request_ids_or_groups_raise():
    """Catches modes that silently run without their required side input."""
    logits = make_logits(4, 8, 0)
    with pytest.raises(ValueError):
        xshare_route(logits, XShareConfig(top_k=2, per_request_budget=1))
    with pytest.raises(ValueError):
        xshare_route(logits, XShareConfig(top_k=2, per_gpu_budget=1))


def test_wrong_shapes_raise():
    """Catches missing shape checks on logits, masks, ids, and groups."""
    logits = make_logits(4, 8, 0)
    cfg = XShareConfig(top_k=2, budget=1)
    with pytest.raises(ValueError):
        xshare_route(logits[0], cfg)
    with pytest.raises(ValueError):
        xshare_route(logits.unsqueeze(0), cfg)
    with pytest.raises(ValueError):
        xshare_route(logits, cfg, valid_mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        xshare_route(
            logits,
            XShareConfig(top_k=2, per_request_budget=1),
            request_ids=torch.zeros(3, dtype=torch.long),
        )
    with pytest.raises(ValueError):
        xshare_route(
            logits,
            XShareConfig(top_k=2, per_gpu_budget=1),
            expert_groups=torch.zeros(7, dtype=torch.long),
        )
    scores = expert_scores(logits)
    with pytest.raises(ValueError):
        select_speculative(scores, torch.zeros(5, dtype=torch.long), 1)
    with pytest.raises(ValueError):
        gpu_aware_greedy(torch.randn(8), torch.zeros(9, dtype=torch.long), 1)
    with pytest.raises(ValueError):
        route_within(logits, torch.ones(7, dtype=torch.bool), 2)
    with pytest.raises(ValueError):
        route_within(logits, None, 9)
    with pytest.raises(ValueError):
        xshare_route(logits, XShareConfig(top_k=9))
    with pytest.raises(ValueError):
        greedy_select(torch.randn(8), 1, None, "bogus")
    with pytest.raises(ValueError):
        expert_scores(logits, "bogus")


# --------------------------------------------------------------------------
# 13. OEA special case
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("top_k,k0", [(2, 2), (3, 3), (2, 3), (4, 4)])
def test_oea_budget_zero_keeps_union_of_per_token_topk0(seed, top_k, k0):
    """Catches budget=0 adding or removing experts beyond the warm-up union."""
    logits = make_logits(8, 24, seed)
    union = torch.zeros(24, dtype=torch.bool)
    union[torch.topk(logits, k0, dim=-1).indices.flatten()] = True
    assert size(union) >= top_k  # holds because k0 >= top_k for these params
    res = xshare_route(logits, XShareConfig(top_k=top_k, warmup_k0=k0, budget=0))
    assert torch.equal(res.keep_mask, union)
    res0 = xshare_route(logits, XShareConfig(top_k=top_k, warmup_k0=k0, budget=0, min_keep=0))
    assert torch.equal(res0.keep_mask, union)


@pytest.mark.parametrize("seed", SEEDS)
def test_oea_small_k0_union_without_min_keep(seed):
    """Catches OEA union being wrong when k0 < top_k and min_keep is off."""
    logits = make_logits(8, 24, seed)
    union = torch.zeros(24, dtype=torch.bool)
    union[torch.topk(logits, 1, dim=-1).indices.flatten()] = True
    res = xshare_route(logits, XShareConfig(top_k=4, warmup_k0=1, budget=0, min_keep=0))
    assert torch.equal(res.keep_mask, union)


# --------------------------------------------------------------------------
# 14. Monotonicity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("budget_mode", BUDGET_MODES)
@pytest.mark.parametrize("k0", [0, 1, 2])
def test_keep_set_and_mass_monotone_in_budget(seed, budget_mode, k0):
    """Catches non-nested selections or mass decreasing as the budget grows."""
    n_experts = 16
    logits = make_logits(8, n_experts, seed)
    valid = make_valid(8, seed)
    prev = None
    for b in range(0, n_experts + 3):
        cfg = XShareConfig(top_k=2, warmup_k0=k0, budget=b, budget_mode=budget_mode)
        res = xshare_route(logits, cfg, valid_mask=valid)
        if prev is not None:
            assert not bool((prev.keep_mask & ~res.keep_mask).any()), f"not nested at b={b}"
            assert res.diagnostics["score_mass_kept"] >= prev.diagnostics["score_mass_kept"] - 1e-6
        prev = res
    assert bool(prev.keep_mask.all())
    assert prev.diagnostics["score_mass_kept"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_select_batch_nested_directly(seed):
    """Catches nesting violations in select_batch without min_keep involved."""
    scores = expert_scores(make_logits(6, 12, seed))
    masks = [select_batch(scores, b, 1) for b in range(14)]
    for lo, hi in zip(masks, masks[1:]):
        assert not bool((lo & ~hi).any())
