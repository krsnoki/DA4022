"""Tree construction / masking tests against the upstream implementation (CPU)."""
from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from conftest import COST_GPU, COST_MODEL  # noqa: E402

from bastion import tree_draft  # noqa: E402


def _build(logits, n_min, n_max, draft=0.004, nonverify=0.002, ctx=128):
    return tree_draft.build_adaptive_best_tree_from_draft_logits(
        root_token_id=torch.tensor(7),
        draft_logits=logits,
        context_prefix_length=ctx,
        draft_latency_s=draft,
        non_verify_latency_s=nonverify,
        cost_model_name=COST_MODEL,
        cost_gpu_type=COST_GPU,
        min_tree_size=n_min,
        max_tree_size=n_max,
    )


def _ancestors(parents, i):
    out = {i}
    while parents[i] >= 0:
        i = parents[i]
        out.add(i)
    return out


def _path_scores(tree, logits):
    """rho(i) = prod of drafter probabilities along the root-to-node path."""
    logp = torch.log_softmax(logits[0], dim=-1).float()  # float32, as upstream ranks nodes
    parents = tree.parent_indices.tolist()
    toks = tree.token_ids.tolist()
    depths = tree.depths.tolist()
    rho = []
    for i in range(len(toks)):
        lp, j = 0.0, i
        while parents[j] >= 0:
            lp += float(logp[depths[j] - 1, toks[j]])
            j = parents[j]
        rho.append(math.exp(lp))
    return rho


def test_tree_mask_is_ancestor_relation():
    parents = [-1, 0, 0, 1, 1, 3, 2]
    depths = [0, 1, 1, 2, 2, 3, 2]
    tree = tree_draft.finalize_tree_draft(list(range(7)), parents, depths, device=torch.device("cpu"))
    for i in range(7):
        anc = _ancestors(parents, i)
        for j in range(7):
            assert bool(tree.tree_mask[i, j]) == (j in anc), (i, j)
    # Every leaf has a retrieve path equal to its root-to-leaf chain.
    for row, plen in zip(tree.retrieve_indices.tolist(), tree.path_lens.tolist()):
        path = row[:plen]
        assert path[0] == 0
        for a, b in zip(path, path[1:]):
            assert parents[b] == a


def test_tree_attention_mask_layout():
    parents = [-1, 0, 0, 1]
    tree = tree_draft.finalize_tree_draft([1, 2, 3, 4], parents, [0, 1, 1, 2], device=torch.device("cpu"))
    prefix = 5
    attn = tree_draft._build_tree_attention_mask(tree.tree_mask, prefix, torch.float32)
    assert attn.shape == (1, 1, 4, prefix + 4)
    neg = torch.finfo(torch.float32).min
    assert torch.all(attn[0, 0, :, :prefix] == 0)  # every node sees the whole prefix
    for i in range(4):
        for j in range(4):
            expect = 0.0 if j in _ancestors(parents, i) else neg
            assert float(attn[0, 0, i, prefix + j]) == expect


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_first_tree_prefix_closed_and_bounded(calib_env, seed):
    calib_env()
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 15, 200, generator=g) * 3.0
    for n_min, n_max in [(32, 8192), (32, 64), (16, 512)]:
        tree = _build(logits, n_min, n_max)
        n = tree.token_ids.numel()
        assert n_min <= n <= n_max
        parents, depths = tree.parent_indices.tolist(), tree.depths.tolist()
        assert parents[0] == -1 and depths[0] == 0
        for i in range(1, n):
            assert 0 <= parents[i] < i, "parent must be present and precede the child"
            assert depths[i] == depths[parents[i]] + 1
        # Path scores are non-increasing in insertion order (best-first, Prop. 2.3 / eq. 5).
        rho = _path_scores(tree, logits)
        assert all(rho[i + 1] <= rho[i] * (1 + 1e-9) for i in range(n - 1))


def test_trees_nested_as_budget_grows(calib_env):
    calib_env()
    logits = torch.randn(1, 15, 300, generator=torch.Generator().manual_seed(5)) * 2.0
    prev = None
    for n in (32, 64, 128, 256, 512, 1024):
        tree = _build(logits, n, n)
        assert tree.token_ids.numel() == n
        if prev is not None:
            k = prev.token_ids.numel()
            assert torch.equal(tree.token_ids[:k], prev.token_ids)
            assert torch.equal(tree.parent_indices[:k], prev.parent_indices)
        prev = tree


def test_best_first_is_optimal_on_tiny_lattice(calib_env):
    """Prop. 2.3: best-first tree of size N maximises sum of rho over prefix-closed trees."""
    calib_env()
    depth, vocab, n = 2, 3, 5
    logits = torch.tensor([[[2.0, 1.0, -1.0], [0.5, 0.4, 0.3]]])
    tree = _build(logits, n, n)
    best_first = sum(_path_scores(tree, logits))
    probs = torch.softmax(logits[0].double(), -1)
    nodes = [p for d in range(1, depth + 1) for p in itertools.product(range(vocab), repeat=d)]
    best = 0.0
    for subset in itertools.combinations(nodes, n - 1):
        s = set(subset)
        if any(len(p) > 1 and p[:-1] not in s for p in s):
            continue
        val = 1.0 + sum(math.prod(float(probs[k, t]) for k, t in enumerate(p)) for p in s)
        best = max(best, val)
    assert best_first == pytest.approx(best, rel=1e-6)  # float32 vs float64 path scores


def test_fixed_budget_respected_even_with_flat_logits(calib_env):
    """min=max=N forces exactly N nodes (stopping rule disabled below min, terminate at max)."""
    calib_env()
    for logits in (torch.zeros(1, 15, 128), torch.randn(1, 15, 128) * 10):
        for n in (32, 100, 1024):
            assert _build(logits, n, n).token_ids.numel() == n


def test_env_override_would_break_fixed_budget(calib_env, monkeypatch):
    """ADAPTIVE_MIN_TREE_SIZE silently replaces min_tree_size upstream; the harness refuses it."""
    import repro_lib

    calib_env()
    monkeypatch.setenv("ADAPTIVE_MIN_TREE_SIZE", "8")
    tree = _build(torch.randn(1, 15, 64) * 0.1, 64, 64)
    assert tree.token_ids.numel() <= 64  # no longer guaranteed to be 64
    with pytest.raises(RuntimeError):
        repro_lib.check_method_env()
