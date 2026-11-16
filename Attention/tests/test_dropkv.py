"""CPU tests for DropKV scoring and cache eviction.

Run from the repository root:
    python -m pytest Attention/tests/test_dropkv.py -q
No model download is needed; the end-to-end tests use a tiny randomly
initialised decoder.
"""

import pytest
import torch

from Attention.methods.kv_eviction import perturbation as pert
from Attention.methods.kv_eviction.scoring import (
    attention_probs,
    dropkv_scores,
    dropkv_scores_naive,
    gather_positions,
    select_keep_indices,
    snapkv_scores,
)


def random_qkv(b=2, h=3, w=4, n=40, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, w, d, generator=g)
    k = torch.randn(b, h, n, d, generator=g)
    v = torch.randn(b, h, n, d, generator=g)
    return q, k, v


def random_pv(n=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(n, generator=g, dtype=torch.float64), dim=0)
    v = torch.randn(n, d, generator=g, dtype=torch.float64)
    return p, v


# ---------------------------------------------------------------- Lemma 1

@pytest.mark.parametrize("evict", [[3], [0, 5], [1, 2, 7, 9]])
def test_lemma1_closed_form_matches_recomputation(evict):
    p, v = random_pv()
    closed = pert.shift_closed_form(p, v, evict)
    direct = pert.shift_recomputed(p, v, evict)
    assert torch.allclose(closed, direct, atol=1e-10)


def test_single_query_score_is_squared_singleton_shift():
    # with one query, score_j must equal ||Delta({j})||^2 obtained by really
    # removing token j and recomputing the attention output
    q, k, v = random_qkv(b=1, h=1, w=1, n=10, d=8)
    q, k, v = q.double(), k.double(), v.double()
    p = torch.softmax(k[0, 0] @ q[0, 0, 0] / 8**0.5, dim=0)
    expected = torch.stack(
        [pert.shift_recomputed(p, v[0, 0], [j]).norm() ** 2 for j in range(10)]
    )
    got = dropkv_scores(q, k, v, eps=0.0)[0, 0]
    assert torch.allclose(got.double(), expected, rtol=1e-3, atol=1e-7)


# ---------------------------------------------------------------- scoring

def test_score_shape_dtype_and_sign():
    q, k, v = random_qkv()
    s = dropkv_scores(q, k, v)
    assert s.shape == (2, 3, 40)
    assert s.dtype == torch.float32
    assert torch.isfinite(s).all() and (s >= 0).all()


def test_fast_path_matches_naive():
    q, k, v = random_qkv()
    assert torch.allclose(dropkv_scores(q, k, v), dropkv_scores_naive(q, k, v), rtol=1e-4, atol=1e-6)


def test_half_precision_inputs_are_scored_in_float32():
    q, k, v = random_qkv()
    s16 = dropkv_scores(q.half(), k.half(), v.half())
    assert s16.dtype == torch.float32
    assert torch.allclose(s16, dropkv_scores(q, k, v), rtol=5e-2, atol=1e-3)


def test_causal_mask_inside_window():
    # the first window query sits at position n - w and must not see later keys
    q, k, _ = random_qkv(w=4, n=20)
    p = attention_probs(q, k)
    assert torch.all(p[:, :, 0, 17:] == 0)
    assert torch.allclose(p.sum(-1), torch.ones(2, 3, 4), atol=1e-5)


def test_token_with_value_equal_to_output_costs_nothing():
    # v_j == a means removing j does not move the output at all. DropKV must
    # give it ~0 cost even when it has the largest attention weight, which is
    # exactly where an attention-only score disagrees.
    q, k, v = random_qkv(b=1, h=1, w=1, n=30, d=16, seed=3)
    j, qv = 5, q[0, 0, 0]
    k[0, 0, j] = qv * (5.0 * 16**0.5 / (qv @ qv))      # logit of token j is exactly 5
    p = attention_probs(q, k)[0, 0, 0]
    others = torch.arange(30) != j
    v[0, 0, j] = (p[others] @ v[0, 0, others]) / (1 - p[j])   # solves v_j = a
    drop = dropkv_scores(q, k, v)[0, 0]
    snap = snapkv_scores(q, k)[0, 0]
    assert snap.argmax().item() == j
    assert drop[j] < 1e-6 * drop.max()


# ---------------------------------------------------------------- selection

def test_selection_keeps_window_and_exact_count():
    scores = torch.rand(2, 3, 50, generator=torch.Generator().manual_seed(1))
    keep = select_keep_indices(scores, n_keep=12, window=8, kernel_size=11)
    assert keep.shape == (2, 3, 12)
    assert torch.all(keep[..., 1:] > keep[..., :-1])          # sorted, no duplicates
    assert torch.all(keep[..., -8:] == torch.arange(42, 50))  # window always kept


def test_selection_edge_cases():
    scores = torch.rand(1, 2, 6)
    assert select_keep_indices(scores, n_keep=100, window=8).shape == (1, 2, 6)
    assert select_keep_indices(scores, n_keep=1, window=8).shape == (1, 2, 6)
    assert select_keep_indices(scores, n_keep=1, window=2).shape == (1, 2, 2)


def test_selection_without_pooling_is_plain_topk():
    scores = torch.tensor([[[5.0, 1.0, 4.0, 0.5, 3.0, 0.1]]])
    keep = select_keep_indices(scores, n_keep=3, window=1, pool=False)
    assert keep.tolist() == [[[0, 2, 5]]]


def test_gather_positions():
    x = torch.arange(2 * 2 * 5 * 3, dtype=torch.float32).reshape(2, 2, 5, 3)
    keep = torch.tensor([[[0, 4], [1, 2]], [[3, 4], [0, 1]]])
    out = gather_positions(x, keep)
    assert out.shape == (2, 2, 2, 3)
    assert torch.equal(out[0, 1, 0], x[0, 1, 1])
    assert torch.equal(out[1, 0, 1], x[1, 0, 4])


# ---------------------------------------------------------------- theory

def _cone_instance(seed, n=14, d=8):
    """Two heavily attended tokens with a large common value direction and a
    pool of lightly attended tokens with small values. Then a - v_i points
    roughly the same way for every pool token, so the cone condition holds."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, generator=g, dtype=torch.float64)
    logits[:2] += 6.0
    p = torch.softmax(logits, dim=0)
    v = 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64)
    v[:2, 0] += 5.0
    return p, v, list(range(2, n))


@pytest.mark.parametrize("seed", range(5))
def test_theorem1_bound_against_brute_force(seed):
    k = 3
    p, v, pool = _cone_instance(seed)
    eps = p[pool].max().item()
    c = pert.cone_constant(p, v, pool)
    assert c > 0 and k * eps < 1                      # assumptions of Theorem 1
    proxy = pert.proxy_set(p, v, k, pool)
    _, best = pert.optimal_set(p, v, k, pool)
    got = pert.eviction_loss(p, v, proxy).item()
    assert got >= best - 1e-12
    assert got <= best / (c * (1 - eps) * (1 - k * eps)) + 1e-12


# ---------------------------------------------------------------- end to end

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    return transformers.LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def prompt():
    return torch.randint(0, 128, (1, 48), generator=torch.Generator().manual_seed(0))


def test_recorded_queries_reproduce_model_attention(tiny_model, prompt):
    # attention recomputed from our recorded queries and the cached keys must
    # equal the attention weights the model itself reports
    from Attention.methods.kv_eviction.eviction import QueryRecorder, _get_kv, group_queries

    w = 8
    rec = QueryRecorder(tiny_model, w)
    with torch.no_grad():
        out = tiny_model(input_ids=prompt, use_cache=True, output_attentions=True)
    rec.remove()
    n = prompt.shape[1]
    for i, attn in enumerate(out.attentions):
        k, _ = _get_kv(out.past_key_values, i)
        q, offsets = group_queries(rec.queries[i], k.shape[1])
        p = attention_probs(q, k, q_pos=n - w + offsets)       # [1, Hkv, G*W, N]
        p = p.reshape(1, attn.shape[1], w, n)
        assert torch.allclose(p, attn[:, :, -w:, :], atol=1e-5)


def greedy_without_cache(model, prompt, steps):
    ids, out = prompt, []
    with torch.no_grad():
        for _ in range(steps):
            nxt = model(input_ids=ids).logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt.item())
            ids = torch.cat([ids, nxt], dim=1)
    return out


def test_full_budget_matches_unmodified_model(tiny_model, prompt):
    from Attention.methods.kv_eviction.eviction import generate

    ref = greedy_without_cache(tiny_model, prompt, 6)
    base, _ = generate(tiny_model, prompt, max_new_tokens=6, method="full")
    same, info = generate(tiny_model, prompt, max_new_tokens=6, method="dropkv", budget=1.0)
    assert base == ref
    assert same == ref
    assert info["kept"] == prompt.shape[1]


@pytest.mark.parametrize("method", ["dropkv", "snapkv", "random"])
def test_eviction_shrinks_cache(tiny_model, prompt, method):
    from Attention.methods.kv_eviction.eviction import _get_kv, generate, prefill

    _, cache, info = prefill(tiny_model, prompt, method=method, budget=0.25, window=8)
    assert info["kept"] == 12
    for i in range(2):
        k, v = _get_kv(cache, i)
        assert k.shape == (1, 2, 12, 8) and v.shape == k.shape
    assert info["kv_bytes"] * 4 == info["kv_bytes_full"]
    tokens, _ = generate(tiny_model, prompt, max_new_tokens=5, method=method, budget=0.25)
    assert len(tokens) == 5
