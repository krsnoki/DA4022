"""
Tests for Attention/methods/context_compression/kvgrad (our KVgrad score variants).

1. Math checks on random tensors (float64):
   - Eq. (6): summed over probes and query heads, the Taylor terms equal v . grad_v Phi computed by autograd.
     This checks the per-head slicing of W_O, the easiest thing to get wrong.
   - Eq. (7): |T[p, c]| <= A[p, c] ||v_c W_O|| ||grad_{a_p} Phi|| for every probe and entry (Cauchy-Schwarz).
   - Reductions: bound_sum >= bound_max >= 0, taylor_sum and taylor_max are the sum / max of the same terms.
2. Reference comparison on a tiny random Qwen3 (CPU, no downloads):
   - variant "bound_max" gives the same scores and the same evicted KV pairs as kvpress's KVgradPress,
     with explicit tolerances, including a context split into several replay chunks.
   - the score cache: a second call on the same context reuses the scores (no replay) and evicts exactly
     what a fresh press of that variant evicts.

Run from the repository root:
    python -m pytest Attention/tests/test_kvgrad_variants.py -q
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kvpress")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the Attention/ folder

from _kvgrad_tiny_model import WORDS, make_tiny_model
from methods.context_compression.kvgrad import (
    VARIANTS,
    KVgradVariantsPress,
    score_variants,
    taylor_terms,
    value_output_norm,
)

# Explicit tolerances for comparing float32 scores with the kvpress reference
RTOL, ATOL = 1e-5, 1e-7


# ------------------------------------------------------------------------------------------------
# 1. Math checks on random tensors
# ------------------------------------------------------------------------------------------------
def _random_layer(seed: int = 0, bsz=1, kv_heads=2, groups=3, head_dim=4, hidden=8, q_len=5, chunk=7):
    g = torch.Generator().manual_seed(seed)
    attn = torch.softmax(torch.randn(bsz, kv_heads, groups, q_len, chunk, generator=g, dtype=torch.float64), -1)
    values = torch.randn(bsz, kv_heads, chunk, head_dim, generator=g, dtype=torch.float64)
    o_proj = torch.randn(hidden, kv_heads * groups * head_dim, generator=g, dtype=torch.float64)
    grad = torch.randn(bsz, q_len, hidden, generator=g, dtype=torch.float64)
    return attn, values, o_proj, grad


def _attention_output(attn, values, o_proj):
    """a_p = sum_h sum_c A_h[p, c] v_c W_O^(h), computed head by head without the code under test."""
    bsz, kv_heads, groups, q_len, _ = attn.shape
    head_dim = values.shape[-1]
    out = torch.zeros(bsz, q_len, o_proj.shape[0], dtype=attn.dtype)
    for h in range(kv_heads):
        for g in range(groups):
            query_head = h * groups + g
            w_head = o_proj[:, query_head * head_dim : (query_head + 1) * head_dim]  # (hidden, head_dim)
            out = out + attn[:, h, g] @ (values[:, h] @ w_head.T)
    return out


def test_taylor_terms_match_autograd():
    attn, values, o_proj, phi = _random_layer(seed=1)
    values = values.clone().requires_grad_(True)
    # Linear objective with gradient phi w.r.t. the attention output, so grad_a Phi = phi exactly
    objective = (_attention_output(attn, values, o_proj) * phi).sum()
    (grad_values,) = torch.autograd.grad(objective, values)
    expected = (values * grad_values).sum(-1)  # v_c . grad_{v_c} Phi, shape (bsz, kv_heads, chunk)

    terms = taylor_terms(attn, values.detach(), phi, o_proj)  # (bsz, kv_heads, groups, q_len, chunk)
    torch.testing.assert_close(terms.sum(dim=(2, 3)), expected.detach(), rtol=1e-10, atol=1e-12)


def test_magnitude_bound_holds_for_every_probe():
    attn, values, o_proj, phi = _random_layer(seed=2)
    terms = taylor_terms(attn, values, phi, o_proj)
    value_norm = value_output_norm(o_proj, values, attn.shape[2])
    bound = attn * value_norm[:, :, :, None, :] * phi.norm(dim=-1)[:, None, None, :, None]
    assert (terms.abs() <= bound + 1e-12).all()


def test_variant_reductions():
    attn, values, o_proj, phi = _random_layer(seed=3)
    scores = score_variants(attn, values, phi, o_proj)
    assert set(scores) == set(VARIANTS)
    for s in scores.values():
        assert s.shape == (1, 2, 7)
    assert (scores["bound_max"] >= 0).all()
    assert (scores["bound_sum"] >= scores["bound_max"] - 1e-12).all()
    terms = taylor_terms(attn, values, phi, o_proj)
    torch.testing.assert_close(scores["taylor_max"], terms.amax(dim=(-3, -2)))
    torch.testing.assert_close(scores["taylor_sum"], terms.sum(dim=-2).amax(dim=-2))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        KVgradVariantsPress(compression_ratio=0.5, variant="not_a_variant")


# ------------------------------------------------------------------------------------------------
# 2. Comparison with kvpress's KVgradPress on a tiny model
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_pipe(tmp_path_factory):
    import kvpress  # noqa: F401  (registers the pipeline)
    from transformers import pipeline

    path = tmp_path_factory.mktemp("kvgrad_variants") / "model"
    make_tiny_model(path)
    return pipeline(
        "kv-press-text-generation",
        model=str(path),
        device="cpu",
        model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
    )


CONTEXT = " ".join(WORDS * 6) + " the secret number is 42 " + " ".join(WORDS * 6)
QUESTION = " what is the secret number ?"


def _run(pipe, press):
    """Run the pipeline and return (scores before eviction, evicted (layer, head, position) set)."""
    captured = {}
    original = press.compress_post

    def capture(model):
        captured["scores"] = press.score_val.detach().clone()
        original(model)
        captured["evicted"] = {
            (layer_idx, int(h), int(t))
            for layer_idx, layer in enumerate(model.model.layers)
            for _, h, t in zip(*layer.self_attn.masked_key_indices)
        }

    press.compress_post = capture
    pipe(CONTEXT, question=QUESTION, press=press, max_new_tokens=2)
    return captured["scores"], captured["evicted"]


@pytest.mark.parametrize("chunk_size", [512, 48])  # one replay chunk, and several chunks
def test_bound_max_matches_kvpress_kvgrad(tiny_pipe, chunk_size):
    from kvpress import KVgradPress

    n_tokens = len(tiny_pipe.tokenizer(CONTEXT).input_ids)
    assert chunk_size == 512 or n_tokens > 2 * chunk_size  # the second case really uses several chunks

    ref_scores, ref_evicted = _run(tiny_pipe, KVgradPress(compression_ratio=0.5, chunk_size=chunk_size))
    our_scores, our_evicted = _run(
        tiny_pipe, KVgradVariantsPress(compression_ratio=0.5, chunk_size=chunk_size, variant="bound_max")
    )
    assert ref_scores.abs().sum() > 0
    torch.testing.assert_close(our_scores, ref_scores, rtol=RTOL, atol=ATOL)
    assert our_evicted == ref_evicted
    assert len(our_evicted) > 0


def test_score_cache_reuses_scores_without_replay(tiny_pipe):
    cache: dict = {}
    first = KVgradVariantsPress(compression_ratio=0.5, variant="bound_max", score_cache=cache)
    _run(tiny_pipe, first)
    assert first.last_scoring_cached is False
    assert len(cache) == 1 and set(next(iter(cache.values()))) == set(VARIANTS)

    for variant in VARIANTS:
        for ratio in (0.5, 0.875):
            fresh_scores, fresh_evicted = _run(tiny_pipe, KVgradVariantsPress(compression_ratio=ratio, variant=variant))
            cached_press = KVgradVariantsPress(compression_ratio=ratio, variant=variant, score_cache=cache)
            cached_scores, cached_evicted = _run(tiny_pipe, cached_press)
            assert cached_press.last_scoring_cached is True
            torch.testing.assert_close(cached_scores, fresh_scores, rtol=RTOL, atol=ATOL)
            assert cached_evicted == fresh_evicted, (variant, ratio)


def test_variants_rank_entries_differently(tiny_pipe):
    """The four variants are genuinely different scores, not copies of one another."""
    cache: dict = {}
    _run(tiny_pipe, KVgradVariantsPress(compression_ratio=0.5, score_cache=cache))
    scores = next(iter(cache.values()))
    flat = {name: s.flatten() for name, s in scores.items()}
    for a in VARIANTS:
        for b in VARIANTS:
            if a < b:
                assert not torch.allclose(flat[a], flat[b]), (a, b)
