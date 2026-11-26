"""Tests for the tempered remasking implementation (TLC / TCT).

CPU-only smoke, shape/dtype, statistical-correctness and edge-case tests.
Run from the repository root:

    python -m pytest Adaptive_Stopping/tests -q
"""

import math

import pytest
import torch

from tempered_remasking import (
    Autoregressive,
    ConfidenceThreshold,
    DecodingStopPolicy,
    LowConfidence,
    RandomRemasking,
    SamplerConfig,
    TemperedConfidenceThreshold,
    TemperedLowConfidence,
    build_policy,
    generate,
    generate_samples,
    get_num_transfer_tokens,
    sample_tokens,
)
from tempered_remasking.remasking import topk_per_row

MASK = 0


class RandomLogitModel(torch.nn.Module):
    """Deterministic toy dLLM: logits depend on position and on the tokens present."""

    def __init__(self, vocab: int = 11, seq_len: int = 40, seed: int = 0, eos: int = None, eos_from: int = None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = torch.nn.Parameter(torch.randn(seq_len, vocab, generator=g) * 2.0)
        self.emb = torch.nn.Embedding(vocab, vocab)
        torch.nn.init.normal_(self.emb.weight, std=0.3, generator=g)
        self.eos, self.eos_from = eos, eos_from

    def forward(self, x):
        B, N = x.shape
        ctx = self.emb(x).mean(dim=1, keepdim=True)  # bidirectional context
        logits = self.base[:N].unsqueeze(0) + ctx
        logits = logits.clone()
        logits[..., MASK] = -1e4  # never predict [MASK]
        if self.eos is not None:
            logits[..., self.eos] = -1e4
            if self.eos_from is not None:
                logits[:, self.eos_from :, self.eos] = 1e4  # force EOS from this position on
        return logits


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- #
# Token sampling & confidences
# --------------------------------------------------------------------------- #
def test_sample_tokens_greedy_and_confidence_modes():
    logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 0.0]]])
    p = sample_tokens(logits, 0.0, confidence_mode="sampled", compute_entropy=True)
    assert p.tokens.tolist() == [[1, 0]]
    probs = torch.softmax(logits.double(), -1)
    assert torch.allclose(p.confidence, probs.max(-1).values)
    assert p.confidence.dtype == torch.float64 and p.tokens.dtype == torch.long
    ent = -(probs * probs.log()).sum(-1)
    assert torch.allclose(p.entropy, ent)
    pm = sample_tokens(logits, 0.0, confidence_mode="max")
    assert torch.allclose(pm.confidence, p.confidence)
    pt = sample_tokens(logits, 0.0, confidence_mode="tempered_sampled")
    assert torch.all(pt.confidence == 1)


def test_sample_tokens_temperature_distribution_and_untempered_confidence():
    logits = torch.tensor([0.0, 1.0, 2.0]).expand(1, 20000, 3)
    T = 0.8
    p = sample_tokens(logits, T, _gen(1), confidence_mode="sampled")
    freq = torch.bincount(p.tokens.flatten(), minlength=3).double() / 20000
    target = torch.softmax(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64) / T, -1)
    assert torch.allclose(freq, target, atol=0.015)
    # footnote 1: confidence is the *untempered* probability of the sampled token
    untempered = torch.softmax(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), -1)
    assert torch.allclose(p.confidence[0], untempered[p.tokens[0]])
    pt = sample_tokens(logits, T, _gen(1), confidence_mode="tempered_sampled")
    assert torch.allclose(pt.confidence[0], target[pt.tokens[0]])


def test_get_num_transfer_tokens_matches_llada_schedule():
    mask = torch.tensor([[True] * 10 + [False] * 2, [True] * 3 + [False] * 9])
    s = get_num_transfer_tokens(mask, 4)
    assert s.tolist() == [[3, 3, 2, 2], [1, 1, 1, 0]]
    assert s.sum(1).tolist() == [10, 3]


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def test_topk_per_row_variable_k_and_candidates():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.8, 0.1]])
    cand = torch.tensor([[True, True, True, False], [True, True, False, True]])
    sel = topk_per_row(scores, cand, torch.tensor([2, 5]))
    assert sel.tolist() == [[False, True, True, False], [True, True, False, True]]


def test_lc_selects_top_k():
    c = torch.tensor([[0.2, 0.9, 0.5, 0.95, 0.1]], dtype=torch.float64)
    cand = torch.tensor([[True, True, True, False, True]])
    sel = LowConfidence().select(c, cand, torch.tensor([2]))
    assert sel.tolist() == [[False, True, True, False, False]]


def test_tlc_limits_recover_lc_and_random():
    torch.manual_seed(0)
    c = torch.rand(64, 30, dtype=torch.float64)
    cand = torch.rand(64, 30) < 0.7
    k = torch.randint(1, 5, (64,))
    lc = LowConfidence().select(c, cand, k)
    assert torch.equal(TemperedLowConfidence(0.0).select(c, cand, k, _gen()), lc)
    # tiny T_pos: Gumbel noise is negligible relative to log(c)/T_pos
    assert torch.equal(TemperedLowConfidence(1e-9).select(c, cand, k, _gen()), lc)
    assert isinstance(build_policy("random"), RandomRemasking)
    assert math.isinf(RandomRemasking().tpos)


@pytest.mark.parametrize("tpos", [0.3, 1.0, 2.0, float("inf")])
def test_tlc_k1_matches_tempered_categorical(tpos):
    """Alg. 2 with K=1: P(select k) = c_k^{1/T} / Σ_j c_j^{1/T} over masked positions."""
    c = torch.tensor([0.9, 0.6, 0.3, 0.8, 0.05], dtype=torch.float64)
    cand = torch.tensor([True, True, True, False, True])
    n = 40000
    sel = TemperedLowConfidence(tpos).select(c.expand(n, -1), cand.expand(n, -1), torch.ones(n, dtype=torch.long), _gen(3))
    assert torch.all(sel.sum(1) == 1) and not sel[:, 3].any()
    freq = sel.double().mean(0)
    w = torch.ones_like(c) if math.isinf(tpos) else c ** (1 / tpos)
    w = w * cand
    assert torch.allclose(freq, w / w.sum(), atol=0.01)


def test_tlc_k2_is_sampling_without_replacement():
    """K=2: probability of the pair {i, j} equals the sequential w/o-replacement probability."""
    c = torch.tensor([0.9, 0.5, 0.2], dtype=torch.float64)
    T = 0.7
    w = c ** (1 / T)
    p = w / w.sum()
    n = 60000
    sel = TemperedLowConfidence(T).select(c.expand(n, -1), torch.ones(n, 3, dtype=torch.bool), torch.full((n,), 2), _gen(4))
    assert torch.all(sel.sum(1) == 2)
    # pair excluding index 2 = {0,1}
    exact = p[0] * p[1] / (1 - p[0]) + p[1] * p[0] / (1 - p[1])
    assert abs(float((~sel[:, 2]).double().mean()) - float(exact)) < 0.01


def test_ct_strict_threshold_and_tct_limits():
    c = torch.tensor([[0.6, 0.61, 0.2, 0.99]], dtype=torch.float64)
    cand = torch.tensor([[True, True, True, True]])
    ct = ConfidenceThreshold(0.6).select(c, cand)
    assert ct.tolist() == [[False, True, False, True]]  # strict ">" (Eq. 4)
    assert torch.equal(TemperedConfidenceThreshold(0.6, 0.0).select(c, cand, None, _gen()), ct)
    tct = TemperedConfidenceThreshold(0.6, float("inf"))
    assert torch.allclose(tct.inclusion_probability(c), torch.full_like(c, 0.5))


def test_tct_inclusion_frequency_is_sigmoid():
    c = torch.tensor([0.3, 0.55, 0.6, 0.7, 0.95], dtype=torch.float64)
    pol = TemperedConfidenceThreshold(0.6, 0.1)
    n = 40000
    sel = pol.select(c.expand(n, -1), torch.ones(n, 5, dtype=torch.bool), None, _gen(5))
    expected = torch.sigmoid((c - 0.6) / 0.1)
    assert torch.allclose(sel.double().mean(0), expected, atol=0.01)


def test_autoregressive_picks_leftmost_candidate():
    cand = torch.tensor([[False, True, True], [False, False, False], [True, False, True]])
    sel = Autoregressive().select(torch.rand(3, 3), cand)
    assert sel.tolist() == [[False, True, False], [False, False, False], [True, False, False]]


def test_build_policy_paper_defaults():
    assert build_policy("tlc").tpos == 1.0
    tct = build_policy("tct")
    assert (tct.tpos, tct.threshold) == (0.1, 0.6)
    assert build_policy("fast_dllm").threshold == 0.6
    with pytest.raises(ValueError):
        build_policy("nope")
    with pytest.raises(ValueError):
        TemperedLowConfidence(-1.0)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["lc", "tlc", "random", "ct", "tct", "ar"])
def test_generate_unmasks_everything_and_keeps_prompt(name):
    model = RandomLogitModel(seq_len=5 + 16)
    prompt = torch.tensor([[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]])
    cfg = SamplerConfig(gen_length=16, block_length=8, token_temperature=0.8)
    res = generate(model, prompt, build_policy(name), cfg, MASK, generator=_gen())
    assert res.sequences.shape == (2, 21) and res.sequences.dtype == torch.long
    assert torch.equal(res.sequences[:, :5], prompt)
    assert not (res.generated == MASK).any()
    assert res.stop_reasons == ["complete", "complete"]
    assert torch.all(res.unmask_step[:, 5:] >= 0) and torch.all(res.unmask_step[:, :5] == -1)
    assert torch.equal(res.tokens_per_step.sum(0), torch.tensor([16, 16]))


def test_semi_ar_blocks_are_filled_in_order():
    model = RandomLogitModel(seq_len=24)
    cfg = SamplerConfig(gen_length=24, block_length=8, token_temperature=0.8)
    for name in ["tlc", "tct", "lc"]:
        res = generate(model, torch.zeros(1, 0, dtype=torch.long), build_policy(name), cfg, MASK, generator=_gen())
        steps = res.unmask_step[0]
        for b in range(2):
            assert steps[b * 8 : (b + 1) * 8].max() < steps[(b + 1) * 8 : (b + 2) * 8].min()


def test_fixed_k_nfe_counts():
    model = RandomLogitModel(seq_len=32)
    p = torch.zeros(3, 0, dtype=torch.long)
    r1 = generate(model, p, LowConfidence(), SamplerConfig(gen_length=32, block_length=8), MASK)
    assert r1.nfe.tolist() == [32, 32, 32]  # K = 1
    r2 = generate(model, p, TemperedLowConfidence(1.0), SamplerConfig(gen_length=32, block_length=8, steps=8), MASK, generator=_gen())
    assert r2.nfe.tolist() == [8, 8, 8] and torch.all(r2.tokens_per_step == 4)  # K = 4
    with pytest.raises(ValueError):
        generate(model, p, LowConfidence(), SamplerConfig(gen_length=32, block_length=8, steps=6), MASK)


def test_ct_lambda_one_unmasks_one_token_per_step():
    """λ = 1.0 never fires (c <= 1), so the Fast-dLLM argmax fallback decodes 1 token/step."""
    model = RandomLogitModel(seq_len=16)
    res = generate(model, torch.zeros(2, 0, dtype=torch.long), ConfidenceThreshold(1.0), SamplerConfig(gen_length=16, block_length=8, token_temperature=0.0), MASK)
    assert res.nfe.tolist() == [16, 16]
    assert res.stats["argmax_fallbacks"] == 32


def test_ct_low_threshold_is_faster_than_lc():
    model = RandomLogitModel(seq_len=32)
    p = torch.zeros(4, 0, dtype=torch.long)
    res = generate(model, p, ConfidenceThreshold(0.3), SamplerConfig(gen_length=32, block_length=8, token_temperature=0.0), MASK)
    assert torch.all(res.nfe < 32) and torch.all(res.nfe >= 4)  # at least one NFE per block


def test_tct_empty_selection_modes():
    model = RandomLogitModel(seq_len=16)
    p = torch.zeros(64, 0, dtype=torch.long)
    pol = TemperedConfidenceThreshold(0.9, 0.05)  # often selects nothing
    base = dict(gen_length=16, block_length=16, token_temperature=0.8)
    # literal Alg. 4 ("skip") can spin for many NFEs; bound it with a budget
    skip = generate(model, p, pol, SamplerConfig(**base, empty_selection="skip", max_nfe=300), MASK, generator=_gen(1))
    resample = generate(model, p, pol, SamplerConfig(**base, empty_selection="resample", max_resample=2), MASK, generator=_gen(1))
    assert not (resample.generated == MASK).any()
    assert skip.stats["skipped_steps"] > 0
    assert torch.all(resample.nfe <= 16)  # never wastes an NFE
    assert skip.nfe.float().mean() > resample.nfe.float().mean()
    with pytest.raises(ValueError):
        generate(model, p, ConfidenceThreshold(0.5), SamplerConfig(**base, empty_selection="skip"), MASK)


def test_autoregressive_order_and_eos_stopping():
    eos = 10
    model = RandomLogitModel(seq_len=20, eos=eos, eos_from=7)
    res = generate(model, torch.zeros(2, 0, dtype=torch.long), Autoregressive(), SamplerConfig(gen_length=20, block_length=4, token_temperature=0.8), MASK, eos_token_ids=[eos], generator=_gen())
    # left-to-right: unmask step equals position until EOS
    assert res.unmask_step[0, :8].tolist() == list(range(8))
    assert res.stop_reasons == ["eos", "eos"]
    assert res.nfe.tolist() == [8, 8]  # positions 0..7, EOS at 7
    assert res.nfe_paper_convention.tolist() == [7, 7]  # tokens before <eos> (footnote 3)
    assert torch.all(res.generated[:, 7:] == eos)  # padded with EOS
    # without EOS ids the AR baseline runs all L steps
    res2 = generate(model, torch.zeros(1, 0, dtype=torch.long), Autoregressive(), SamplerConfig(gen_length=20, block_length=4), MASK)
    assert res2.nfe.tolist() == [20] and res2.stop_reasons == ["complete"]
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 0, dtype=torch.long), LowConfidence(), SamplerConfig(gen_length=20, block_length=4, stop_at_eos=True), MASK, [eos])


def test_budget_edge_cases():
    model = RandomLogitModel(seq_len=16)
    p = torch.zeros(2, 0, dtype=torch.long)
    base = dict(gen_length=16, block_length=8)
    # zero budget: no forward pass, output stays masked
    r0 = generate(model, p, TemperedLowConfidence(1.0), SamplerConfig(**base, max_nfe=0), MASK, generator=_gen())
    assert r0.nfe.tolist() == [0, 0] and r0.stop_reasons == ["budget", "budget"]
    assert torch.all(r0.generated == MASK)
    # minimum budget with one-shot finish
    r1 = generate(model, p, TemperedLowConfidence(1.0), SamplerConfig(**base, max_nfe=1), MASK, generator=_gen())
    assert r1.nfe.tolist() == [1, 1] and not (r1.generated == MASK).any()
    r1b = generate(model, p, TemperedLowConfidence(1.0), SamplerConfig(**base, max_nfe=1, on_budget_exhausted="leave_masked"), MASK, generator=_gen())
    assert (r1b.generated == MASK).sum(1).tolist() == [15, 15]
    # budget larger than needed: never binds
    rmax = generate(model, p, TemperedLowConfidence(1.0), SamplerConfig(**base, max_nfe=10_000), MASK, generator=_gen())
    assert rmax.nfe.tolist() == [16, 16] and rmax.stop_reasons == ["complete", "complete"]
    with pytest.raises(ValueError):
        SamplerConfig(**base, max_nfe=-1).validate(LowConfidence())


def test_stop_policy_contract_and_missing_telemetry():
    sp = DecodingStopPolicy(stop_at_eos=True)
    sp.reset({"id": 1})
    stop = sp.should_stop({"masked_remaining": [0, 3, 3, 3], "eos_generated": [False, True, False, False]}, {"nfe": [5, 5, 5, 9]}, 9)
    assert stop.tolist() == [True, True, False, True]
    with pytest.raises(KeyError):
        sp.should_stop({"masked_remaining": [1]}, {"nfe": [1]}, None)  # eos telemetry missing
    with pytest.raises(KeyError):
        DecodingStopPolicy().should_stop({"masked_remaining": [1]}, {}, 5)  # nfe missing with a budget
    assert DecodingStopPolicy().should_stop({"masked_remaining": [1]}, {}, None).tolist() == [False]
    sp.record("eos", 7)
    rep = sp.report()
    assert rep["num_samples"] == 1 and rep["stop_rate_eos"] == 1.0


def test_determinism_and_batch_independence():
    model = RandomLogitModel(seq_len=16)
    p = torch.zeros(3, 0, dtype=torch.long)
    cfg = SamplerConfig(gen_length=16, block_length=8, token_temperature=0.8)
    a = generate(model, p, TemperedConfidenceThreshold(0.6, 0.1), cfg, MASK, generator=_gen(7))
    b = generate(model, p, TemperedConfidenceThreshold(0.6, 0.1), cfg, MASK, generator=_gen(7))
    assert torch.equal(a.sequences, b.sequences) and torch.equal(a.nfe, b.nfe)
    g = generate(model, p, LowConfidence(), SamplerConfig(gen_length=16, block_length=8, token_temperature=0.0), MASK)
    assert g.stats["stop_policy"]["deterministic"] is True
    assert torch.equal(g.sequences[0], g.sequences[1])  # greedy LC is deterministic
    assert a.stats["stop_policy"]["deterministic"] is False


def test_generate_samples_records():
    model = RandomLogitModel(seq_len=3 + 8)
    out = generate_samples(model, torch.tensor([1, 2, 3]), 5, build_policy("tct"), SamplerConfig(gen_length=8, block_length=4), MASK, batch_size=2, generator=_gen())
    assert len(out) == 5 and all(len(o["tokens"]) == 8 and o["nfe"] >= 2 for o in out)


def test_tempering_increases_sample_diversity():
    """TLC/TCT produce more distinct generations than their greedy-order counterparts."""
    model = RandomLogitModel(seq_len=12, seed=3)
    p = torch.zeros(300, 0, dtype=torch.long)
    cfg = SamplerConfig(gen_length=12, block_length=12, token_temperature=0.5)
    def distinct(pol):
        r = generate(model, p, pol, cfg, MASK, generator=_gen(11))
        return len({tuple(s) for s in r.generated.tolist()})
    assert distinct(TemperedLowConfidence(1.0)) > distinct(LowConfidence())


def test_sample_tokens_chunking_is_exact():
    logits = torch.randn(7, 5, 13, generator=_gen(2))
    full = sample_tokens(logits, 0.0, confidence_mode="sampled", compute_entropy=True)
    chunked = sample_tokens(logits, 0.0, confidence_mode="sampled", compute_entropy=True, max_chunk_elements=5 * 13 * 2)
    assert torch.equal(full.tokens, chunked.tokens)
    assert torch.allclose(full.confidence, chunked.confidence) and torch.allclose(full.entropy, chunked.entropy)
    st = sample_tokens(logits, 0.8, _gen(3), max_chunk_elements=13)  # one position-row per chunk
    assert st.tokens.shape == (7, 5)
