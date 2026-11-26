"""Appendix D (formal model) checked analytically and by Monte Carlo with the real sampler."""

import math
import random

import pytest
import torch

from tempered_remasking import SamplerConfig, build_policy, generate
from tempered_remasking import theory as th
from tempered_remasking.anchor_fork import FORK_A, AnchorForkModel


def make_model(fork=1, anchors=None, length=8):
    anchors = anchors or {3: (0.97, 0.15), 5: (0.96, 0.15), 6: (0.95, 0.10)}
    return AnchorForkModel(length=length, fork_position=fork, anchors=anchors, filler_confidence=0.8)


def mc(model, policy, n=20000, seed=0, token_temperature=1.0):
    cfg = SamplerConfig(gen_length=model.length, block_length=model.length, token_temperature=token_temperature,
                        confidence_mode="max", record_entropy=True)
    res = generate(model, model.prompt(n), policy, cfg, model.mask_token_id, generator=torch.Generator().manual_seed(seed))
    f = model.prompt_length + model.fork_position
    H = res.entropy_at_unmask[:, f].mean().item()
    pA = (res.sequences[:, f] == FORK_A).double().mean().item()
    first = {a: (res.unmask_step[:, model.prompt_length + a] < res.unmask_step[:, f]).double().mean().item() for a in model.anchors}
    return H, pA, first


def test_definition_d1_and_sandwich_bound():
    # semantic outcome determined by fork value up to small noise
    joint = {("numpy", "pop_std"): 0.45, ("numpy", "nan"): 0.05, ("statistics", "sample_std"): 0.5}
    rep = th.fork_token_report(joint, eps=0.0, delta=0.2)
    assert rep["condition_1"] and rep["condition_2"] and rep["is_fork"] and rep["sandwich_holds"]
    assert not th.fork_token_report(joint, eps=0.0, delta=0.1)["is_fork"]
    assert rep["sandwich_lower"] <= rep["H_S"] <= rep["sandwich_upper"]


def test_model_satisfies_assumption_d2_and_rejects_violations():
    m = make_model()
    import itertools
    anchors = list(m.anchors)
    fork_conf = {frozenset(S): m.fork_confidence(frozenset(S)) for r in range(4) for S in itertools.combinations(anchors, r)}
    chk = th.check_anchor_fork_degeneracy(fork_conf, {a: c for a, (c, _) in m.anchors.items()}, m.h0, m.etas)
    assert all(chk.values())
    # linear entropy degradation (condition 2)
    for S, c in fork_conf.items():
        assert math.isclose(th.binary_entropy(c), m.h0 - sum(m.etas[a] for a in S), abs_tol=1e-9)
    # gap is only required for anchors that are still masked: one anchor never violates it
    AnchorForkModel(length=6, fork_position=0, anchors={2: (0.7, 0.5)})
    with pytest.raises(ValueError):  # revealing anchor 3 makes the fork more confident than anchor 2
        AnchorForkModel(length=6, fork_position=0, anchors={2: (0.7, 0.3), 3: (0.99, 0.3)})


def test_lemma_d3_exact_formula_and_bounds():
    for ca, cl, T in [(0.9, 0.6, 1.0), (0.99, 0.5, 0.3), (0.8, 0.79, 2.0)]:
        w = ca ** (1 / T) / (ca ** (1 / T) + cl ** (1 / T))
        assert math.isclose(th.pairwise_order_probability(ca, cl, T), w, rel_tol=1e-12)
    assert th.pairwise_order_probability(0.9, 0.6, 0) == 1.0
    assert th.pairwise_order_probability(0.9, 0.6, float("inf")) == 0.5
    m = make_model()
    consts = m.lemma_constants()
    for T in [0.2, 0.5, 1.0, 3.0]:
        ex = m.exact_tlc(T)
        for a, (d, D) in consts.items():
            lo, hi = th.pairwise_order_bounds(d, D, T)
            assert lo - 1e-12 <= ex["p_anchor_first"][a] <= hi + 1e-12


def test_exact_dp_matches_sampler_monte_carlo():
    m = make_model()
    for T in [0.25, 1.0, float("inf")]:
        ex = m.exact_tlc(T, token_temperature=1.0)
        H, pA, first = mc(m, build_policy("tlc", tpos=T), seed=1)
        assert abs(H - ex["expected_fork_entropy"]) < 0.01
        assert abs(pA - ex["p_fork_A"]) < 0.015
        for a in m.anchors:
            assert abs(first[a] - ex["p_anchor_first"][a]) < 0.015
        # tower-rule identity used in the proof of Prop. D.4
        assert math.isclose(th.expected_fork_entropy(m.h0, m.etas, ex["p_anchor_first"]), ex["expected_fork_entropy"], abs_tol=1e-12)


def test_proposition_d4_and_remark_d5():
    m = make_model()
    consts = m.lemma_constants()
    ratio = th.proposition_d4_threshold([d for d, _ in consts.values()], [D for _, D in consts.values()])
    T = 0.2
    T_prime = T * ratio * 1.01
    assert th.proposition_d4_holds(T, T_prime, [d for d, _ in consts.values()], [D for _, D in consts.values()])
    assert m.exact_tlc(T_prime)["expected_fork_entropy"] > m.exact_tlc(T)["expected_fork_entropy"]
    # monotone in practice over a grid (stronger than the sufficient condition)
    grid = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, float("inf")]
    hs = [m.exact_tlc(t)["expected_fork_entropy"] for t in grid]
    assert all(a < b for a, b in zip(hs, hs[1:]))
    # Remark D.5: LC attains the minimum; any T_pos > 0 is strictly better
    lc = m.lc_fork_entropy()
    H_lc, _, _ = mc(m, build_policy("lc"), n=2000)
    assert math.isclose(H_lc, lc, abs_tol=1e-9)
    assert all(h > lc for h in hs)


def test_corollary_d7_semantic_entropy_increases():
    m = make_model()  # fork is a (0, 0)-fork: semantic outcome == fork value
    lo, hi = m.exact_tlc(0.1), m.exact_tlc(2.0)
    assert th.corollary_d7_applies(hi["expected_fork_entropy"], lo["expected_fork_entropy"], 0.0, 0.0)
    assert hi["semantic_entropy"] > lo["semantic_entropy"]


def test_proposition_d8_token_temperature_does_not_change_lc_order():
    rng = random.Random(0)
    for _ in range(200):
        zl = [rng.uniform(-2, 2) for _ in range(6)]
        zl_sorted = sorted(zl, reverse=True)
        # anchor logits: stretch the gaps of the fork (strengthened gap condition)
        za = [zl_sorted[0]] + [zl_sorted[0] - (zl_sorted[0] - v) * rng.uniform(1.05, 3.0) for v in zl_sorted[1:]]
        assert th.strengthened_gap_holds(za, zl)
        for T in [0.1, 0.3, 0.8, 1.0, 1.5, 5.0]:
            assert th.confidence_at_temperature(za, T) > th.confidence_at_temperature(zl, T)
    # With the real sampler: the model satisfies the strengthened gap, so LC with
    # post-tempering confidences reveals every anchor first for *every* T_token
    # (fork entropy stays at the LC minimum), whereas TLC does not.
    m = make_model()
    for Tt in [0.3, 1.0, 1.5, 3.0]:
        cfg = SamplerConfig(gen_length=8, block_length=8, token_temperature=Tt, confidence_mode="tempered_max", record_entropy=True)
        res = generate(m, m.prompt(500), build_policy("lc"), cfg, m.mask_token_id, generator=torch.Generator().manual_seed(0))
        assert math.isclose(res.entropy_at_unmask[:, 1].mean().item(), m.lc_fork_entropy(), abs_tol=1e-9)
        res_tlc = generate(m, m.prompt(500), build_policy("tlc", tpos=1.0), cfg, m.mask_token_id, generator=torch.Generator().manual_seed(0))
        assert res_tlc.entropy_at_unmask[:, 1].mean().item() > m.lc_fork_entropy() + 0.05


def test_proposition_d9_ar_beats_lc_only_if_an_anchor_follows_the_fork():
    m = make_model(fork=1)  # all anchors after the fork
    H_ar, _, _ = mc(m, build_policy("ar"), n=2000)
    assert math.isclose(H_ar, m.ar_fork_entropy(), abs_tol=1e-9) and H_ar > m.lc_fork_entropy()
    m2 = make_model(fork=7, anchors={1: (0.97, 0.2), 3: (0.96, 0.2)})  # all anchors before the fork
    H_ar2, _, _ = mc(m2, build_policy("ar"), n=500)
    assert math.isclose(H_ar2, m2.lc_fork_entropy(), abs_tol=1e-9)


def test_tct_increases_fork_entropy_over_ct():
    m = make_model()
    H_ct, pA_ct, _ = mc(m, build_policy("ct", threshold=0.6), n=5000)
    H_tct, pA_tct, _ = mc(m, build_policy("tct", threshold=0.6, tpos=0.1), n=5000)
    assert H_tct > H_ct + 0.05
    assert th.binary_entropy(pA_tct) > th.binary_entropy(pA_ct)
