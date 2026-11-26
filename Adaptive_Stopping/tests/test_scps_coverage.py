"""Property tests for the semantic-coverage objects (paper Theorem 1, Theorem 2, Eq. 5, Prop. P3)."""

import itertools
import random

import pytest

from Adaptive_Stopping.methods.budget_policy.scps.coverage import (
    answer_profile,
    answer_value,
    bonferroni_lower_bound,
    expected_batch_gain,
    residual_hit_terms,
    semantic_value,
)

TARGETS = ["t0", "t1", "t2", "t3"]


def rand_profile(rng, binary=False):
    return {t: (rng.random() < 0.4) * 1.0 if binary else round(rng.random(), 3) for t in TARGETS if rng.random() < 0.7}


def rand_instance(rng, binary=False):
    mu = {t: rng.uniform(0.1, 2.0) for t in TARGETS}
    base = {t: (0.0 if binary else round(rng.random() * 0.5, 3)) for t in TARGETS}
    return mu, base


@pytest.mark.parametrize("seed", range(30))
def test_theorem1_normalized_monotone_submodular(seed):
    rng = random.Random(seed)
    mu, base = rand_instance(rng)
    ground = [rand_profile(rng) for _ in range(5)]
    F = lambda S: semantic_value([ground[i] for i in S], base, mu)
    assert F(()) == pytest.approx(0.0)
    for r in range(len(ground) + 1):
        for A in itertools.combinations(range(len(ground)), r):
            for extra in range(len(ground)):
                if extra in A:
                    continue
                B = A + (extra,)
                assert F(B) >= F(A) - 1e-12  # monotone
                for y in range(len(ground)):
                    if y in B:
                        continue
                    gain_A = F(A + (y,)) - F(A)
                    gain_B = F(B + (y,)) - F(B)
                    assert gain_A >= gain_B - 1e-12  # diminishing returns


def test_answer_value_is_exact_answer_quotient():
    pi = {"A": 0.5, "B": 0.3, "C": 0.2}
    answers = ["A", "A", "C", None]
    assert answer_value(answers, pi) == pytest.approx(0.7)
    # Eq. (2) is Eq. (1) with k(z, a) = 1{canon(z) = a}, b = 0, mu = pi
    assert semantic_value([answer_profile(a) for a in answers], {}, pi) == pytest.approx(0.7)


def _random_realizations(rng, actions, k=6, binary=False):
    ps = [rng.random() for _ in range(k)]
    z = sum(ps)
    reals = []
    for p in ps:
        reals.append((p / z, {e: [rand_profile(rng, binary) for _ in range(rng.randint(0, 2))] for e in actions}))
    return reals


@pytest.mark.parametrize("seed", range(30))
def test_overcount_identity_and_bonferroni(seed):
    rng = random.Random(1000 + seed)
    actions = ["e1", "e2", "e3"]
    mu, base = rand_instance(rng)
    portfolio = [rand_profile(rng)]
    reals = _random_realizations(rng, actions)
    G = expected_batch_gain(reals, actions, portfolio, base, mu)
    terms = residual_hit_terms(reals, actions, portfolio, base, mu)
    # Prop. P3: additive - G = E[(N - 1)_+] exactly
    assert terms["additive"] - G == pytest.approx(terms["overcount"], abs=1e-9)
    # Eq. (5): second-order Bonferroni lower bound
    assert bonferroni_lower_bound(terms) <= G + 1e-9
    # Eq. (4): single-action gain equals v(e)
    for e in actions:
        assert expected_batch_gain(reals, [e], portfolio, base, mu) == pytest.approx(terms["v"][e], abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_theorem2_lifted_gain_is_submodular(seed):
    rng = random.Random(2000 + seed)
    actions = ["e1", "e2", "e3", "e4"]
    mu, base = rand_instance(rng)
    reals = _random_realizations(rng, actions)
    G = lambda B: expected_batch_gain(reals, B, [], base, mu)
    assert G([]) == pytest.approx(0.0)
    for r in range(len(actions)):
        for A in itertools.combinations(actions, r):
            for x in actions:
                if x in A:
                    continue
                for y in actions:
                    if y in A or y == x:
                        continue
                    B = list(A) + [x]
                    assert G(B) >= G(list(A)) - 1e-12
                    assert G(list(A) + [y]) - G(list(A)) >= G(B + [y]) - G(B) - 1e-12


def test_invalid_profile_values_rejected():
    with pytest.raises(ValueError):
        semantic_value([{"t0": 1.5}], {}, {"t0": 1.0})
    with pytest.raises(ValueError):
        expected_batch_gain([(0.5, {})], [], [], {}, {"t0": 1.0})
