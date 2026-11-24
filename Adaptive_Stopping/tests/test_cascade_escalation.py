"""CPU tests for the cascade-escalation implementation (Bouchard 2026).

Run: ``pytest Adaptive_Stopping/tests -q`` (no GPU, no network).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Adaptive_Stopping" / "experiments" / "cascade_escalation"))

from Adaptive_Stopping.baselines.cascade_baselines import always_model, random_router  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation import theory  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.data import CascadeTable, stratified_split  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.learned_scores import ScoreProvider, fit_logistic  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.policy import (  # noqa: E402
    CascadePolicy,
    CascadeStoppingPolicy,
    evaluate_policy,
    stop_index,
)
from Adaptive_Stopping.methods.cascade_escalation.scores import SCORE_NAMES, token_scores_from_topk  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.search import (  # noqa: E402
    budget_grid,
    build_frontiers,
    enumerate_sequence,
    pareto_frontier,
    select_and_evaluate,
    threshold_grid,
)
from Dataloaders.evaluation import cascade_benchmarks as bench  # noqa: E402


# ----------------------------------------------------------------- fixtures

def synthetic(n=400, m=4, seed=0):
    """Shared-difficulty synthetic cascade data: bigger models are more accurate
    and more expensive; each model's score is informative about its correctness."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    skill = np.linspace(-0.5, 1.5, m)
    logits = skill[None, :] - difficulty[:, None] + 0.5 * rng.normal(size=(n, m))
    U = (rng.random((n, m)) < 1 / (1 + np.exp(-2 * logits))).astype(float)
    base = 1 / (1 + np.exp(-(logits + rng.normal(scale=1.0, size=(n, m)))))
    S = {name: np.clip(base + 0.01 * k, 0, 1) for k, name in enumerate(SCORE_NAMES)}
    sizes = np.array([1.0, 3.0, 9.0, 27.0])[:m]
    C = sizes[None, :] * (1 + 0.1 * rng.random((n, m)))
    return U, C, S


# ----------------------------------------------------------------- scores

def test_scores_known_values():
    # two tokens, K=2: probs (0.5, 0.5) and (0.9, 0.1)
    top = [[math.log(0.5), math.log(0.5)], [math.log(0.9), math.log(0.1)]]
    chosen = [math.log(0.5), math.log(0.9)]
    s = token_scores_from_topk(chosen, top)
    assert s["min_prob"] == pytest.approx(0.5)
    assert s["length_norm"] == pytest.approx(math.sqrt(0.45))
    assert s["prob_margin"] == pytest.approx((0.0 + 0.8) / 2)
    h2 = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1)) / math.log(2)
    assert s["min_negentropy"] == pytest.approx(0.0, abs=1e-12)
    assert s["mean_negentropy"] == pytest.approx((0.0 + 1 - h2) / 2)


def test_scores_range_and_empty():
    rng = np.random.default_rng(1)
    for _ in range(20):
        L = rng.integers(1, 12)
        top = np.log(rng.dirichlet(np.ones(15), size=L)).tolist()
        chosen = [max(r) for r in top]
        s = token_scores_from_topk(chosen, top)
        assert set(s) == set(SCORE_NAMES)
        assert all(0.0 <= v <= 1.0 + 1e-12 for v in s.values())
    assert all(v == 0.0 for v in token_scores_from_topk([], []).values())
    with pytest.raises(ValueError):
        token_scores_from_topk([0.0], [[0.0]])


def test_scores_k_truncation():
    top = [np.log([0.4, 0.3, 0.2, 0.1]).tolist()]
    full = token_scores_from_topk([math.log(0.4)], top)
    k2 = token_scores_from_topk([math.log(0.4)], top, k=2)
    assert k2["prob_margin"] == pytest.approx(0.4 / 0.7 - 0.3 / 0.7)
    assert full["prob_margin"] == pytest.approx(0.1)


# ----------------------------------------------------------------- policy

def test_policy_validation():
    with pytest.raises(ValueError):
        CascadePolicy((0, 1), ())
    with pytest.raises(ValueError):
        CascadePolicy((0, 0), (0.5,))
    with pytest.raises(ValueError):
        CascadePolicy((0, 1), (float("nan"),))
    assert CascadePolicy((0, 1, 2), (0.5, -np.inf)).effective_depth == 2


def test_policy_strict_escalation_and_costs():
    U = np.array([[1, 0], [0, 1], [0, 1], [1, 1]], float)
    C = np.array([[1, 10], [1, 10], [1, 10], [1, 10]], float)
    s = np.array([[0.9, 0], [0.2, 0], [0.5, 0], [0.5, 0]])
    fn = lambda m, nxt=None: s[:, m]  # noqa: E731
    pol = CascadePolicy((0, 1), (0.5,))  # strict: 0.5 does NOT escalate
    assert stop_index(pol, fn, 4).tolist() == [0, 1, 0, 0]
    r = evaluate_policy(pol, U, C, fn)
    assert r["accuracy"] == pytest.approx(3 / 4)
    assert r["mean_cost"] == pytest.approx((1 + 11 + 1 + 1) / 4)
    assert r["escalation_rate"] == pytest.approx(0.25)
    never = evaluate_policy(CascadePolicy((0, 1), (-np.inf,)), U, C, fn)
    always = evaluate_policy(CascadePolicy((0, 1), (np.inf,)), U, C, fn)
    assert never["accuracy"] == pytest.approx(U[:, 0].mean()) and never["mean_cost"] == 1
    assert always["accuracy"] == pytest.approx(U[:, 1].mean()) and always["mean_cost"] == 11


def test_stopping_policy_contract():
    sp = CascadeStoppingPolicy(CascadePolicy((0, 1, 2), (0.5, 0.7)))
    sp.reset()
    assert sp.should_stop(0, {"score": 0.6}) is True
    sp.reset()
    assert sp.should_stop(0, {"score": 0.4}) is False
    assert sp.should_stop(1, {"score": 0.69}) is False
    assert sp.should_stop(2, {"score": 0.0}) is True  # terminal always stops
    sp.reset()
    assert sp.should_stop(0, {}) is False  # missing telemetry escalates
    assert sp.should_stop(1, {"score": 0.0}, budget={"max_stages": 2}) is True  # budget cap
    rep = sp.report()
    assert rep["requests"] == 3 and rep["escalations"] == 3
    with pytest.raises(ValueError):
        sp.should_stop(5, {"score": 1.0})


# ----------------------------------------------------------------- search

def test_threshold_grid_sentinels_and_ties():
    g = threshold_grid(np.array([0.3, 0.3, 0.1, 0.9]), None)
    assert g[0] == -np.inf and g[-1] == np.inf
    assert list(g[1:-1]) == [0.3, 0.9]
    assert len(threshold_grid(np.linspace(0, 1, 1000), 10)) <= 12


@pytest.mark.parametrize("seq", [(0, 1), (0, 2, 3), (0, 1, 2, 3)])
def test_enumeration_matches_direct_evaluation(seq):
    U, C, S = synthetic(n=150)
    fn = lambda m, nxt=None: S["mean_negentropy"][:, m]  # noqa: E731
    cal = np.arange(150)
    cs = enumerate_sequence(seq, U, C, fn, cal, grid_sizes={3: 7, 4: 5})
    rng = np.random.default_rng(0)
    for k in rng.choice(len(cs.acc), size=min(25, len(cs.acc)), replace=False):
        r = evaluate_policy(cs.policy(k), U, C, fn, cal)
        assert r["accuracy"] == pytest.approx(cs.acc[k], abs=1e-12)
        assert r["mean_cost"] == pytest.approx(cs.cost[k], abs=1e-9)


def test_pair_search_is_exhaustive():
    U, C, S = synthetic(n=60)
    fn = lambda m, nxt=None: S["min_prob"][:, m]  # noqa: E731
    cal = np.arange(60)
    cs = enumerate_sequence((0, 1), U, C, fn, cal)
    # every achievable escalation set of a pair is a prefix by score -> n_distinct + 1 policies
    assert len(cs.acc) == len(np.unique(S["min_prob"][:, 0])) + 1


def test_frontier_monotone_and_nested_classes():
    U, C, S = synthetic(n=300)
    fn = lambda m, nxt=None: S["mean_negentropy"][:, m]  # noqa: E731
    cal, test = stratified_split(U[:, 0], 42)
    fr = build_frontiers(U, C, fn, cal, max_depth=4, grid_sizes={3: 9, 4: 5})
    for d, f in fr.items():
        assert np.all(np.diff(f.cost) >= 0) and np.all(np.diff(f.acc) > 0)
    budgets = np.linspace(C[cal].mean(0).min(), C[cal].mean(0).max(), 30)
    prev = None
    for d in range(1, 5):
        cal_best = np.array([fr[d].acc[fr[d].select(b)] for b in budgets])
        if prev is not None:  # nesting: calibration accuracy never decreases with depth
            assert np.all(cal_best >= prev - 1e-12)
        prev = cal_best
    rows = select_and_evaluate(fr[2], budgets, U, C, fn, test)
    assert all(r["feasible"] for r in rows)
    assert all(r["cal_cost"] <= b + 1e-9 for r, b in zip(rows, budgets))


def test_s1_frontier_is_best_single_model():
    U, C, S = synthetic(n=200)
    fn = lambda m, nxt=None: S["mean_negentropy"][:, m]  # noqa: E731
    cal = np.arange(200)
    fr = build_frontiers(U, C, fn, cal, max_depth=1)[1]
    for b in C.mean(0):
        k = fr.select(b)
        feas = C.mean(0) <= b + 1e-12
        assert fr.acc[k] == pytest.approx(U.mean(0)[feas].max())


def test_budget_grid_endpoints():
    acc = np.array([0.5, 0.9, 0.7])
    cost = np.array([1.0, 5.0, 9.0])
    b = budget_grid(acc, cost, 11)
    assert b[0] == 1.0 and b[-1] == 5.0 and len(b) == 11


def test_informative_score_beats_random_router():
    U, C, S = synthetic(n=2000, seed=3)
    fn = lambda m, nxt=None: S["mean_negentropy"][:, m]  # noqa: E731
    rnd = ScoreProvider(S, "random", seed=0)
    cal, test = stratified_split(U[:, 0], 42)
    good = pareto_frontier([enumerate_sequence((0, 3), U, C, fn, cal)])
    bad = pareto_frontier([enumerate_sequence((0, 3), U, C, rnd, cal)])
    b = C[cal, 0].mean() + 0.5 * C[cal, 3].mean()
    g = evaluate_policy(good.policies[good.select(b)], U, C, fn, test)
    r = evaluate_policy(bad.policies[bad.select(b)], U, C, rnd, test)
    assert g["accuracy"] > r["accuracy"]


# ----------------------------------------------------------------- theory

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prop32_identity_exact(seed):
    """Prop. 3.2: integral of (a - abar) == p [Gamma(U_i) - Gamma(U_>i)], both forms."""
    U, C, S = synthetic(n=500, seed=seed)
    s1 = S["mean_negentropy"][:, 0] + 1e-9 * np.arange(500)  # continuous (no ties)
    s2 = S["mean_negentropy"][:, 1] + 1e-9 * np.arange(500)
    # stage 2 of a 3-model cascade, stage-1 threshold fixed
    reached = s1 < np.quantile(s1, 0.6)
    tau2 = np.quantile(s2, 0.4)
    U_after = np.where(s2 < tau2, U[:, 2], U[:, 1])  # outcome of stage-2 onward... for stage 1:
    lhs = theory.escalation_gain_over_random(s1, U[:, 0], U_after, np.ones(500, bool))
    rhs = theory.prop32_rhs(s1, U[:, 0], U_after, np.ones(500, bool))
    assert lhs == pytest.approx(rhs["rhs_form1"], abs=1e-12)
    assert rhs["rhs_form1"] == pytest.approx(rhs["rhs_form2"], abs=1e-12)
    # intermediate stage with p_i < 1
    lhs2 = theory.escalation_gain_over_random(s2, U[:, 1], U[:, 2], reached, U_other=U[:, 0])
    rhs2 = theory.prop32_rhs(s2, U[:, 1], U[:, 2], reached)
    assert rhs2["p_reach"] == pytest.approx(reached.mean())
    assert lhs2 == pytest.approx(rhs2["rhs_form1"], abs=1e-12)


def test_auroc_basics():
    assert theory.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert theory.auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert math.isnan(theory.auroc(np.array([1.0, 2.0]), np.array([1, 1])))
    assert theory.gamma_stat(np.array([1.0, 2.0]), np.array([1, 1])) == 0.0


def test_foc_two_model_slope_matches_ratio():
    """At interior frontier points the slope dU*/dC ~ (m_H - m_L)/gamma (Eq. 15)."""
    rng = np.random.default_rng(0)
    n = 20000
    s = rng.random(n)
    U_L = (rng.random(n) < s).astype(float)          # m_L(s) = s
    U_H = (rng.random(n) < 0.9).astype(float)        # m_H = 0.9
    C = np.column_stack([np.ones(n), np.full(n, 4.0)])
    U = np.column_stack([U_L, U_H])
    S = np.column_stack([s, np.zeros(n)])
    fn = lambda m, nxt=None: S[:, m]  # noqa: E731
    cs = enumerate_sequence((0, 1), U, C, fn, np.arange(n), grid_sizes={2: 41})
    order = np.argsort(cs.cost)
    cost, acc = cs.cost[order], cs.acc[order]
    taus = np.array([cs.policy(k).thresholds[0] for k in order])
    slope = theory.frontier_slopes(cost, acc)
    est = theory.local_benefit_cost(s, U_L, U_H, C[:, 1], taus, window=2000)
    mid = slice(5, -5)
    # true ratio (0.9 - tau)/4; compare estimates loosely
    assert np.nanmean(np.abs(slope[mid] - (0.9 - taus[mid]) / 4)) < 0.02
    assert np.nanmean(np.abs(est["ratio"][mid] - (0.9 - taus[mid]) / 4)) < 0.02
    assert theory.stagewise_dominance(s, U_L, U_H, window=2000) > 0.8
    assert theory.concavity_violations(cost, acc) < 0.35


# ----------------------------------------------------------------- learned scores

def test_logistic_and_learned_providers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] + 0.1 * rng.normal(size=500) > 0).astype(float)
    p = fit_logistic(X, y)(X)
    assert ((p > 0.5) == y).mean() > 0.9
    U, C, S = synthetic(n=300)
    cal = np.arange(150)
    cp = ScoreProvider(S, "correctness_tokenprob", U=U, cal_idx=cal)
    sc = cp(0, 1)
    assert sc.shape == (300,) and np.all((sc >= 0) & (sc <= 1))
    assert theory.auroc(sc[150:], U[150:, 0]) > 0.6
    bp = ScoreProvider(S, "benefit_tokenprob", U=U, cal_idx=cal)
    assert not np.allclose(bp(0, 1), bp(0, 3))  # depends on the next model
    with pytest.raises(ValueError):
        bp(0, None)
    with pytest.raises(KeyError):
        ScoreProvider(S, "nope")


# ----------------------------------------------------------------- baselines / data

def test_baselines():
    U, C, S = synthetic(n=1000)
    a0, a3 = always_model(U, C, 0), always_model(U, C, 3)
    assert a0["accuracy"] == pytest.approx(U[:, 0].mean())
    r = random_router(U, C, 0, 3, 0.5, seed=1)
    assert r["escalation_rate"] == pytest.approx(0.5)
    assert r["mean_cost"] == pytest.approx(C[:, 0].mean() + 0.5 * C[:, 3].mean(), rel=0.05)
    assert min(a0["accuracy"], a3["accuracy"]) - 0.05 < r["accuracy"] < max(a0["accuracy"], a3["accuracy"]) + 0.05
    with pytest.raises(ValueError):
        random_router(U, C, 0, 3, 1.5)


def test_stratified_split():
    strata = np.array([0] * 7 + [1] * 5)
    cal, test = stratified_split(strata, 42)
    assert len(np.intersect1d(cal, test)) == 0 and len(cal) + len(test) == 12
    assert (strata[cal] == 0).sum() == 3 and (strata[cal] == 1).sum() == 2
    cal2, _ = stratified_split(strata, 42)
    assert np.array_equal(cal, cal2)


def test_table_from_rows():
    rows = {
        "a": [{"uid": "q1", "correct": 1, "n_in": 10, "n_out": 2, "gpu_s": 0.1, "scores": dict.fromkeys(SCORE_NAMES, 0.5)},
              {"uid": "q2", "correct": 0, "n_in": 10, "n_out": 4, "gpu_s": 0.2, "scores": dict.fromkeys(SCORE_NAMES, 0.1)}],
        "b": [{"uid": "q2", "correct": 1, "n_in": 12, "n_out": 1, "gpu_s": 0.3, "scores": dict.fromkeys(SCORE_NAMES, 0.9)},
              {"uid": "q1", "correct": 1, "n_in": 12, "n_out": 1, "gpu_s": 0.3, "scores": dict.fromkeys(SCORE_NAMES, 0.9)}],
    }
    t = CascadeTable.from_rows("mmlu", rows, {"a": 1e9, "b": 2e9})
    assert t.uids == ["q1", "q2"] and t.U.tolist() == [[1, 1], [0, 1]]
    assert t.C["flops"][1, 0] == pytest.approx(2 * 1e9 * 14 / 1e12)


def test_graders_and_loader():
    import pandas as pd

    assert bench.grade_mmlu("B", {"ground_truth": "B"}) == 1
    assert bench.grade_mmlu("The answer is C.", {"ground_truth": "B"}) == 0
    assert bench.grade_triviaqa("The Beatles!", {"ground_truth": "x", "aliases": ["beatles"]}) == 1
    assert bench.grade_triviaqa("Rolling Stones", {"ground_truth": "x", "aliases": ["beatles"]}) == 0
    assert bench.extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert bench.grade_math(r"\boxed{4}", {"ground_truth": "4"}) == 1
    assert bench.grade_math(r"\boxed{0.5}", {"ground_truth": r"\frac{1}{2}"}) == 1
    assert bench.grade_math(r"\boxed{5}", {"ground_truth": "4"}) == 0
    df = pd.DataFrame({"id": range(50), "prompt": [f"q{i}" for i in range(50)],
                       "ground_truth": ["x"] * 50, "answer_aliases": ["x | y"] * 50})
    a = bench.load_records("triviaqa", num_examples=10, seed=7, frame=df)
    b = bench.load_records("triviaqa", num_examples=10, seed=7, frame=df.sample(frac=1, random_state=0))
    assert [r["uid"] for r in a] == [r["uid"] for r in b]  # order-independent, deterministic
    assert a[0]["aliases"] == ["x", "y"] and len({r["uid"] for r in a}) == 10


# ----------------------------------------------------------------- end-to-end (tiny model, CPU)

def test_generation_end_to_end_tiny_model():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from generation import load_hf_generator

    g = load_hf_generator("tiny-random-0", top_k=15, device="cpu")
    res = g.generate(["What is 2+2?", "A much longer prompt about cascades"], max_new_tokens=6, batch_size=2)
    for r in res:
        assert r["n_out"] == len(r["topk_logprobs"]) == len(r["chosen_logprobs"]) <= 6
        assert all(len(row) == 15 for row in r["topk_logprobs"])
        # greedy: chosen token is the top-1 candidate
        assert all(c == pytest.approx(row[0]) for c, row in zip(r["chosen_logprobs"], r["topk_logprobs"]))
        s = token_scores_from_topk(r["chosen_logprobs"], r["topk_logprobs"])
        assert all(0 <= v <= 1 for v in s.values())
    one = g.generate_one_timed("What is 2+2?", 6)
    assert one["latency_s"] > 0
    # batch and single generation agree (left padding is handled)
    assert one["response"] == res[0]["response"]


_REAL_LOAD_RECORDS = bench.load_records


def _fake_records(name, num_examples=None, seed=2026, cache_dir=None, frame=None):
    import pandas as pd

    n = 30
    if name == "mmlu":
        df = pd.DataFrame({"id": range(n), "prompt": [f"Question {i}? A. x B. y" for i in range(n)],
                           "ground_truth": ["AB"[i % 2] for i in range(n)]})
    elif name == "triviaqa":
        df = pd.DataFrame({"id": range(n), "prompt": [f"Who {i}?" for i in range(n)],
                           "ground_truth": ["x"] * n, "answer_aliases": ["x | y"] * n})
    else:
        df = pd.DataFrame({"id": range(n), "prompt": [f"Solve {i}+1" for i in range(n)],
                           "ground_truth": [str(i + 1) for i in range(n)]})
    return _REAL_LOAD_RECORDS(name, num_examples=num_examples, seed=seed, frame=df)


def test_pipeline_cli_tiny(tmp_path, monkeypatch):
    """collect_generations -> live_cascade on CPU with tiny random models."""
    pytest.importorskip("torch")
    import collect_generations
    import live_cascade

    monkeypatch.setattr(bench, "load_records", _fake_records)
    cfg = str(ROOT / "Adaptive_Stopping" / "tests" / "fixtures" / "tiny_config.yaml")
    for m in ("tiny-random-0", "tiny-random-1", "tiny-random-2"):
        assert collect_generations.main(["--config", cfg, "--model", m, "--out", str(tmp_path), "--backend", "auto"]) == 0
    from common import generation_path, read_jsonl

    rows = read_jsonl(generation_path(tmp_path, "mmlu", "tiny1"))
    assert len(rows) == 24 and set(rows[0]["scores"]) == set(SCORE_NAMES)
    assert (tmp_path / "generations" / "meta" / "tiny1.json").exists()
    # re-run is a no-op (resume)
    assert collect_generations.main(["--config", cfg, "--model", "tiny-random-0", "--out", str(tmp_path)]) == 0
    assert live_cascade.main(["--config", cfg, "--results", str(tmp_path), "--device", "cpu"]) == 0
    assert (tmp_path / "live" / "live_summary.csv").exists()
