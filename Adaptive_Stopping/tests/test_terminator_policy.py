import math

import pytest

from Adaptive_Stopping.methods.learned_exit.terminator.policy import (
    FixedBudgetPolicy,
    TerminatorPolicy,
    exit_index,
    window_fires,
)


def run_policy(policy, probs, budget=None):
    policy.reset()
    for i, p in enumerate(probs, start=1):
        if policy.should_stop({"thinking_tokens": i}, {"terminator_prob": p}, budget):
            return i
    return None


def test_fewer_than_window_never_fires():
    assert exit_index([0.99] * 9, threshold=0.7, window=10) is None
    assert run_policy(TerminatorPolicy(0.7, 10), [0.99] * 9) is None


def test_fires_once_window_is_full():
    assert exit_index([0.99] * 10, threshold=0.7, window=10) == 10


def test_exact_majority_boundary():
    # 5 of 10 above is not a strict majority, 6 of 10 is
    five = [0.9] * 5 + [0.1] * 5
    assert not window_fires(five, 0.7, 10)
    assert exit_index(five, 0.7, 10) is None
    six = [0.1] * 4 + [0.9] * 6
    assert window_fires(six, 0.7, 10)
    assert exit_index(six, 0.7, 10) == 10


def test_threshold_is_strict():
    assert exit_index([0.7] * 20, threshold=0.7, window=10) is None
    assert exit_index([0.7000001] * 20, threshold=0.7, window=10) == 10


def test_window_slides():
    probs = [0.1] * 15 + [0.9] * 6
    # the 6th high value completes a majority in the last 10
    assert exit_index(probs, 0.7, 10) == 21
    assert run_policy(TerminatorPolicy(0.7, 10), probs) == 21


def test_offline_and_online_agree():
    import random

    rng = random.Random(0)
    for _ in range(50):
        probs = [rng.random() for _ in range(200)]
        for thr in (0.5, 0.7, 0.9):
            assert exit_index(probs, thr, 10) == run_policy(TerminatorPolicy(thr, 10), probs)


def test_window_one():
    assert exit_index([0.1, 0.8], threshold=0.7, window=1) == 2


def test_invalid_window():
    with pytest.raises(ValueError):
        exit_index([0.9], window=0)
    with pytest.raises(ValueError):
        TerminatorPolicy(window=0)


def test_report_fields():
    pol = TerminatorPolicy(0.7, 10)
    run_policy(pol, [0.9] * 12)
    rep = pol.report()
    assert rep["stopped"] and rep["reason"] == "probe" and rep["exit_pos"] == 10
    assert rep["n_predictions"] == 10 and rep["n_missing"] == 0


def test_reset_clears_state():
    pol = TerminatorPolicy(0.7, 10)
    run_policy(pol, [0.9] * 10)
    pol.reset()
    assert pol.report()["stopped"] is False
    assert pol.should_stop({"thinking_tokens": 1}, {"terminator_prob": 0.9}, None) is False


# budget edge cases


def test_zero_budget_stops_at_first_call():
    pol = TerminatorPolicy()
    assert pol.should_stop({"thinking_tokens": 1}, {"terminator_prob": 0.0}, 0)
    assert pol.report()["reason"] == "budget"


def test_minimum_budget():
    assert run_policy(TerminatorPolicy(), [0.0] * 5, budget=1) == 1


def test_maximum_budget_never_binds():
    assert run_policy(TerminatorPolicy(), [0.0] * 100, budget=10**9) is None


def test_probe_fires_before_budget():
    assert run_policy(TerminatorPolicy(0.7, 10), [0.9] * 50, budget=40) == 10


def test_missing_telemetry_falls_back_to_budget():
    pol = TerminatorPolicy(0.7, 10)
    stops = [pol.should_stop({"thinking_tokens": i}, {}, 5) for i in range(1, 6)]
    assert stops == [False] * 4 + [True]
    rep = pol.report()
    assert rep["reason"] == "budget" and rep["n_missing"] == 5 and rep["n_predictions"] == 0


def test_missing_telemetry_without_budget_never_stops():
    pol = TerminatorPolicy()
    for i in range(30):
        assert not pol.should_stop(None, None, None)
        assert not pol.should_stop({}, {"terminator_prob": math.nan}, None)
    assert pol.report()["n_missing"] == 60


def test_state_without_counter_uses_call_count():
    pol = TerminatorPolicy()
    stops = [pol.should_stop(None, {"terminator_prob": 0.0}, 3) for _ in range(3)]
    assert stops == [False, False, True]


def test_fixed_budget_policy():
    pol = FixedBudgetPolicy(budget_tokens=3)
    stops = [pol.should_stop({"thinking_tokens": i}, None, None) for i in range(1, 5)]
    assert stops == [False, False, True, True]
    assert pol.report()["exit_pos"] == 3
    pol.reset()
    assert not pol.should_stop({"thinking_tokens": 1}, None, None)
    assert FixedBudgetPolicy(None).should_stop({"thinking_tokens": 10**6}, None, None) is False
