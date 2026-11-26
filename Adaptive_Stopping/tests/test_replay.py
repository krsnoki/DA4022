import random

import pytest

from Adaptive_Stopping.baselines.step_count import StepCountStopping
from Adaptive_Stopping.common.replay import (
    eligible_steps, replay_policy, st_es_cut_points, step_count_cut_points, token_budget_cut_point,
)
from Adaptive_Stopping.common.steps import Step
from Adaptive_Stopping.methods.step_tagging.policy import StepTagEarlyStopping
from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS


def random_trace(rng, n_steps=40):
    steps, tags, pos = [], [], 0
    think_end = rng.randint(n_steps // 2, n_steps - 2)
    for i in range(n_steps):
        pos += rng.randint(3, 40)
        closed = i < n_steps - 1
        steps.append(Step(i, f"step {i}", pos, in_thinking=i < think_end, closed=closed))
        tags.append(rng.choice(TAG_IDS))
    return steps, tags, pos


@pytest.mark.parametrize("seed", range(10))
def test_fast_st_es_cut_points_equal_policy_replay(seed):
    rng = random.Random(seed)
    steps, tags, n_tok = random_trace(rng)
    for tag in TAG_IDS:
        fast = st_es_cut_points(steps, tags, tag, range(0, 25))
        for d, cut in fast.items():
            out = replay_policy(StepTagEarlyStopping(tag, d), steps, n_tok, tags=tags)
            assert out.stopped == (cut is not None)
            assert out.cut_tokens == (cut if cut is not None else n_tok)


@pytest.mark.parametrize("seed", range(5))
def test_fast_step_count_cut_points_equal_policy_replay(seed):
    rng = random.Random(seed)
    steps, _, n_tok = random_trace(rng)
    fast = step_count_cut_points(steps, range(1, 60))
    for k, cut in fast.items():
        out = replay_policy(StepCountStopping(k), steps, n_tok)
        assert out.cut_tokens == (cut if cut is not None else n_tok)


def test_trailing_unclosed_and_post_thinking_steps_never_stop():
    steps = [Step(0, "a", 5, True), Step(1, "b", 9, False), Step(2, "c", 12, False, closed=False)]
    tags = ["verification", "verification", "verification"]
    assert [s.index for s in eligible_steps(steps)] == [0]
    assert st_es_cut_points(steps, tags, "verification", [0, 1]) == {0: 5, 1: None}
    unclosed = [Step(0, "a", 5, True, closed=False)]
    assert replay_policy(StepTagEarlyStopping("verification", 0), unclosed, 5, tags=["verification"]).stopped is False


def test_token_budget_cut_point():
    steps = [Step(i, "x", 10 * (i + 1), True) for i in range(5)]
    assert token_budget_cut_point(steps, 35) == 30
    assert token_budget_cut_point(steps, 5) is None
    assert token_budget_cut_point(steps, 1000) == 50
