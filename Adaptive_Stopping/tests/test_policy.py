import pytest

from Adaptive_Stopping.baselines.no_stop import NoStopping
from Adaptive_Stopping.baselines.step_count import StepCountStopping
from Adaptive_Stopping.common.policy import StepState
from Adaptive_Stopping.methods.step_tagging.policy import StepTagEarlyStopping


class FixedTagger:
    """Tags a step by looking up its text in a dict (no rules involved)."""

    name = "fixed"

    def __init__(self, mapping):
        self.mapping = mapping

    def tag(self, text):
        return self.mapping.get(text, "other")


TAGS = ["problem_restatement", "formula_substitution", "verification", "self_talk", "verification",
        "formula_substitution", "verification", "final_conclusion"]
STEPS = [StepState(i, f"s{i}", 10 * (i + 1)) for i in range(len(TAGS))]
TAGGER = FixedTagger({f"s{i}": t for i, t in enumerate(TAGS)})


def run(policy, steps=STEPS, budget=None):
    policy.reset()
    for s in steps:
        if policy.should_stop(s, None, budget):
            return s.step_index
    return None


@pytest.mark.parametrize("delta,expected", [(0, 2), (1, 4), (2, 6), (3, None), (20, None)])
def test_st_es_stops_when_count_exceeds_delta(delta, expected):
    # Eq. 3: continue while count <= delta; the step that makes count = delta + 1 is the stop step
    assert run(StepTagEarlyStopping("verification", delta, TAGGER)) == expected


def test_st_es_report_and_reset():
    p = StepTagEarlyStopping("verification", 1, TAGGER)
    assert run(p) == 4
    rep = p.report()
    assert rep["stopped"] and rep["stopped_at_step"] == 4 and rep["count"] == 2 and rep["stop_reason"] == "constraint"
    assert rep["tag_counts"]["verification"] == 2
    p.reset()
    assert p.report()["count"] == 0 and not p.report()["stopped"]


def test_st_es_ignores_steps_after_thinking():
    steps = [StepState(i, s.text, s.n_tokens, in_thinking=i < 2) for i, s in enumerate(STEPS)]
    assert run(StepTagEarlyStopping("verification", 0, TAGGER), steps) is None


def test_st_es_missing_telemetry_never_counts():
    steps = [StepState(i, None, 10 * (i + 1)) for i in range(5)]
    p = StepTagEarlyStopping("verification", 0, TAGGER)
    assert run(p, steps) is None
    assert p.report()["missing_steps"] == 5


def test_st_es_precomputed_tag_in_metrics_is_used():
    p = StepTagEarlyStopping("verification", 0, TAGGER)
    p.reset()
    assert p.should_stop(StepState(0, None, 5), {"tag": "verification"}) is True


def test_st_es_monitor_only_when_delta_is_none():
    p = StepTagEarlyStopping("verification", None, TAGGER)
    assert run(p) is None and p.report()["count"] == 3


@pytest.mark.parametrize("budget,expected", [
    ({"max_steps": 0}, 0),        # zero budget: stop at the first completed step
    ({"max_steps": 2}, 1),
    ({"max_tokens": 30}, 2),      # n_tokens of step 2 is 30
    ({"max_steps": 100}, 4),      # large budget: the constraint decides (delta=1 -> step 4)
    ({}, 4),
    (None, 4),
])
def test_budget_edge_cases(budget, expected):
    assert run(StepTagEarlyStopping("verification", 1, TAGGER), budget=budget) == expected


def test_negative_budget_rejected():
    p = StepTagEarlyStopping("verification", 1, TAGGER)
    with pytest.raises(ValueError):
        run(p, budget={"max_steps": -1})


@pytest.mark.parametrize("bad", [-1, 1.5, "2"])
def test_invalid_delta_rejected(bad):
    with pytest.raises(ValueError):
        StepTagEarlyStopping("verification", bad, TAGGER)


def test_unknown_tag_rejected():
    with pytest.raises(ValueError):
        StepTagEarlyStopping("not_a_tag", 0, TAGGER)


@pytest.mark.parametrize("k,expected", [(1, 0), (3, 2), (8, 7), (9, None)])
def test_step_count_baseline(k, expected):
    assert run(StepCountStopping(k)) == expected


@pytest.mark.parametrize("bad", [0, -2, 2.0])
def test_step_count_rejects_bad_k(bad):
    with pytest.raises(ValueError):
        StepCountStopping(bad)


def test_no_stopping_never_stops():
    p = NoStopping()
    assert run(p) is None and p.report()["reasoning_steps_seen"] == len(STEPS)


def test_policies_declare_contract_flags():
    for p in (StepTagEarlyStopping("verification", 0, TAGGER), StepCountStopping(3)):
        assert p.requires_extra_forward_pass is False and p.deterministic is True
