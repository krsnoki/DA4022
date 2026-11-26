import pytest

from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Adaptive_Stopping.methods.step_tagging.tagger_v2 import KeywordStepTaggerV2
from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS

# Patterns observed in our *calibration* traces (GSM8K/MATH train), not in the held-out validation steps.
CASES_V2 = [
    ("Let me write that down:", "self_talk"),
    ("Which is:", "self_talk"),
    ("Let me break it down step by step.", "self_talk"),
    ("Blue: 10", "context_repetition"),
    ("Day 1: 78\nDay 2: 70\nDay 3: 56", "context_repetition"),
    ("Therefore, he kept 188 cookies in the fridge.", "interpretation"),
    ("So, Manny will have $45 left after making the purchase.", "interpretation"),
    ("So, in conclusion, her new salary after a 2% increase is 10,200.", "final_conclusion"),
    ("Wait, that's a negative value. Let me check if I made a mistake in the cross-multiplication step.", "verification"),
    ("I don't see any complications here, it's straightforward.", "verification"),
    ("Same result.", "verification"),
    ("Wait, maybe another way: each browser has three windows.", "exploration"),
    ("Okay, so I have this geometry problem here, and I need to find the ratio of the angles.", "problem_restatement"),
    ("\"A company has 200 employees. 60% of the employees drive to work. Of the rest, half take the bus.\"",
     "problem_restatement"),
    ("Let the number of blue marbles be 8x and yellow marbles be 5x.", "problem_restatement"),
    ("Where C(n, k) is the combination of n things taken k at a time.", "definition_recall"),
    ("First, the binomial coefficient. That's 20.", "formula_substitution"),
]


@pytest.mark.parametrize("text,expected", CASES_V2)
def test_v2_calibration_patterns(text, expected):
    assert KeywordStepTaggerV2().tag(text) == expected


def test_v2_keeps_v1_behaviour_on_clear_cases():
    # the v1 canned cases whose intent is unambiguous are unchanged in v2
    v1, v2 = KeywordStepTagger(), KeywordStepTaggerV2()
    for text in ["Wait, let me double-check that calculation.", "Alternatively, maybe I can use the Euclidean algorithm.",
                 "**Final Answer**\n\\boxed{23}", "So 48 times 100 is 4800, and 48 times 32 is 1536.", "", "Moving on."]:
        assert v2.tag(text) == v1.tag(text)


def test_v2_identity_and_label_space():
    v2 = KeywordStepTaggerV2()
    assert v2.name == "keyword_reasontype.v2" and KeywordStepTagger.name == "keyword_reasontype.v1"
    assert set(v2.tag_all(t for t, _ in CASES_V2)) <= set(TAG_IDS)
