import pytest

from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Adaptive_Stopping.methods.step_tagging.taxonomy import (
    ID_TO_LABEL, LABEL_TO_ID, PAPER_BERT_TAGS, REASONTYPE, TAG_IDS, to_tag_id,
)

CASES = [
    ("Wait, let me double-check that calculation.", "verification"),
    ("Let me verify: 48 * 132 = 6336, and 6432 - 6336 = 96.", "verification"),
    ("Yes, that seems consistent. So the total is 900.", "verification"),
    ("Just to make sure I didn't miss anything, let me go through each part again.", "verification"),
    ("Alternatively, maybe I can use the Euclidean algorithm instead.", "exploration"),
    ("Hmm, I wonder if I misread the problem.", "self_talk"),
    ("Wait, hold on.", "self_talk"),
    ("Okay, so I need to find the greatest common factor of 6432 and 132.", "problem_restatement"),
    ("The problem says that each box holds 12 eggs.", "problem_restatement"),
    ("I remember that the sum of the angles of a triangle is 180 degrees.", "definition_recall"),
    ("So 48 times 100 is 4800, and 48 times 32 is 1536.", "formula_substitution"),
    ("Let me simplify by factoring out x from both sides.", "symbolic_transformation"),
    ("Case 1: x is negative.", "edge_case"),
    ("I notice that the pattern repeats every 4 terms.", "pattern_recognition"),
    ("Intuitively, the easiest way is to count backwards.", "heuristics"),
    ("That means she spent about 3.5 hours in total.", "interpretation"),
    ("As I found earlier, the width is 7.", "context_repetition"),
    ("**Final Answer**\n\\boxed{23}", "final_conclusion"),
    ("", "other"),
    ("   ", "other"),
    ("Moving on.", "other"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_keyword_tagger_canned_steps(text, expected):
    assert KeywordStepTagger().tag(text) == expected


def test_tagger_is_deterministic_and_returns_known_ids():
    tg = KeywordStepTagger()
    texts = [t for t, _ in CASES] * 3
    tags = tg.tag_all(texts)
    assert tags == tg.tag_all(texts)
    assert set(tags) <= set(TAG_IDS)


def test_taxonomy_matches_paper():
    assert len(REASONTYPE) == 14
    assert [t[1] for t in REASONTYPE] == list(range(1, 15))
    assert set(PAPER_BERT_TAGS) <= set(TAG_IDS)
    for tid, label in ID_TO_LABEL.items():
        assert to_tag_id(label) == tid and to_tag_id(tid) == tid
    assert LABEL_TO_ID["Other - please, specify"] == "other"
    with pytest.raises(KeyError):
        to_tag_id("Not a label")
