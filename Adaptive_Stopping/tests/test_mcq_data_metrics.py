"""Canonicalizer, MMLU-Pro preprocessing, portfolio metrics and result-schema tests."""

import json
import re
from pathlib import Path

import pytest

from Adaptive_Stopping.common.mcq import build_mcq_prompt, canonicalize_answer, option_labels
from Adaptive_Stopping.common.portfolio import Completion, PortfolioResult
from Dataloaders.evaluation.mmlu_pro import sample_questions, strip_na
from Dataloaders.synthetic.mock_mcq import make_synthetic_mcq
from Performance_Metrics.schema import COLUMNS, make_row, percentile, write_csv

REPO = Path(__file__).resolve().parents[2]
L10 = option_labels(10)


@pytest.mark.parametrize("text,expected", [
    ("B", "B"), ("(C)", "C"), ("D.", "D"), ("The answer is (E).", "E"), ("answer: F", "F"),
    ("Answer is G", "G"), ("I think option H is right", "H"), ("I am not sure", None),
    ("the answer is a bit unclear", None), ("The answer is (A). Wait, the answer is (C).", "C"),
    ("The answer is (Z).", None), ("", None), ("I", "I"),
])
def test_canonicalizer(text, expected):
    assert canonicalize_answer(text, L10) == expected


def test_prompt_contains_options_and_target():
    p = build_mcq_prompt("What?", ["x", "y", "z"], target="B")
    assert "A. x" in p and "C. z" in p and "option B" in p
    with pytest.raises(ValueError):
        build_mcq_prompt("What?", ["x"], target="B")


def test_strip_na_keeps_letters():
    rec = {"question_id": 1, "question": "q", "options": ["a", "b", "c", "N/A", "N/A"], "answer": "B", "answer_index": 1}
    out = strip_na(rec)
    assert out["options"] == ["a", "b", "c"] and out["answer"] == "B"
    with pytest.raises(ValueError):
        strip_na({**rec, "answer_index": 4, "answer": "E"})
    with pytest.raises(ValueError):
        strip_na({**rec, "answer": "C"})


def test_sample_questions_deterministic_and_excludes():
    recs = [{"question_id": i} for i in range(100)]
    a = sample_questions(recs, 10, seed=47, exclude_ids=[0, 1, 2])
    b = sample_questions(list(reversed(recs)), 10, seed=47, exclude_ids=[0, 1, 2])
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
    assert not {0, 1, 2} & {r["question_id"] for r in a}
    with pytest.raises(ValueError):
        sample_questions(recs, 101, seed=1)


def test_synthetic_items_deterministic():
    assert make_synthetic_mcq(5, seed=1) == make_synthetic_mcq(5, seed=1)
    it = make_synthetic_mcq(1, seed=1)[0]
    assert it["answer"] == "ABCDEFGHIJ"[it["answer_index"]]


def test_portfolio_metrics():
    c = [Completion("A", 10, "root"), Completion("A", 30, "root"), Completion(None, 5, "root"), Completion("B", 5, "t")]
    r = PortfolioResult("x", c, {}, 1, "done")
    assert r.unique_answers() == 2 and r.pass_at_n("B") == 1 and r.pass_at_n("C") == 0
    assert r.duplicate_token_fraction() == pytest.approx(30 / 50)
    assert r.majority_answer() == "A"
    assert PortfolioResult("x", [], {}, 0, "done").duplicate_token_fraction() is None


def test_schema_matches_metrics_readme():
    text = (REPO / "Performance_Metrics" / "README_metrics.md").read_text()
    header = re.search(r"```text\n(experiment_id[^\n]*)\n```", text).group(1)
    assert header.split(",") == COLUMNS


def test_make_row_rejects_unknown_and_bad_status(tmp_path):
    with pytest.raises(KeyError):
        make_row(status="approximate", foo=1)
    with pytest.raises(ValueError):
        make_row(status="measured")
    row = make_row(status="approximate", method="scps")
    assert row["energy_j"] is None
    write_csv([row], str(tmp_path / "r.csv"))
    lines = (tmp_path / "r.csv").read_text().splitlines()
    assert lines[0].split(",") == COLUMNS and ",," in lines[1]


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == 5
