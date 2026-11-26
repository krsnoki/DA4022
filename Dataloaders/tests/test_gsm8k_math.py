"""Offline tests for the math-reasoning loaders (canned rows; nothing is downloaded)."""

import pytest

from Dataloaders.evaluation.gsm8k_math import (
    MathExample, build_messages, build_user_prompt, check_disjoint, gsm8k_final_answer, last_boxed_content,
    records_to_examples, sample_indices,
)
from Dataloaders.registry import get_dataset_entry, list_dataset_ids

IDS = ["text.gsm8k.train", "text.gsm8k.test", "text.math.train", "text.math500.test"]
REQUIRED = ["id", "name", "modality", "task", "source_url", "hf_path", "license", "version", "role", "split",
            "files", "num_examples", "seed", "preprocessing_id", "loader", "card", "gold_format"]


@pytest.mark.parametrize("dataset_id", IDS)
def test_registry_entries_complete(dataset_id):
    e = get_dataset_entry(dataset_id)
    missing = [k for k in REQUIRED if k not in e]
    assert not missing, missing
    assert len(e["version"]) == 40  # pinned commit hash


def test_roles_keep_calibration_and_test_apart():
    assert get_dataset_entry("text.gsm8k.train")["role"] == "calibration"
    assert get_dataset_entry("text.math.train")["role"] == "calibration"
    assert get_dataset_entry("text.gsm8k.test")["role"] == "test"
    assert get_dataset_entry("text.math500.test")["role"] == "test"
    assert set(IDS) <= set(list_dataset_ids())
    with pytest.raises(KeyError):
        get_dataset_entry("text.unknown")


def test_gsm8k_final_answer():
    assert gsm8k_final_answer("He has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == "7"
    assert gsm8k_final_answer("...\n#### 1,000") == "1000"
    with pytest.raises(ValueError):
        gsm8k_final_answer("no marker")


def test_last_boxed_content_nested_and_last():
    assert last_boxed_content("a \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed_content("\\fbox{7}") == "7"
    assert last_boxed_content("no box") is None
    assert last_boxed_content("\\boxed{unbalanced") is None
    # brace-less form found in MATH train rows 888 and 1011
    assert last_boxed_content("our answer is $\\boxed 2$.") == "2"
    assert last_boxed_content("$x^2 + y^2 = \\boxed 9$.") == "9"
    assert last_boxed_content("\\boxed 3 earlier, finally \\boxed{4}") == "4"


def test_records_to_examples_per_dataset():
    g = records_to_examples(get_dataset_entry("text.gsm8k.test"), [{"question": "Q?", "answer": "x\n#### 12"}])
    assert g[0] == MathExample("text.gsm8k.test:0", "text.gsm8k.test", 0, "Q?", "12", "number", {})
    m = records_to_examples(get_dataset_entry("text.math.train"),
                            [{"problem": "P", "solution": "so \\boxed{\\dfrac{3}{4}}.", "level": "Level 2",
                              "type": "Algebra"}])
    assert m[0].gold == "\\dfrac{3}{4}" and m[0].meta == {"level": "Level 2", "subject": "Algebra"}
    m5 = records_to_examples(get_dataset_entry("text.math500.test"),
                             [{"problem": "P5", "answer": "\\pi", "level": 3, "subject": "Geometry",
                               "unique_id": "test/geometry/1.json"}])
    assert m5[0].gold == "\\pi" and m5[0].gold_format == "latex"
    with pytest.raises(ValueError):
        records_to_examples(get_dataset_entry("text.math.train"), [{"problem": "P", "solution": "no box"}])


def test_sample_indices_deterministic_and_bounded():
    a = sample_indices(100, 10, seed=42)
    assert a == sample_indices(100, 10, seed=42) and len(set(a)) == 10
    assert a != sample_indices(100, 10, seed=43)
    # offset continues the same permutation: disjoint consecutive blocks
    b = sample_indices(100, 10, seed=42, offset=10)
    assert not set(a) & set(b)
    assert sample_indices(100, 5, seed=42) == a[:5]
    assert sorted(sample_indices(7, None, seed=1)) == list(range(7))
    assert sample_indices(10, 0, seed=1) == []
    for bad in [dict(num_samples=11), dict(num_samples=-1), dict(num_samples=1, offset=-1)]:
        with pytest.raises(ValueError):
            sample_indices(10, seed=1, **{"num_samples": None, **bad})


def test_prompt_and_messages():
    ex = MathExample("u", "d", 0, "  What is 2+2? ", "4", "number")
    assert build_user_prompt(ex).startswith("What is 2+2?\nPlease reason step by step")
    assert build_user_prompt(ex, "raw_question.v1") == "What is 2+2?"
    msgs = build_messages(ex)
    assert msgs == [{"role": "user", "content": build_user_prompt(ex)}]  # no system prompt (model card advice)


def test_check_disjoint():
    a = [MathExample("d:1", "d", 1, "q1", "1", "number")]
    b = [MathExample("e:1", "e", 1, "q2", "1", "number")]
    check_disjoint(a, b)
    with pytest.raises(ValueError):
        check_disjoint(a, [MathExample("e:9", "e", 9, "q1", "1", "number")])
