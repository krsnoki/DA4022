import pytest

pytest.importorskip("math_verify")

from Performance_Metrics.quality.math_answer import answer_section, is_correct  # noqa: E402


def test_answer_section_uses_text_after_last_think_end():
    assert answer_section("reasoning \\boxed{1}\n</think>\n\nFinal \\boxed{2}") == "\n\nFinal \\boxed{2}"
    assert answer_section("no think end \\boxed{3}") == "no think end \\boxed{3}"


@pytest.mark.parametrize("pred,gold,fmt,ok", [
    ("...</think>\n\nThus the total is \\(\\boxed{900}\\).", "900", "number", True),
    ("</think>\n\n**Final Answer**\n\\boxed{78.50}", "78.5", "number", True),
    ("</think>\n\n\\boxed{1000}", "1,000", "number", True),
    ("</think>\n\n\\boxed{80}", "40", "number", False),
    ("I think \\boxed{40} but</think>\n\nthe answer is \\boxed{80}", "40", "number", False),  # thinking ignored
    ("</think>\n\n\\boxed{\\dfrac{14}{3}}", "\\frac{14}{3}", "latex", True),
    ("</think>\n\n\\boxed{(3, \\frac{\\pi}{2})}", "\\left( 3, \\frac{\\pi}{2} \\right)", "latex", True),
    ("</think>\n\nno final answer", "10", "latex", False),
])
def test_is_correct(pred, gold, fmt, ok):
    assert is_correct(pred, gold, fmt) is ok


def test_unknown_gold_format_rejected():
    with pytest.raises(ValueError):
        is_correct("\\boxed{1}", "1", "words")
