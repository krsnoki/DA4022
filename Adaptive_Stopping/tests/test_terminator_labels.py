from Adaptive_Stopping.methods.learned_exit.terminator.labels import (
    answer_token_position,
    char_to_token,
    find_answer_char_span,
    normalized_view,
    token_char_starts,
    token_labels,
)


def span_text(text, span):
    return text[span[0] : span[1]]


def test_answer_at_start():
    text = "123 is what I get first. Let me check."
    span = find_answer_char_span(text, "123")
    assert span == (0, 3)


def test_short_number_needs_context():
    text = "10! = 10 x 9 x 8 x 7 x 6 x 5, and so the count is 6."
    span = find_answer_char_span(text, "6")
    assert span[0] == text.rindex("6")


def test_thousands_separator_and_slash_form():
    text = "so 108 * 256 = 27,648 in total"
    assert span_text(text, find_answer_char_span(text, "27648")) == "27,648"
    text = "the slope is -2/3, then we check \\frac{-2}{3}"
    assert find_answer_char_span(text, "-\\frac{2}{3}")[0] == text.index("-2/3")


def test_answer_absent():
    assert find_answer_char_span("we try 41 and 43", "42") is None
    assert find_answer_char_span("", "42") is None
    assert find_answer_char_span("text", None) is None
    assert token_labels(5, None) is None


def test_multiple_occurrences_first_wins():
    text = "so x = 7. Check: 7 again. Final: 7"
    span = find_answer_char_span(text, "7")
    assert span == (7, 8)


def test_number_inside_larger_number_is_skipped():
    text = "we get 142 and 4.2 and x^42, then the answer is 42."
    span = find_answer_char_span(text, "42")
    assert span_text(text, span) == "42"
    assert span[0] == text.rindex("42")


def test_answer_in_problem_needs_context():
    problem = "A box has 3 red balls and 5 blue balls."
    text = "There are 3 red and 5 blue. The count is 3 after removing two."
    span = find_answer_char_span(text, "3", problem)
    # the first "3" restates the problem; the one after "is" counts
    assert span[0] == text.index("is 3") + 3


def test_boxed_and_latex_forms():
    text = "So the value is $\\dfrac{1}{2}$, i.e. \\boxed{\\frac{1}{2}}."
    span = find_answer_char_span(text, "\\frac12".replace("12", "{1}{2}"))
    assert span is not None and span[0] == text.index("\\dfrac")


def test_whitespace_insensitive_match():
    text = "we find ( 3 , 4 ) works"
    span = find_answer_char_span(text, "(3,4)")
    assert span_text(text, span) == "( 3 , 4 )"


def test_normalized_view_map():
    norm, idx = normalized_view("a $\\left( b \\right)$")
    assert norm == "a(b)"
    assert [ "a $\\left( b \\right)$"[i] for i in idx] == list("a(b)")


def test_char_to_token():
    starts = [0, 3, 10, 13]
    assert char_to_token(starts, 0) == 0
    assert char_to_token(starts, 2) == 0
    assert char_to_token(starts, 3) == 1
    assert char_to_token(starts, 12) == 2
    assert char_to_token(starts, 50) == 3


class ByteTok:
    """Byte-level BPE style tokens, as Qwen's tokenizer returns them."""

    def __init__(self, toks):
        self.toks = toks

    def convert_ids_to_tokens(self, ids):
        return [self.toks[i] for i in ids]


def test_token_char_starts_byte_level():
    toks = ["The", "Ġanswer", "Ġis", "Ġ4", "2", "."]
    text, starts = token_char_starts(ByteTok(toks), list(range(len(toks))))
    assert text == "The answer is 42."
    assert starts == [0, 3, 10, 13, 15, 16]
    pos = answer_token_position(text, starts, "42")
    # label 1 starts at the token holding the last character of the answer
    assert pos == 4
    assert token_labels(6, pos) == [0, 0, 0, 0, 1, 1]


def test_token_char_starts_multibyte():
    # "é" is two bytes (Ã ©) split across two tokens
    toks = ["c", "Ã", "©", "Ġ=", "Ġ5"]
    text, starts = token_char_starts(ByteTok(toks), [0, 1, 2, 3, 4])
    assert text == "cé = 5"
    assert starts[3:] == [2, 4]
    assert answer_token_position(text, starts, "5") == 4


def test_token_char_starts_decode_fallback(char_tokenizer):
    ids = char_tokenizer("ab 7")["input_ids"]

    class NoBytes:
        decode = staticmethod(char_tokenizer.decode)

    text, starts = token_char_starts(NoBytes(), ids)
    assert text == "ab 7" and starts == [0, 1, 2, 3]


def test_token_labels_bounds():
    assert token_labels(4, 0) == [1, 1, 1, 1]
    assert token_labels(4, 4) == [0, 0, 0, 0]
    assert token_labels(3, 10) == [0, 0, 0]


def test_token_metrics_skips_unlabelled_and_clamps_position():
    from Adaptive_Stopping.experiments.terminator.summarize_results import token_metrics

    labels = {"a": 2, "b": None, "c": 10}
    probs = {"a": [0.1, 0.9, 0.8, 0.2], "b": [0.9, 0.9], "c": [0.4, 0.6]}
    m = token_metrics(labels, probs)
    # a: y=0,0,1,1 pred=0,1,1,0 -> tp1 fp1 fn1 tn1; c: pos clamped to 2, y=0,0 pred=0,1 -> fp1 tn1
    assert m["n_traces"] == 2 and m["n_tokens"] == 6
    assert m["f1_label1"] == round(2 / 5, 4)
    assert m["f1_label0"] == round(4 / 7, 4)
    assert m["accuracy"] == round(3 / 6, 4)
