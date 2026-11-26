import random

import pytest

from Adaptive_Stopping.common.steps import (
    IncrementalDetokenizer, StepSegmenter, segment_text_pieces, segment_token_ids, split_steps,
)

TRACE = ("Okay, so I need to find x.\n\nFirst, 2 + 3 = 5.\n\n\n\nWait, let me check.\n\n\nYes, 5 is right."
         "\n</think>\n\nThe answer is \\boxed{5}.\n\nDone")


def random_chunks(text, rng):
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 4)
        out.append(text[i:i + n])
        i += n
    return out


@pytest.mark.parametrize("seed", range(20))
def test_online_segmentation_equals_str_split_for_any_chunking(seed):
    pieces = random_chunks(TRACE, random.Random(seed))
    steps = segment_text_pieces(pieces)
    assert [s.text for s in steps] == split_steps(TRACE)
    assert all(s.closed for s in steps[:-1]) and not steps[-1].closed


def test_thinking_flags_and_end_tokens():
    pieces = list(TRACE)  # one character per "token"
    steps = segment_text_pieces(pieces)
    texts = split_steps(TRACE)
    think_idx = next(i for i, t in enumerate(texts) if "</think>" in t)
    assert [s.in_thinking for s in steps] == [i < think_idx for i in range(len(texts))]
    # end_token = number of characters consumed up to and including the second "\n" of the delimiter
    pos = 0
    for s, t in zip(steps[:-1], texts[:-1]):
        pos += len(t) + 2
        assert s.end_token == pos
    assert steps[-1].end_token == len(TRACE)


def test_delimiter_split_across_tokens():
    steps = segment_text_pieces(["a", "\n", "\n", "b"])
    assert [(s.text, s.end_token, s.closed) for s in steps] == [("a", 3, True), ("b", 4, False)]


def test_empty_generation_gives_one_empty_unclosed_step():
    steps = segment_text_pieces([])
    assert len(steps) == 1 and steps[0].text == "" and not steps[0].closed


def test_segmenter_rejects_empty_delimiter():
    with pytest.raises(ValueError):
        StepSegmenter(delimiter="")


class _ByteTokenizer:
    """Stand-in byte-level tokenizer: every token is one UTF-8 byte (exercises incomplete characters)."""

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8", errors="replace")


def test_incremental_detokenizer_handles_multibyte_characters():
    tok = _ByteTokenizer()
    text = "x ≈ π/2 — ½\n\nnext ✓"
    det = IncrementalDetokenizer(tok)
    deltas = [det.push(t) for t in tok.encode(text)]
    assert "".join(deltas) == text
    assert all("\ufffd" not in d for d in deltas)


def test_segment_token_ids_matches_split_with_byte_tokenizer():
    tok = _ByteTokenizer()
    steps = segment_token_ids(tok.encode(TRACE), tok)
    assert [s.text for s in steps] == split_steps(TRACE)


def test_real_tokenizer_roundtrip_if_cached():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            local_files_only=True)
    except Exception:
        pytest.skip("backbone tokenizer not cached")
    ids = tok(TRACE + " ≈ π ½ 测试", add_special_tokens=False)["input_ids"]
    steps = segment_token_ids(ids, tok)
    assert [s.text for s in steps] == split_steps(tok.decode(ids))
