"""Step segmentation of reasoning traces, offline and online.

A *step* is the text between two occurrences of the delimiter ``"\\n\\n"``
(the paper's Eq. 1 and the authors' released code, ``answer.split('\\n\\n')``).
The online segmenter yields exactly the pieces of ``str.split(delimiter)``
regardless of how the text arrives in chunks, so offline replay and live
generation see the same steps.

Steps produced after the end-of-thinking marker (``</think>`` for DeepSeek-R1
distills) belong to the final answer. They are flagged ``in_thinking=False``
and are never used to trigger a stop.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

STEP_DELIMITER = "\n\n"
THINK_END = "</think>"


@dataclasses.dataclass(frozen=True)
class Step:
    index: int
    text: str
    end_token: int  # number of generated tokens up to and including the token that closed the step
    in_thinking: bool
    closed: bool = True  # False for the trailing piece that was never followed by a delimiter


def split_steps(text: str, delimiter: str = STEP_DELIMITER) -> list[str]:
    return text.split(delimiter)


class StepSegmenter:
    """Accumulates text pieces and emits completed steps at each delimiter."""

    def __init__(self, delimiter: str = STEP_DELIMITER, think_end: str = THINK_END):
        if not delimiter:
            raise ValueError("delimiter must be non-empty")
        self.delimiter = delimiter
        self.think_end = think_end
        self.reset()

    def reset(self) -> None:
        self.buffer = ""
        self.n_steps = 0
        self.thinking_done = False

    def feed(self, piece: str, n_tokens: int) -> list[Step]:
        """Add ``piece`` (text of the newest token(s)); ``n_tokens`` is the running token count."""
        self.buffer += piece
        out = []
        while True:
            i = self.buffer.find(self.delimiter)
            if i < 0:
                break
            out.append(self._emit(self.buffer[:i], n_tokens, closed=True))
            self.buffer = self.buffer[i + len(self.delimiter):]
        return out

    def flush(self, n_tokens: int) -> Step:
        """Emit the trailing piece (possibly empty), which has no closing delimiter.

        With it, the emitted texts are exactly ``str.split(delimiter)`` of the full text.
        """
        step = self._emit(self.buffer, n_tokens, closed=False)
        self.buffer = ""
        return step

    def _emit(self, text: str, n_tokens: int, closed: bool) -> Step:
        in_thinking = not self.thinking_done and self.think_end not in text
        if self.think_end in text:
            self.thinking_done = True
        step = Step(index=self.n_steps, text=text, end_token=n_tokens, in_thinking=in_thinking, closed=closed)
        self.n_steps += 1
        return step


class IncrementalDetokenizer:
    """Turns a growing list of token IDs into text deltas in O(1) decode work per token.

    Same scheme as vLLM's incremental detokenisation: decode a short window and
    hold back output while the window ends in an incomplete UTF-8 sequence.
    ``"".join(deltas)`` equals ``tokenizer.decode(ids)`` for byte-level BPE
    tokenizers; the unit tests check this on real traces when the tokenizer is cached.
    """

    def __init__(self, tokenizer, skip_special_tokens: bool = False):
        self.tok = tokenizer
        self.skip = skip_special_tokens
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        if len(full) > len(prefix) and not full.endswith("�"):
            delta = full[len(prefix):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            return delta
        return ""


def segment_token_ids(gen_ids: Sequence[int], tokenizer, delimiter: str = STEP_DELIMITER,
                      think_end: str = THINK_END) -> list[Step]:
    """Offline segmentation of a cached trace, using the same code path as live generation."""
    detok = IncrementalDetokenizer(tokenizer)
    seg = StepSegmenter(delimiter, think_end)
    steps: list[Step] = []
    for n, tid in enumerate(gen_ids, start=1):
        steps.extend(seg.feed(detok.push(tid), n))
    steps.append(seg.flush(len(gen_ids)))
    return steps


def segment_text_pieces(pieces: Iterable[str], delimiter: str = STEP_DELIMITER,
                        think_end: str = THINK_END) -> list[Step]:
    """Segmentation of a pre-tokenised list of text pieces (one piece = one token). Used in tests."""
    seg = StepSegmenter(delimiter, think_end)
    steps: list[Step] = []
    n = 0
    for n, piece in enumerate(pieces, start=1):
        steps.extend(seg.feed(piece, n))
    steps.append(seg.flush(n))
    return steps
