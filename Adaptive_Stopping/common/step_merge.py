"""Minimal step size k (Algorithm 1 of the ST-ES proceedings version).

The proceedings-format paper adds one requirement to Algorithm 1: "Generate step
s_i using M, alpha, where |s_i| > k". A delimiter therefore closes a step only
once the step has more than k tokens; a shorter piece is merged with what follows.
arXiv v2 and the authors' released code split at every delimiter, and the paper
gives no value for k, so k is a parameter here (None or 0 keeps plain splitting).

``|s_i|`` is measured in generated tokens from the end of the previous step up to
and including the token that closes this one (the delimiter included).
``StepMerger`` works on the stream of raw steps that ``StepSegmenter`` emits, so
the live loop and offline replay merge identically. ``merge_steps`` is the offline form.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from Adaptive_Stopping.common.steps import STEP_DELIMITER, Step, segment_token_ids


class StepMerger:
    """Merges raw steps until a closed step is longer than ``min_step_tokens`` tokens."""

    def __init__(self, min_step_tokens: int | None, delimiter: str = STEP_DELIMITER):
        if min_step_tokens is not None and (not isinstance(min_step_tokens, int) or min_step_tokens < 0):
            raise ValueError("min_step_tokens must be None or an int >= 0")
        self.k = min_step_tokens
        self.delimiter = delimiter
        self.reset()

    def reset(self) -> None:
        self.parts: list[Step] = []
        self.start = 0  # token count at the end of the last emitted step
        self.n = 0

    def _emit(self, closed: bool) -> Step:
        parts = self.parts
        step = Step(index=self.n, text=self.delimiter.join(p.text for p in parts), end_token=parts[-1].end_token,
                    in_thinking=all(p.in_thinking for p in parts), closed=closed)
        self.n += 1
        self.start = step.end_token
        self.parts = []
        return step

    def push(self, raw: Step) -> Step | None:
        """Feed one closed raw step; returns a merged step once it is long enough, else None."""
        self.parts.append(raw)
        if self.k is None or (raw.closed and raw.end_token - self.start > self.k):
            return self._emit(closed=raw.closed)
        return None

    def flush(self, tail: Step) -> Step:
        """End of generation: merge whatever is pending with the trailing (unclosed) piece."""
        self.parts.append(tail)
        return self._emit(closed=False)


def merge_steps(steps: Sequence[Step], min_step_tokens: int | None) -> list[Step]:
    """Offline merge of a full segmentation (as returned by ``segment_token_ids``; last item = trailing piece)."""
    if not steps:
        return []
    m = StepMerger(min_step_tokens)
    out = [s for s in (m.push(raw) for raw in steps[:-1]) if s is not None]
    out.append(m.flush(steps[-1]))
    return out


def segment_with_min_size(min_step_tokens: int | None):
    """A ``steps_fn`` for ``replay_lib.prepare``: segment, then merge short steps."""

    def steps_fn(gen_ids: Sequence[int], tokenizer) -> list[Step]:
        return merge_steps(segment_token_ids(gen_ids, tokenizer), min_step_tokens)

    return steps_fn


def iter_merged(raw_steps: Iterable[Step], min_step_tokens: int | None) -> Iterable[Step]:
    """Online form for a stream of closed raw steps (the live loop calls ``StepMerger.push`` directly)."""
    m = StepMerger(min_step_tokens)
    for raw in raw_steps:
        s = m.push(raw)
        if s is not None:
            yield s
