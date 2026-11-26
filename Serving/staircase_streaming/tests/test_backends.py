"""MockBackend behaviour (token count, monotonic timing, determinism)."""

from __future__ import annotations

import time

from ..backends import MockBackend, MockModelSpec, TokenEvent, to_messages
from ..timing import CallRecord, consume, percentile

FAST = MockModelSpec(base_latency_s=0.001, prefill_s_per_token=0.0, decode_s_per_token=0.0005,
                     answer_tokens=1000)


def _collect(backend: MockBackend, **kw) -> list[TokenEvent]:
    return list(backend.stream("m", "hello world", **kw))


def test_yields_requested_number_of_tokens_then_one_final_event() -> None:
    events = _collect(MockBackend(default=FAST), max_new_tokens=17)
    tokens = [e for e in events if not e.final]
    assert len(tokens) == 17
    assert events[-1].final and sum(e.final for e in events) == 1
    assert events[-1].finish_reason == "length"
    assert events[-1].completion_tokens == 17
    assert events[-1].prompt_tokens == 2


def test_timestamps_are_monotonic() -> None:
    events = _collect(MockBackend(default=FAST), max_new_tokens=30)
    ts = [e.t for e in events]
    assert all(b >= a for a, b in zip(ts, ts[1:]))


def test_runs_are_deterministic() -> None:
    b = MockBackend(default=FAST)
    a1 = "".join(e.text for e in _collect(b, max_new_tokens=25))
    a2 = "".join(e.text for e in _collect(b, max_new_tokens=25))
    assert a1 == a2 and a1


def test_eos_after_answer_tokens_and_clean_ending() -> None:
    b = MockBackend(default=MockModelSpec(decode_s_per_token=0.0, answer_tokens=5))
    events = _collect(b, max_new_tokens=50)
    text = "".join(e.text for e in events)
    assert len([e for e in events if not e.final]) == 5
    assert events[-1].finish_reason == "stop"
    assert text.endswith("done.")


def test_continue_assistant_resumes_position() -> None:
    spec = MockModelSpec(decode_s_per_token=0.0, answer_tokens=10)
    b = MockBackend(default=spec)
    full = "".join(e.text for e in b.stream("m", "q", max_new_tokens=10))
    first = "".join(e.text for e in b.stream("m", "q", max_new_tokens=4))
    rest = "".join(e.text for e in b.stream("m", "q", max_new_tokens=10, continue_assistant=first))
    assert first + rest == full


def test_prefill_delay_scales_with_prompt_length() -> None:
    spec = MockModelSpec(base_latency_s=0.0, prefill_s_per_token=0.001, decode_s_per_token=0.0)
    b = MockBackend(default=spec)

    def first_token_latency(n_words: int) -> float:
        t0 = time.perf_counter()
        ev = next(iter(b.stream("m", "w " * n_words, max_new_tokens=1)))
        return ev.t - t0

    assert first_token_latency(100) > first_token_latency(5) + 0.05


def test_consume_fills_call_record() -> None:
    b = MockBackend(default=FAST)
    t0 = time.perf_counter()
    rec = CallRecord(model_id="m", start_s=0.0)
    toks = list(consume(b.stream("m", "a b c", max_new_tokens=8), rec, t0))
    assert len(toks) == rec.n_tokens == rec.completion_tokens == 8
    assert rec.prompt_tokens == 3
    assert rec.prefill_s is not None and rec.decode_s is not None and rec.decode_s >= 0


def test_to_messages_and_percentile() -> None:
    assert to_messages("hi") == [{"role": "user", "content": "hi"}]
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) is None
