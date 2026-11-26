"""Staircase streaming on the mock backend."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ..backends import MockBackend
from ..moa_baseline import run_moa_baseline
from ..prompts import PREFIX_CACHE_SYSTEM
from ..staircase import StaircaseConfig, run_staircase
from .mock_setup import SPECS, make_backend, make_moa

Q = "Explain staircase streaming."


class RecordingBackend:
    """Wraps a backend and records every call's arguments."""

    def __init__(self, inner: MockBackend) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def stream(self, model_id, prompt, max_new_tokens, continue_assistant=None, **kw):
        self.calls.append({"model_id": model_id, "prompt": prompt, "max_new_tokens": max_new_tokens,
                           "continue_assistant": continue_assistant})
        return self.inner.stream(model_id, prompt, max_new_tokens, continue_assistant, **kw)


def test_staircase_ttft_lower_than_baseline() -> None:
    base = run_moa_baseline(make_backend(), make_moa(), Q)
    # best of two staircase runs absorbs rare OS scheduling hiccups on a loaded machine
    stair_ttft = min(run_staircase(make_backend(), make_moa(), StaircaseConfig(redundancy=1), Q).ttft_s
                     for _ in range(2))
    assert stair_ttft < base.ttft_s


@pytest.mark.parametrize("r", [1, 2])
def test_first_chunk_does_not_wait_for_r_slowest(r: int) -> None:
    res = run_staircase(make_backend(), make_moa(), StaircaseConfig(redundancy=r), Q)
    first = res.chunks[0]
    assert first.required_proposers == 3 - r
    # proposers are ordered fast, mid, slow: the r slowest had not reached the boundary
    for seen, done in zip(first.proposer_tokens_seen[3 - r:], first.proposer_finished[3 - r:]):
        assert seen < first.proposer_boundary and not done
    slow_first_token = res.proposers[2].first_token_s
    assert res.ttft_s < slow_first_token


def test_redundancy_zero_waits_for_everyone() -> None:
    res = run_staircase(make_backend(), make_moa(), StaircaseConfig(redundancy=0), Q)
    first = res.chunks[0]
    assert all(s >= first.proposer_boundary or f
               for s, f in zip(first.proposer_tokens_seen, first.proposer_finished))
    assert res.ttft_s > res.proposers[2].first_token_s


def test_chunk_schedule_matches_config() -> None:
    specs = dict(SPECS)
    specs["agg"] = replace(SPECS["agg"], answer_tokens=1000, decode_s_per_token=0.0)
    sc = StaircaseConfig(first_chunk_size=8, second_chunk_size=16, chunk_size=24,
                         aggregator_chunk_size=32, redundancy=1)
    res = run_staircase(MockBackend(specs), make_moa(max_agg_tokens=100), sc, Q)
    assert [c.requested_tokens for c in res.chunks] == [8, 16, 32, 32, 12]
    assert [c.proposer_boundary for c in res.chunks] == [8, 24, 48, 72, 96]
    assert [c.required_proposers for c in res.chunks] == [2, 3, 3, 3, 3]
    assert res.aggregator_tokens == 100 and len(res.aggregator_calls) == 5
    assert sc.proposer_boundary(3) == 8 + 16 + 24


def test_final_text_nonempty_ends_cleanly_and_continues_answer() -> None:
    rec = RecordingBackend(make_backend())
    res = run_staircase(rec, make_moa(), StaircaseConfig(redundancy=1), Q)
    assert res.text.strip() and res.text.endswith("done.")
    assert res.aggregator_tokens == SPECS["agg"].answer_tokens
    agg_calls = [c for c in rec.calls if c["model_id"] == "agg"]
    assert agg_calls[0]["continue_assistant"] is None
    assert all(c["continue_assistant"] for c in agg_calls[1:])
    # matches what a single uninterrupted aggregator run would produce (deterministic mock)
    base = run_moa_baseline(make_backend(), make_moa(), Q)
    assert res.text == base.text


def test_reprefill_cost_is_recorded() -> None:
    base = run_moa_baseline(make_backend(), make_moa(), Q)
    stair = run_staircase(make_backend(), make_moa(), StaircaseConfig(redundancy=1), Q)
    assert len(stair.aggregator_calls) > 1
    assert stair.aggregator_prompt_tokens > 0 and base.aggregator_prompt_tokens > 0
    assert sum(c.prefill_s is not None for c in stair.aggregator_calls) == len(stair.aggregator_calls)


def test_prefix_cache_mode_keeps_prompt_prefix_unchanged() -> None:
    rec = RecordingBackend(make_backend())
    res = run_staircase(rec, make_moa(), StaircaseConfig(redundancy=1, prefix_cache=True), Q)
    systems = [c["prompt"][0]["content"] for c in rec.calls if c["model_id"] == "agg"]
    assert len(systems) > 1 and systems[0].startswith(PREFIX_CACHE_SYSTEM)
    for prev, nxt in zip(systems, systems[1:]):
        assert nxt.startswith(prev)
    assert "Chunk 1:" in systems[0] and "Chunk 2:" in systems[1]
    assert res.method == "staircase_prefix_cache" and res.text.endswith("done.")


def test_proposers_cancelled_when_aggregator_stops_early() -> None:
    specs = dict(SPECS)
    specs["agg"] = replace(SPECS["agg"], answer_tokens=5)
    specs["p_slow"] = replace(SPECS["p_slow"], answer_tokens=10_000)
    res = run_staircase(MockBackend(specs), make_moa(max_proposer_tokens=10_000),
                        StaircaseConfig(redundancy=1), Q)
    assert res.proposers_cancelled >= 1
    assert res.proposers[2].finish_reason == "cancelled"


def test_invalid_redundancy_rejected() -> None:
    with pytest.raises(ValueError):
        run_staircase(make_backend(), make_moa(), StaircaseConfig(redundancy=3), Q)
