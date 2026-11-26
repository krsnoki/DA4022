"""Smoke test against real running vLLM servers (slow; skipped by default).

Enable on the GPU machine after ``launch_vllm ... start``::

    STAIRCASE_LIVE_CONFIG=Serving/staircase_streaming/configs/default.yaml \
        python -m pytest Serving/staircase_streaming/tests -m slow -q
"""

from __future__ import annotations

import os

import pytest

from ..config import build_backend, load_config

LIVE = os.environ.get("STAIRCASE_LIVE_CONFIG")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not LIVE, reason="set STAIRCASE_LIVE_CONFIG to run against live servers"),
]


def test_every_server_generates_a_few_tokens() -> None:
    cfg = load_config(LIVE)
    backend = build_backend(cfg)
    backend.check_models()
    for model_id in cfg["servers"]:
        events = list(backend.stream(model_id, "Say hello in five words.", 8, temperature=0.0))
        assert [e for e in events if not e.final], f"{model_id} produced no tokens"
        assert events[-1].completion_tokens and events[-1].prompt_tokens


def test_aggregator_continues_partial_answer() -> None:
    cfg = load_config(LIVE)
    backend = build_backend(cfg)
    agg = cfg["moa"]["aggregator"]
    msgs = [{"role": "user", "content": "List three primary colors."}]
    first = "".join(e.text for e in backend.stream(agg, msgs, 6, temperature=0.0) if not e.final)
    rest = list(backend.stream(agg, msgs, 6, continue_assistant=first, temperature=0.0))
    assert first and any(not e.final and e.text for e in rest)
