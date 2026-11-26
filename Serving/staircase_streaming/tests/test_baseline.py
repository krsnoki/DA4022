"""Baseline MoA on the mock backend."""

from __future__ import annotations

from ..moa_baseline import run_moa_baseline
from ..prompts import MOA_AGGREGATOR_SYSTEM, aggregator_messages
from .mock_setup import SPECS, make_backend, make_moa


def test_baseline_output_and_timing_fields() -> None:
    res = run_moa_baseline(make_backend(), make_moa(), "What is MoA?", "q1")
    assert res.text.strip() and res.text.endswith("done.")
    assert res.aggregator_tokens == SPECS["agg"].answer_tokens
    assert len(res.aggregator_calls) == 1
    assert [p.n_tokens for p in res.proposers] == [40, 60, 80]
    assert res.ttft_s is not None and 0 < res.ttft_s < res.e2e_s
    assert res.tps is not None and res.tps > 0
    assert res.aggregator_prompt_tokens and res.proposer_prompt_tokens


def test_baseline_ttft_is_slowest_proposer_plus_aggregator_prefill() -> None:
    res = run_moa_baseline(make_backend(), make_moa(), "What is MoA?", "q1")
    agg = SPECS["agg"]
    expected_prefill = agg.base_latency_s + agg.prefill_s_per_token * res.aggregator_prompt_tokens
    expected = res.slowest_proposer_s + expected_prefill
    assert abs(res.ttft_s - expected) < 0.03 + 0.1 * expected
    # and the aggregator really waited for the slowest proposer
    assert res.aggregator_calls[0].start_s >= res.slowest_proposer_s - 1e-3


def test_aggregator_prompt_uses_table2_template() -> None:
    msgs = aggregator_messages("Q?", ["alpha", "", "gamma"])
    assert msgs[0]["role"] == "system" and msgs[1] == {"role": "user", "content": "Q?"}
    assert msgs[0]["content"].startswith(MOA_AGGREGATOR_SYSTEM)
    assert msgs[0]["content"].endswith("Responses from models:\n1. alpha\n2. gamma")
