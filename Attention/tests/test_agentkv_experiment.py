import pytest

from Dataloaders.synthetic.agent_memory import episodes, exact_code_match
from Performance_Metrics.inference import summarize
from Attention.experiments.agentkv.run import validate_config


def test_data_deterministic_and_seed_separated():
    first = list(episodes(3, 42, 2))
    assert first == list(episodes(3, 42, 2))
    assert first != list(episodes(3, 43, 2))
    for episode in first:
        assert episode.expected in episode.first_prompt
        assert episode.expected not in episode.continuation
        assert "<tool_response>" in episode.continuation


def test_exact_match_does_not_credit_reasoning_or_unfinished_trace():
    assert exact_code_match("checking</think>\n123456<|im_end|>", "123456")
    assert not exact_code_match("123456", "123456")
    assert not exact_code_match("123456</think>654321", "123456")
    assert not exact_code_match("</think>The code is 123456.", "123456")


def test_summary_counts_failures_and_does_not_invent_cpu_memory_or_tail_latency():
    row = dict(status="partial", correct=True, prefill_seconds=2., decode_seconds=3.,
               output_tokens=9, peak_memory_mb=None, kv_bytes=100, query_buffer_bytes=20,
               eviction_seconds=0.1)
    summary = summarize([row, dict(status="failed")])
    assert summary["failures"] == 1
    assert summary["num_examples"] == 1
    assert summary["throughput"] == 3
    assert summary["latency_median_ms"] == 5000
    assert summary["latency_p95_ms"] is None
    assert summary["peak_memory_mb"] is None
    assert summary["energy_j"] is None
    assert summary["quality_value"] is None
    assert summary["quality_completed_only"] == 1.0


def test_config_rejects_missing_warmup():
    with pytest.raises(ValueError, match="warmup"):
        validate_config(dict(num_examples=1, max_new_tokens=2, prefill_chunk_size=4, warmup_examples=0))
