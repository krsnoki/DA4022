"""Shared mock MoA setup: 3 proposers with different speeds and one aggregator."""

from __future__ import annotations

from ..backends import MockBackend, MockModelSpec
from ..moa_baseline import MoAConfig
from ..proposers import AgentSpec

SPECS = {
    "p_fast": MockModelSpec(base_latency_s=0.005, prefill_s_per_token=0.0, decode_s_per_token=0.001, answer_tokens=40),
    "p_mid": MockModelSpec(base_latency_s=0.020, prefill_s_per_token=0.0, decode_s_per_token=0.002, answer_tokens=60),
    "p_slow": MockModelSpec(base_latency_s=0.080, prefill_s_per_token=0.0, decode_s_per_token=0.003, answer_tokens=80),
    "agg": MockModelSpec(base_latency_s=0.005, prefill_s_per_token=0.0001, decode_s_per_token=0.001, answer_tokens=50),
}


def make_backend() -> MockBackend:
    return MockBackend(SPECS)


def make_moa(max_proposer_tokens: int = 200, max_agg_tokens: int = 200) -> MoAConfig:
    return MoAConfig(
        proposers=[AgentSpec(m, max_proposer_tokens) for m in ("p_fast", "p_mid", "p_slow")],
        aggregator=AgentSpec("agg", max_agg_tokens),
        seed=42,
    )
