"""Baseline Mixture-of-Agents: all proposers finish, then the aggregator streams.

TTFT ≈ slowest proposer's full generation + aggregator prefill (paper Eq. 1).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .backends import Backend
from .prompts import aggregator_messages, proposer_messages
from .proposers import AgentSpec, raise_on_error, start_proposers
from .timing import CallRecord, MoAResult, consume


@dataclass
class MoAConfig:
    """Which models play which role, and their generation settings."""

    proposers: list[AgentSpec]
    aggregator: AgentSpec
    proposer_system: str | None = None
    timeout_s: float = 600.0
    seed: int | None = None  # per-request base seed for proposer sampling

    def proposer_specs(self) -> list[AgentSpec]:
        """Proposer specs with a distinct deterministic seed each (if ``seed`` is set)."""
        if self.seed is None:
            return list(self.proposers)
        return [
            AgentSpec(p.model_id, p.max_new_tokens, {**p.gen_kwargs, "seed": self.seed + i})
            for i, p in enumerate(self.proposers)
        ]


@dataclass
class StreamSink:
    """Optional callback receiving aggregator text as it is streamed to the user."""

    callback: Callable[[str], None] | None = None
    parts: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> None:
        self.parts.append(text)
        if self.callback is not None:
            self.callback(text)


def run_moa_baseline(backend: Backend, cfg: MoAConfig, query: str, query_id: str = "",
                     on_text: Callable[[str], None] | None = None) -> MoAResult:
    """Run one baseline MoA request and return its output and timings."""
    t0 = time.perf_counter()
    streams, cond = start_proposers(
        backend, cfg.proposer_specs(), lambda s: proposer_messages(query, cfg.proposer_system), t0
    )
    with cond:
        done = cond.wait_for(lambda: all(s.finished for s in streams), timeout=cfg.timeout_s)
    if not done:
        for s in streams:
            s.cancel()
        raise TimeoutError(f"proposers did not finish within {cfg.timeout_s}s")
    for s in streams:
        s.join()
    raise_on_error(streams)

    texts = [s.text() for s in streams]
    agg = cfg.aggregator
    record = CallRecord(model_id=agg.model_id, start_s=time.perf_counter() - t0,
                        max_new_tokens=agg.max_new_tokens)
    sink = StreamSink(on_text)
    events = backend.stream(agg.model_id, aggregator_messages(query, texts),
                            agg.max_new_tokens, **agg.gen_kwargs)
    for ev in consume(events, record, t0):
        sink(ev.text)
    e2e = (record.end_s if record.end_s is not None else time.perf_counter() - t0)

    return MoAResult(
        method="baseline",
        query_id=query_id,
        text="".join(sink.parts),
        ttft_s=record.first_token_s,
        e2e_s=e2e,
        aggregator_tokens=record.n_tokens,
        proposers=[s.record for s in streams],
        aggregator_calls=[record],
        proposer_texts=texts,
        settings={"proposers": [p.model_id for p in cfg.proposers],
                  "aggregator": agg.model_id},
    )
