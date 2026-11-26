"""Staircase streaming for MoA (paper Algorithm 1; hyperparameters from Appendix A).

Chunk schedule (1-based chunk index ``j``):

* proposer chunk sizes ``C_P,j``: ``first_chunk_size``, ``second_chunk_size``, then
  ``chunk_size``. The cumulative sum gives the proposer boundary ``B_j``.
* aggregator chunk sizes ``C_A,j``: ``first_chunk_size``, ``second_chunk_size``, then
  ``aggregator_chunk_size``.

Before aggregator chunk ``j`` the aggregator waits until enough proposers are
*ready* for chunk ``j`` (buffer reached ``B_j`` or stream finished): all ``N`` of
them, or ``N - redundancy`` for the first chunk. It then rebuilds its prompt and
continues its own partial answer for ``C_A,j`` tokens.

Simplification (see PAPER_CARD): proposers stream continuously in their own
threads instead of being restarted per chunk.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .backends import Backend
from .moa_baseline import MoAConfig, StreamSink
from .prompts import aggregator_messages, prefix_cache_messages, proposer_messages
from .proposers import ProposerStream, raise_on_error, start_proposers
from .timing import CallRecord, ChunkRecord, MoAResult, consume


@dataclass(frozen=True)
class StaircaseConfig:
    """Paper Appendix A defaults."""

    first_chunk_size: int = 8
    second_chunk_size: int = 128
    chunk_size: int = 256
    aggregator_chunk_size: int = 128
    redundancy: int = 2
    prefix_cache: bool = False

    def validate(self, n_proposers: int) -> None:
        sizes = (self.first_chunk_size, self.second_chunk_size, self.chunk_size,
                 self.aggregator_chunk_size)
        if min(sizes) <= 0:
            raise ValueError(f"chunk sizes must be positive, got {sizes}")
        if not 0 <= self.redundancy < n_proposers:
            raise ValueError(
                f"redundancy must satisfy 0 <= r < n_proposers ({n_proposers}), got {self.redundancy}"
            )

    def proposer_chunk(self, j: int) -> int:
        return (self.first_chunk_size, self.second_chunk_size)[j - 1] if j <= 2 else self.chunk_size

    def aggregator_chunk(self, j: int) -> int:
        if j <= 2:
            return (self.first_chunk_size, self.second_chunk_size)[j - 1]
        return self.aggregator_chunk_size

    def proposer_boundary(self, j: int) -> int:
        """Cumulative proposer tokens required for chunk ``j``."""
        return sum(self.proposer_chunk(k) for k in range(1, j + 1))

    def required_proposers(self, j: int, n_proposers: int) -> int:
        return n_proposers - self.redundancy if j == 1 else n_proposers


def _wait_ready(streams: list[ProposerStream], cond, boundary: int, need: int,
                timeout_s: float) -> None:
    def enough() -> bool:
        return sum(s.ready(boundary) for s in streams) >= need

    with cond:
        if not cond.wait_for(enough, timeout=timeout_s):
            raise TimeoutError(f"proposers not ready for boundary {boundary} within {timeout_s}s")


def run_staircase(backend: Backend, cfg: MoAConfig, sc: StaircaseConfig, query: str,
                  query_id: str = "", on_text: Callable[[str], None] | None = None) -> MoAResult:
    """Run one staircase-streaming MoA request and return its output and timings."""
    n = len(cfg.proposers)
    sc.validate(n)
    t0 = time.perf_counter()
    streams, cond = start_proposers(
        backend, cfg.proposer_specs(), lambda s: proposer_messages(query, cfg.proposer_system), t0
    )
    agg = cfg.aggregator
    sink = StreamSink(on_text)
    calls: list[CallRecord] = []
    chunks: list[ChunkRecord] = []
    emitted = [0] * n  # prefix-cache mode: tokens of each proposer already in a block
    blocks: list[list[str]] = []
    ttft: float | None = None
    agg_tokens = 0
    j = 0
    try:
        while agg_tokens < agg.max_new_tokens:
            j += 1
            boundary = sc.proposer_boundary(j)
            need = sc.required_proposers(j, n)
            wait_start = time.perf_counter()
            _wait_ready(streams, cond, boundary, need, cfg.timeout_s)
            wait_s = time.perf_counter() - wait_start

            with cond:  # consistent snapshot of all buffers
                seen = [s.n_tokens for s in streams]
                finished = [s.finished for s in streams]
                if sc.prefix_cache:
                    block = []
                    for i, s in enumerate(streams):
                        end = s.n_tokens if s.finished else min(s.n_tokens, boundary)
                        block.append(s.text(emitted[i], end))
                        emitted[i] = max(emitted[i], end)
                    blocks.append(block)
                    seen = list(emitted)
                else:
                    texts = [s.text() for s in streams]
            raise_on_error(streams)

            messages = (prefix_cache_messages(query, blocks) if sc.prefix_cache
                        else aggregator_messages(query, texts))
            size = min(sc.aggregator_chunk(j), agg.max_new_tokens - agg_tokens)
            chunk = ChunkRecord(index=j, proposer_boundary=boundary, required_proposers=need,
                                wait_s=wait_s, proposer_tokens_seen=seen,
                                proposer_finished=finished, requested_tokens=size)
            record = CallRecord(model_id=agg.model_id, start_s=time.perf_counter() - t0,
                                max_new_tokens=size)
            answer = "".join(sink.parts)
            events = backend.stream(agg.model_id, messages, size,
                                    continue_assistant=answer or None, **agg.gen_kwargs)
            for ev in consume(events, record, t0):
                if ttft is None:
                    ttft = ev.t - t0
                sink(ev.text)
            chunk.generated_tokens = record.n_tokens
            agg_tokens += record.n_tokens
            calls.append(record)
            chunks.append(chunk)
            if record.finish_reason == "stop" or record.n_tokens == 0:
                break
        e2e = calls[-1].end_s if calls and calls[-1].end_s is not None else time.perf_counter() - t0
    finally:
        cancelled = 0
        for s in streams:
            if not s.finished:
                s.cancel()
                cancelled += 1
        for s in streams:
            s.join(timeout=cfg.timeout_s)
    raise_on_error(streams)

    return MoAResult(
        method="staircase_prefix_cache" if sc.prefix_cache else "staircase",
        query_id=query_id,
        text="".join(sink.parts),
        ttft_s=ttft,
        e2e_s=e2e,
        aggregator_tokens=agg_tokens,
        proposers=[s.record for s in streams],
        aggregator_calls=calls,
        proposer_texts=[s.text() for s in streams],
        chunks=chunks,
        proposers_cancelled=cancelled,
        settings={"proposers": [p.model_id for p in cfg.proposers],
                  "aggregator": agg.model_id, **asdict(sc)},
    )
