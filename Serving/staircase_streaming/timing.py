"""Timing records, summary statistics and a sampled GPU-memory monitor.

All timestamps are ``time.perf_counter()`` values. Records store them relative to
the request start ``t0`` so they can be serialised and compared across requests.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .backends import TokenEvent


@dataclass
class CallRecord:
    """One streamed backend call (a proposer run or one aggregator chunk)."""

    model_id: str
    start_s: float  # call issued, relative to request t0
    first_token_s: float | None = None
    end_s: float | None = None
    n_tokens: int = 0  # streamed (non-empty) token events
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    max_new_tokens: int | None = None

    @property
    def prefill_s(self) -> float | None:
        """Call start to first token (prefill + 1 decode step + queue/network)."""
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.start_s

    @property
    def decode_s(self) -> float | None:
        """First token to end of stream."""
        if self.first_token_s is None or self.end_s is None:
            return None
        return self.end_s - self.first_token_s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prefill_s"] = self.prefill_s
        d["decode_s"] = self.decode_s
        return d


def consume(
    events: Iterable[TokenEvent],
    record: CallRecord,
    t0: float,
    on_token: Callable[[TokenEvent], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[TokenEvent]:
    """Pass events through while filling ``record``; yields only token events.

    If ``should_stop`` returns True the underlying stream is closed (which aborts an
    HTTP request) and ``record.finish_reason`` is set to ``"cancelled"``.
    """
    it = iter(events)
    try:
        for ev in it:
            if ev.final:
                record.end_s = ev.t - t0
                record.finish_reason = ev.finish_reason
                record.prompt_tokens = ev.prompt_tokens
                record.completion_tokens = ev.completion_tokens
                record.cached_prompt_tokens = ev.cached_prompt_tokens
                continue
            if record.first_token_s is None:
                record.first_token_s = ev.t - t0
            record.n_tokens += 1
            if on_token is not None:
                on_token(ev)
            yield ev
            if should_stop is not None and should_stop():
                record.finish_reason = "cancelled"
                record.end_s = time.perf_counter() - t0
                break
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()
    if record.end_s is None:
        record.end_s = time.perf_counter() - t0


@dataclass
class ChunkRecord:
    """Staircase bookkeeping for aggregator chunk ``index`` (1-based)."""

    index: int
    proposer_boundary: int  # cumulative proposer tokens required for this chunk
    required_proposers: int  # N, or N - redundancy for chunk 1
    wait_s: float  # time the aggregator waited for proposers before this chunk
    proposer_tokens_seen: list[int]  # tokens of each proposer included in the prompt
    proposer_finished: list[bool]
    requested_tokens: int
    generated_tokens: int = 0


@dataclass
class MoAResult:
    """Output and timing of one MoA request (baseline or staircase)."""

    method: str
    query_id: str
    text: str
    ttft_s: float | None
    e2e_s: float
    aggregator_tokens: int
    proposers: list[CallRecord]
    aggregator_calls: list[CallRecord]
    proposer_texts: list[str] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    proposers_cancelled: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def tps(self) -> float | None:
        """Paper definition: aggregator tokens / (request start -> last token)."""
        return self.aggregator_tokens / self.e2e_s if self.e2e_s > 0 else None

    @property
    def aggregator_prefill_s(self) -> float:
        return sum(c.prefill_s or 0.0 for c in self.aggregator_calls)

    @property
    def aggregator_decode_s(self) -> float:
        return sum(c.decode_s or 0.0 for c in self.aggregator_calls)

    @property
    def aggregator_prompt_tokens(self) -> int | None:
        vals = [c.prompt_tokens for c in self.aggregator_calls]
        return None if any(v is None for v in vals) else sum(vals)  # type: ignore[misc]

    @property
    def aggregator_cached_tokens(self) -> int | None:
        vals = [c.cached_prompt_tokens for c in self.aggregator_calls]
        return None if any(v is None for v in vals) else sum(vals)  # type: ignore[misc]

    @property
    def proposer_prompt_tokens(self) -> int | None:
        vals = [c.prompt_tokens for c in self.proposers]
        return None if any(v is None for v in vals) else sum(vals)  # type: ignore[misc]

    @property
    def slowest_proposer_s(self) -> float | None:
        ends = [p.end_s for p in self.proposers if p.end_s is not None]
        return max(ends) if ends else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query_id": self.query_id,
            "text": self.text,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "tps": self.tps,
            "aggregator_tokens": self.aggregator_tokens,
            "aggregator_calls_n": len(self.aggregator_calls),
            "aggregator_prefill_s": self.aggregator_prefill_s,
            "aggregator_decode_s": self.aggregator_decode_s,
            "aggregator_prompt_tokens": self.aggregator_prompt_tokens,
            "aggregator_cached_tokens": self.aggregator_cached_tokens,
            "proposer_prompt_tokens": self.proposer_prompt_tokens,
            "slowest_proposer_s": self.slowest_proposer_s,
            "proposers_cancelled": self.proposers_cancelled,
            "proposers": [p.to_dict() for p in self.proposers],
            "aggregator_calls": [c.to_dict() for c in self.aggregator_calls],
            "chunks": [asdict(c) for c in self.chunks],
            "proposer_texts": self.proposer_texts,
            "settings": self.settings,
        }


# --------------------------------------------------------------------- statistics
def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default), ``q`` in [0, 100]."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float | None:
    return percentile(values, 50.0)


# --------------------------------------------------------------------- GPU memory
class GpuMemorySampler:
    """Samples ``nvidia-smi --query-gpu=memory.used`` in a background thread.

    Method label for reports: "sampled nvidia-smi memory.used (MiB, device-wide,
    includes all processes and vLLM's pre-allocated KV-cache pool)".
    """

    METHOD = "sampled nvidia-smi memory.used every {interval}s (device-wide MiB, incl. vLLM KV pool)"

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.available = shutil.which("nvidia-smi") is not None
        self.peak_mib: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def method(self) -> str:
        return self.METHOD.format(interval=self.interval_s)

    def sample_once(self) -> dict[int, int]:
        if not self.available:
            return {}
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        used = {}
        for line in out.strip().splitlines():
            idx, mib = (s.strip() for s in line.split(","))
            used[int(idx)] = int(mib)
        for idx, mib in used.items():
            self.peak_mib[idx] = max(self.peak_mib.get(idx, 0), mib)
        return used

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> GpuMemorySampler:
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict[int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return dict(self.peak_mib)
