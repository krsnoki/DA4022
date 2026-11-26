"""Proposers that stream concurrently into thread-safe buffers.

All proposers of one request share a single ``threading.Condition`` so the caller
can wait for "k of N proposers have reached boundary B" without polling.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, Prompt
from .timing import CallRecord, consume


@dataclass
class AgentSpec:
    """A model plus its generation settings."""

    model_id: str
    max_new_tokens: int
    gen_kwargs: dict[str, Any] = field(default_factory=dict)


class ProposerStream:
    """Runs one proposer in a background thread, buffering streamed tokens."""

    def __init__(self, backend: Backend, spec: AgentSpec, prompt: Prompt, t0: float,
                 cond: threading.Condition) -> None:
        self.backend = backend
        self.spec = spec
        self.prompt = prompt
        self.t0 = t0
        self.cond = cond
        self.tokens: list[str] = []
        self.finished = False
        self.error: BaseException | None = None
        self.record = CallRecord(model_id=spec.model_id, start_s=0.0,
                                 max_new_tokens=spec.max_new_tokens)
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> ProposerStream:
        self._thread.start()
        return self

    def _on_token(self, ev: Any) -> None:
        with self.cond:
            self.tokens.append(ev.text)
            self.cond.notify_all()

    def _run(self) -> None:
        self.record.start_s = time.perf_counter() - self.t0
        try:
            events = self.backend.stream(self.spec.model_id, self.prompt,
                                         self.spec.max_new_tokens, **self.spec.gen_kwargs)
            for _ in consume(events, self.record, self.t0, on_token=self._on_token,
                             should_stop=self._cancel.is_set):
                pass
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
            self.error = exc
        finally:
            with self.cond:
                self.finished = True
                self.cond.notify_all()

    # ------------------------------------------------------------------ queries
    # Callers must hold ``cond`` when calling these from a wait predicate.
    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def ready(self, boundary: int) -> bool:
        """Chunk with cumulative ``boundary`` tokens is available (or stream ended)."""
        return self.finished or len(self.tokens) >= boundary

    def text(self, start: int = 0, end: int | None = None) -> str:
        return "".join(self.tokens[start:end])

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


def start_proposers(backend: Backend, specs: Sequence[AgentSpec],
                    prompt_for: Callable[[AgentSpec], Prompt], t0: float
                    ) -> tuple[list[ProposerStream], threading.Condition]:
    """Start all proposers concurrently; returns the streams and their shared condition."""
    cond = threading.Condition()
    streams = [ProposerStream(backend, s, prompt_for(s), t0, cond) for s in specs]
    for s in streams:
        s.start()
    return streams, cond


def raise_on_error(streams: Sequence[ProposerStream]) -> None:
    """Re-raise the first proposer error, if any."""
    for s in streams:
        if s.error is not None:
            raise RuntimeError(f"proposer {s.spec.model_id} failed: {s.error!r}") from s.error
