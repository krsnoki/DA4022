"""Token-streaming backends behind one small interface.

Every backend implements :class:`Backend`. ``stream`` yields zero or more token
events followed by exactly one *final* event (``final=True``, empty text) that
carries the finish reason and, when the backend knows them, token usage counts.

* :class:`MockBackend`: deterministic fake tokens with simulated prefill and decode
  delays, used for tests and for dry-running the whole pipeline on CPU.
* :class:`OpenAICompatBackend`: streams from OpenAI-compatible HTTP servers
  (vLLM ``vllm serve``; also SGLang or a hosted API). Stdlib only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

Prompt = str | list[dict[str, str]]


@dataclass(frozen=True)
class TokenEvent:
    """One streamed token (or the final bookkeeping event of a call).

    ``t`` is a ``time.perf_counter()`` timestamp taken when the token was received.
    The first non-final event of a call therefore marks the end of its prefill
    (plus one decode step and, for HTTP backends, network/queueing time).
    """

    text: str
    t: float
    final: bool = False
    finish_reason: str | None = None  # "stop" (EOS) | "length" | "cancelled" | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None


class Backend(Protocol):
    """Streaming text-generation interface used by MoA and staircase code."""

    def stream(
        self,
        model_id: str,
        prompt: Prompt,
        max_new_tokens: int,
        continue_assistant: str | None = None,
        **gen_kwargs: Any,
    ) -> Iterator[TokenEvent]:
        """Stream up to ``max_new_tokens`` tokens.

        ``prompt`` is a plain string (treated as one user message) or a list of chat
        messages. ``continue_assistant`` continues a partial assistant answer: it is
        appended as the final assistant message and rendered without a generation
        prompt (``continue_final_message=True``), so the model resumes mid-answer.
        """
        ...


def to_messages(prompt: Prompt) -> list[dict[str, str]]:
    """Normalise a prompt to a list of chat messages."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


# --------------------------------------------------------------------------- mock
_VOCAB = (
    "the model answer is based on careful analysis of each response and combines "
    "accurate details into one clear reply with useful steps"
).split()


def mock_word_count(text: str) -> int:
    """The mock 'tokenizer': one token per whitespace-separated word."""
    return len(text.split())


@dataclass(frozen=True)
class MockModelSpec:
    """Timing and length behaviour of one fake model.

    Prefill time = ``base_latency_s + prefill_s_per_token * prompt_tokens``; every
    token after the first costs ``decode_s_per_token``. The model emits EOS after
    ``answer_tokens`` tokens of its answer in total (counting ``continue_assistant``).
    """

    base_latency_s: float = 0.002
    prefill_s_per_token: float = 0.00002
    decode_s_per_token: float = 0.001
    answer_tokens: int = 64


class MockBackend:
    """Deterministic fake backend: same inputs give the same token texts."""

    def __init__(
        self,
        specs: Mapping[str, MockModelSpec] | None = None,
        default: MockModelSpec | None = None,
    ) -> None:
        self.specs = dict(specs or {})
        self.default = default or MockModelSpec()

    def spec(self, model_id: str) -> MockModelSpec:
        return self.specs.get(model_id, self.default)

    @staticmethod
    def token_text(model_id: str, position: int, answer_tokens: int) -> str:
        """Deterministic token text for ``position`` of ``model_id``'s answer."""
        if position == answer_tokens - 1:
            return "done."
        word = _VOCAB[zlib.crc32(f"{model_id}:{position}".encode()) % len(_VOCAB)]
        return f"{word} "

    def stream(
        self,
        model_id: str,
        prompt: Prompt,
        max_new_tokens: int,
        continue_assistant: str | None = None,
        **gen_kwargs: Any,
    ) -> Iterator[TokenEvent]:
        spec = self.spec(model_id)
        messages = to_messages(prompt)
        prompt_tokens = sum(mock_word_count(m["content"]) for m in messages)
        already = mock_word_count(continue_assistant) if continue_assistant else 0
        prompt_tokens += already

        time.sleep(spec.base_latency_s + spec.prefill_s_per_token * prompt_tokens)
        produced = 0
        finish = "length"
        for k in range(max_new_tokens):
            position = already + k
            if position >= spec.answer_tokens:
                finish = "stop"
                break
            if k > 0:
                time.sleep(spec.decode_s_per_token)
            produced += 1
            yield TokenEvent(self.token_text(model_id, position, spec.answer_tokens),
                             time.perf_counter())
        else:
            if already + produced >= spec.answer_tokens:
                finish = "stop"
        yield TokenEvent("", time.perf_counter(), final=True, finish_reason=finish,
                         prompt_tokens=prompt_tokens, completion_tokens=produced)


# ------------------------------------------------------------ OpenAI-compatible
class OpenAICompatBackend:
    """Streams chat completions from OpenAI-compatible servers (e.g. ``vllm serve``).

    ``endpoints`` maps each logical model id to a base URL such as
    ``http://127.0.0.1:8001/v1``; the id is sent as the ``model`` field, so vLLM
    servers must be started with ``--served-model-name <id>``.

    ``continue_assistant`` uses vLLM's chat-endpoint extensions
    ``add_generation_prompt=False`` + ``continue_final_message=True``: the server's
    own chat template renders the partial answer as an open assistant turn.
    Usage (prompt/completion and, if enabled, cached prompt tokens) is requested with
    ``stream_options.include_usage``.
    """

    def __init__(self, endpoints: Mapping[str, str], timeout_s: float = 600.0,
                 api_key: str | None = None) -> None:
        self.endpoints = {k: v.rstrip("/") for k, v in endpoints.items()}
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _request(self, url: str, body: dict[str, Any] | None = None) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        return urllib.request.Request(url, data=data, headers=headers,
                                      method="GET" if body is None else "POST")

    def build_body(self, model_id: str, prompt: Prompt, max_new_tokens: int,
                   continue_assistant: str | None, gen_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """JSON body for ``/chat/completions`` (exposed for tests)."""
        messages = to_messages(prompt)
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **gen_kwargs,
        }
        if continue_assistant is not None:
            body["messages"] = messages + [{"role": "assistant", "content": continue_assistant}]
            body["add_generation_prompt"] = False
            body["continue_final_message"] = True
        return body

    def stream(
        self,
        model_id: str,
        prompt: Prompt,
        max_new_tokens: int,
        continue_assistant: str | None = None,
        **gen_kwargs: Any,
    ) -> Iterator[TokenEvent]:
        if model_id not in self.endpoints:
            raise KeyError(f"no endpoint configured for model {model_id!r}")
        body = self.build_body(model_id, prompt, max_new_tokens, continue_assistant, gen_kwargs)
        req = self._request(f"{self.endpoints[model_id]}/chat/completions", body)
        finish: str | None = None
        usage: dict[str, Any] = {}
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{model_id}: HTTP {exc.code}: {detail}") from exc
        try:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield TokenEvent(text, time.perf_counter())
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        finally:
            resp.close()
        details = usage.get("prompt_tokens_details") or {}
        yield TokenEvent("", time.perf_counter(), final=True, finish_reason=finish,
                         prompt_tokens=usage.get("prompt_tokens"),
                         completion_tokens=usage.get("completion_tokens"),
                         cached_prompt_tokens=details.get("cached_tokens"))

    # ------------------------------------------------------------------ helpers
    def get_json(self, url: str, timeout_s: float = 10.0) -> Any:
        with urllib.request.urlopen(self._request(url), timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_models(self) -> dict[str, list[str]]:
        """``GET /models`` for every endpoint; raises if a model id is not served."""
        served: dict[str, list[str]] = {}
        for model_id, base in self.endpoints.items():
            ids = [m["id"] for m in self.get_json(f"{base}/models").get("data", [])]
            if model_id not in ids:
                raise RuntimeError(f"{base} serves {ids}, not {model_id!r}")
            served[model_id] = ids
        return served

    def server_versions(self) -> dict[str, str | None]:
        """vLLM exposes ``GET /version`` at the server root; None if unavailable."""
        out: dict[str, str | None] = {}
        for model_id, base in self.endpoints.items():
            root = base[:-3] if base.endswith("/v1") else base
            try:
                out[model_id] = self.get_json(f"{root}/version").get("version")
            except (urllib.error.URLError, OSError, ValueError):
                out[model_id] = None
        return out
