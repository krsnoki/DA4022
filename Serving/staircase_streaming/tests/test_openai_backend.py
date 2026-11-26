"""OpenAICompatBackend against a fake OpenAI-compatible streaming server."""

from __future__ import annotations

from ..backends import OpenAICompatBackend
from ..launch_vllm import build_command
from ..moa_baseline import MoAConfig, run_moa_baseline
from ..proposers import AgentSpec
from ..staircase import StaircaseConfig, run_staircase
from .fake_openai_server import ANSWER_TOKENS, FakeServer


def _backend(server: FakeServer) -> OpenAICompatBackend:
    return OpenAICompatBackend({m: server.base_url for m in ANSWER_TOKENS}, timeout_s=30)


def _moa() -> MoAConfig:
    return MoAConfig(proposers=[AgentSpec("fast", 100), AgentSpec("slow", 100)],
                     aggregator=AgentSpec("agg", 100, {"temperature": 0.0}), seed=7)


def test_stream_tokens_usage_and_finish() -> None:
    with FakeServer() as srv:
        events = list(_backend(srv).stream("fast", "hello there", 10, temperature=0.5))
    toks = [e for e in events if not e.final]
    assert len(toks) == 10 and toks[0].text == "w0 "
    final = events[-1]
    assert final.final and final.finish_reason == "length"
    assert final.prompt_tokens == 2 and final.completion_tokens == 10
    assert final.cached_prompt_tokens == 0
    body = srv.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.5 and body["max_tokens"] == 10


def test_continue_assistant_request_shape() -> None:
    with FakeServer() as srv:
        events = list(_backend(srv).stream("agg", [{"role": "user", "content": "q"}], 50,
                                           continue_assistant="w0 w1 w2 "))
    body = srv.bodies[0]
    assert body["messages"][-1] == {"role": "assistant", "content": "w0 w1 w2 "}
    assert body["add_generation_prompt"] is False and body["continue_final_message"] is True
    assert [e.text for e in events if not e.final][0] == "w3 "
    assert events[-1].finish_reason == "stop"


def test_health_helpers() -> None:
    with FakeServer() as srv:
        b = _backend(srv)
        assert set(b.check_models()) == set(ANSWER_TOKENS)
        assert set(b.server_versions().values()) == {"fake-0.0"}


def test_baseline_and_staircase_over_http() -> None:
    with FakeServer() as srv:
        b = _backend(srv)
        base = run_moa_baseline(b, _moa(), "q", "q1")
        stair = run_staircase(b, _moa(), StaircaseConfig(redundancy=1), "q", "q1")
        bodies = list(srv.bodies)
    assert base.text == stair.text  # deterministic fake: same final answer
    assert stair.ttft_s < base.ttft_s
    assert len(stair.aggregator_calls) > 1
    assert stair.aggregator_prompt_tokens > base.aggregator_prompt_tokens
    seeds = sorted(body["seed"] for body in bodies if body["model"] in ("fast", "slow"))
    assert seeds == [7, 7, 8, 8]


def test_vllm_command_line() -> None:
    cfg = {"experiment": {"seed": 42}, "backend": {"host": "127.0.0.1"},
           "vllm": {"enable_prefix_caching": True, "optional_args": ["--x"], "extra_args": []}}
    srv = {"hf_model": "Org/M", "revision": "abc", "port": 8001, "gpu_memory_utilization": 0.3,
           "max_model_len": 4096, "dtype": "half"}
    cmd = build_command("m", srv, cfg)
    assert cmd[:3] == ["vllm", "serve", "Org/M"]
    assert cmd[cmd.index("--served-model-name") + 1] == "m"
    assert "--enable-prefix-caching" in cmd and "--x" in cmd
    assert "--x" not in build_command("m", srv, cfg, optional=False)
