import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from Attention.methods.agentkv import AgentKVConfig
from Attention.methods.agentkv.qwen3 import Qwen3Session


@pytest.fixture
def model():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize("mode", ["full", "recent", "agentkv"])
def test_no_eviction_matches_unmodified_chunked_and_full(model, mode):
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    with torch.inference_mode():
        expected = model(ids[None]).logits[0, -1]
    with Qwen3Session(model, mode) as session:
        session.append(ids[:4], ["<think>"] * 4)
        result = session.append(ids[4:], ["word"] * 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
        assert session.report()["cached_tokens"] == 7
    assert not model._agentkv_session_active


def test_repeated_eviction_logical_positions_and_turns(model):
    cfg = AgentKVConfig(budget=4, query_budget=4, interval=2, sink_tokens=1, recent_tokens=1)
    with Qwen3Session(model, config=cfg) as session:
        session.append(list(range(1, 9)), ["<think>"] * 8)
        session.compress()
        assert session.logical_length == 8
        assert session.cache.get_seq_length() == 4
        for token in (9, 10, 11, 12):
            logits = session.append([token], ["x"], decode=True)
            assert torch.isfinite(logits).all()
        session.append([13, 14, 15], ["<tool_response>", "value", "</tool_response>"])
        session.compress()
        assert session.report()["compression_events"] == 4
        assert session.report()["logical_tokens"] == 15
        for pos in session.positions.values():
            assert (pos[:, 0] == 0).all()
            assert (pos[:, -1] == 14).all()
            assert (pos[:, 1:] > pos[:, :-1]).all()
        assert session.report()["kv_bytes"] == 2 * 2 * 2 * 4 * 8 * 4
        session.reset()
        assert session.logical_length == session.cache.get_seq_length() == 0
        assert session.report()["query_buffer_bytes"] == 0


def test_compressed_chunk_mask_matches_token_by_token(model):
    cfg = AgentKVConfig(budget=4, query_budget=4, interval=128, sink_tokens=1, recent_tokens=1)
    def run(chunked):
        with Qwen3Session(model, config=cfg) as session:
            session.append(list(range(1, 9)), ["<think>"] * 8)
            session.compress()
            if chunked:
                return session.append([9, 10, 11], ["x"] * 3)
            for token in (9, 10, 11):
                result = session.append([token], ["x"])
            return result
    torch.testing.assert_close(run(True), run(False), atol=1e-6, rtol=1e-5)


def test_hooks_removed_on_failure_and_session_exclusive(model):
    with pytest.raises(RuntimeError, match="example"):
        with Qwen3Session(model) as session:
            with pytest.raises(ValueError, match="one session"):
                Qwen3Session(model)
            raise RuntimeError("example")
    assert not model._agentkv_session_active
    assert not model.model.layers[0].self_attn._forward_hooks
    with pytest.raises(RuntimeError, match="closed"):
        session.append([1], ["x"])
