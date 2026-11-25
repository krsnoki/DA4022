import math

import pytest
import torch

from Attention.methods.agentkv import AgentKVConfig, PhaseParser, QueryBuffer, attention_scores, select_indices
from Attention.methods.agentkv.policy import gather_cache


def test_parser_split_markers_and_multiple_transitions():
    parser = PhaseParser()
    assert parser.feed("<thi") == "others"
    assert parser.feed("nk>reasoning") == "think"
    assert parser.feed("</think><tool_call>{}") == "act"
    assert parser.feed("</tool_call><tool_response>42") == "tool"
    assert parser.feed("</tool_response>") == "others"
    assert parser.feed("hello") == "others"  # old markers must not fire again
    parser.reset()
    assert parser.feed("nk>") == "others"


def test_parser_every_marker_split():
    from Attention.methods.agentkv.phases import MARKERS
    for marker, phase in MARKERS.items():
        for cut in range(1, len(marker)):
            parser = PhaseParser()
            parser.feed(marker[:cut])
            assert parser.feed(marker[cut:]) == phase


@pytest.mark.parametrize("kwargs", [
    {"budget": 0}, {"query_budget": 7}, {"interval": 0},
    {"sink_tokens": -1}, {"recent_tokens": 600}, {"pool_kernel": 2}, {"budget": True},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AgentKVConfig(**kwargs)


def test_phase_fifo_and_independent_storage():
    q = torch.arange(12.).reshape(1, 6, 2)
    buffer = QueryBuffer(4)
    buffer.update(q, ["tool", "think", "act", "others", "think", "think"])
    expected = q[:, [5, 2, 0, 3]].clone()
    q.zero_()
    torch.testing.assert_close(buffer.representatives(), expected)
    assert buffer.nbytes == 4 * 2 * 4


def test_recent_fifo_and_empty_buffer():
    buffer = QueryBuffer(4, "recent")
    with pytest.raises(ValueError):
        buffer.representatives()
    q = torch.randn(2, 9, 4)
    buffer.update(q, ["think"] * 9)
    torch.testing.assert_close(buffer.representatives(), q[:, -4:])


def test_scoring_matches_independent_loop_reference():
    torch.manual_seed(12)
    q, k = torch.randn(4, 3, 8), torch.randn(2, 11, 8)
    expected = []
    for h in range(2):
        group = []
        for g in range(2):
            reps = []
            for r in range(3):
                logits = torch.tensor([torch.dot(q[2*h+g, r], key) / math.sqrt(8) for key in k[h]])
                reps.append(torch.softmax(logits, 0))
            group.append(torch.stack(reps).mean(0))
        expected.append(torch.stack(group).max(0).values)
    torch.testing.assert_close(attention_scores(q, k), torch.stack(expected))


def test_phase_representatives_retain_old_tool_key():
    q = torch.tensor([[[10., 0.], [0., 10.], [0., 10.], [0., 10.], [0., 10.]]])
    k = torch.tensor([[[1., 0.], [0., 1.], [0., 1.]]])
    phased, recent = QueryBuffer(4), QueryBuffer(4, "recent")
    for b in (phased, recent):
        b.update(q, ["tool", "think", "think", "think", "think"])
    pos = torch.arange(3)[None]
    cfg = AgentKVConfig(budget=1, query_budget=4, sink_tokens=0, recent_tokens=0)
    assert select_indices(attention_scores(phased.representatives(), k), pos, cfg).item() == 0
    assert select_indices(attention_scores(recent.representatives(), k), pos, cfg).item() == 1


def test_protection_ties_and_physical_compaction():
    cfg = AgentKVConfig(budget=4, sink_tokens=1, recent_tokens=1)
    positions = torch.tensor([[0, 5, 8, 12, 13, 20], [0, 2, 3, 9, 12, 20]])
    scores = torch.tensor([[0., 8., 8., 8., 0., 0.], [0., 0., 0., 9., 8., 0.]])
    keep = select_indices(scores, positions, cfg)
    assert keep.tolist() == [[0, 1, 2, 5], [0, 3, 4, 5]]
    cache = torch.arange(24.).reshape(1, 2, 6, 2)
    compact = gather_cache(cache, keep)
    assert compact.shape == (1, 2, 4, 2)
    assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()
    torch.testing.assert_close(compact[0, 1], cache[0, 1, [0, 3, 4, 5]])


def test_short_cache_unchanged_and_invalid_scores_rejected():
    positions = torch.arange(3)[None]
    assert select_indices(torch.zeros(1, 3), positions, AgentKVConfig()).tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError):
        select_indices(torch.full((1, 3), float("nan")), positions, AgentKVConfig())
