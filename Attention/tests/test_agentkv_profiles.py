import pytest
import torch

from Attention.methods.agentkv.policy import AgentKVConfig, attention_scores, policy_indices, select_indices
from Dataloaders.synthetic.agent_memory import answer_diagnostics


def test_reference_profile_preserves_pilot_selection():
    torch.manual_seed(16)
    q, k = torch.randn(4, 8, 4), torch.randn(2, 20, 4)
    positions = torch.arange(20).expand(2, -1)
    cfg = AgentKVConfig(budget=8, sink_tokens=1, recent_tokens=1)
    torch.testing.assert_close(policy_indices(q, k, positions, cfg),
                               select_indices(attention_scores(q, k), positions, cfg))


def test_upstream_mean_matches_independent_candidate_reference():
    torch.manual_seed(7)
    q, k = torch.randn(4, 8, 4), torch.randn(2, 20, 4)
    positions = torch.arange(20).expand(2, -1)
    cfg = AgentKVConfig(budget=6, sink_tokens=1, recent_tokens=1, scoring_profile="upstream_mean")
    expected = []
    for head in range(2):
        per_query = []
        for h in range(2 * head, 2 * head + 2):
            for rep in q[h]:
                per_query.append(torch.softmax(k[head, 1:-1] @ rep / 2.0, dim=0))
        scores = torch.stack(per_query).mean(0)
        selected = torch.argsort(scores, descending=True, stable=True)[:4] + 1
        expected.append(torch.cat((torch.tensor([0, 19]), selected)).sort().values)
    torch.testing.assert_close(policy_indices(q, k, positions, cfg), torch.stack(expected))
    # Protected key contents cannot change candidate scores in this profile.
    k[:, [0, -1]] *= 10000
    torch.testing.assert_close(policy_indices(q, k, positions, cfg), torch.stack(expected))


def test_mean_gqa_is_not_max_gqa():
    q = torch.tensor([[[1., 0.]], [[0., 1.]]])
    k = torch.tensor([[[10., 0.], [0., 1.], [0., 0.]]])
    assert not torch.allclose(attention_scores(q, k), attention_scores(q, k, group_reduction="mean"))


@pytest.mark.parametrize("text,truncated,category", [
    ("</think>123456", False, "correct"),
    ("123456", False, "missing_final_delimiter"),
    ("reason", True, "truncated"),
    ("</think>654321", False, "wrong_final_answer"),
])
def test_diagnostics_do_not_rescore_reasoning(text, truncated, category):
    result = answer_diagnostics(text, "123456", truncated)
    assert result["answer_category"] == category
    assert result["correct"] == (category == "correct")
