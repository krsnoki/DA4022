import pytest
import torch

from Mixture_of_Experts.moe_to_dense.merging import ExpertWeights, merge_groups, round_robin_groups


def expert(value: float) -> ExpertWeights:
    gate = torch.full((2, 3), float(value))
    up = torch.full((2, 3), float(value + 1))
    down = torch.full((3, 2), float(value + 2))
    return ExpertWeights(gate, up, down)


def test_round_robin_assigns_every_expert_once():
    groups = round_robin_groups([4, 1, 3, 2], 2)
    assert groups == [[4, 3], [1, 2]]
    assert sorted(sum(groups, [])) == [1, 2, 3, 4]


def test_pure_pruning_does_not_average():
    experts = [expert(1), expert(5)]
    merged, scores = merge_groups(experts, [[1], [0]], torch.tensor([0.25, 0.75]))
    assert torch.equal(merged[0].gate, experts[1].gate)
    assert torch.equal(merged[1].down, experts[0].down)
    assert torch.allclose(scores, torch.tensor([0.75, 0.25]))


def test_weighted_merge_matches_manual_result():
    experts = [expert(1), expert(5)]
    merged, _ = merge_groups(experts, [[0, 1]], torch.tensor([1.0, 3.0]))
    assert torch.allclose(merged[0].gate, 0.25 * experts[0].gate + 0.75 * experts[1].gate)


def test_invalid_empty_group_rejected():
    with pytest.raises(ValueError):
        merge_groups([expert(1)], [[]], torch.ones(1))
