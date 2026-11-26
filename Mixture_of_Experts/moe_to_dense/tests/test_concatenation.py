import pytest
import torch
from torch.nn import functional as F

from Mixture_of_Experts.moe_to_dense.concatenation import concatenate_experts
from Mixture_of_Experts.moe_to_dense.merging import ExpertWeights


def make_expert(seed: int) -> ExpertWeights:
    generator = torch.Generator().manual_seed(seed)
    return ExpertWeights(
        torch.randn(4, 3, generator=generator),
        torch.randn(4, 3, generator=generator),
        torch.randn(3, 4, generator=generator),
    )


def expert_forward(expert: ExpertWeights, x: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ expert.gate.T) * (x @ expert.up.T)) @ expert.down.T


@pytest.mark.parametrize("scaling,scores,alpha", [("uniform", torch.tensor([2.0, 1.0]), [0.5, 0.5]), ("proportional", torch.tensor([3.0, 1.0]), [0.75, 0.25])])
def test_dense_output_equals_scaled_expert_sum(scaling, scores, alpha):
    experts = [make_expert(1), make_expert(2)]
    dense = concatenate_experts(experts, scores, scaling)
    x = torch.randn(7, 3)
    expected = sum(weight * expert_forward(expert, x) for weight, expert in zip(alpha, experts))
    assert dense.gate_proj.out_features == 8
    assert torch.allclose(dense(x), expected, atol=1e-6, rtol=1e-5)
