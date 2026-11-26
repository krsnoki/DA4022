import torch

from Mixture_of_Experts.moe_to_dense.calibration import CalibrationStats
from Mixture_of_Experts.moe_to_dense.scoring import ExpertScorer, greedy_logdet_select, importance_weighted_kernel


def controlled_stats() -> CalibrationStats:
    return CalibrationStats(
        token_count=10,
        selection_count=torch.tensor([8.0, 4.0, 2.0]),
        selected_probability_sum=torch.tensor([4.0, 3.0, 0.4]),
        selected_activation_norm_sum=torch.tensor([8.0, 20.0, 2.0]),
        output_gram=torch.tensor([[4.0, 3.8, 0.0], [3.8, 4.0, 0.0], [0.0, 0.0, 4.0]]),
    )


def test_scores_are_finite_normalized_and_expected_order():
    scorer = ExpertScorer(controlled_stats())
    assert torch.isclose(scorer.sf().sum(), torch.tensor(1.0, dtype=torch.float64))
    assert scorer.select("sf", 2) == [0, 1]
    assert scorer.select("cp", 1) == [1]
    assert scorer.select("acp", 1) == [1]
    assert all(torch.isfinite(scorer.scores(method)).all() for method in ("sf", "cp", "acp"))


def test_doptimal_prefers_diverse_expert():
    scorer = ExpertScorer(controlled_stats())
    selected = scorer.select("do_cp", 2)
    assert 2 in selected
    assert len(set(selected)) == 2


def test_greedy_logdet_tie_breaks_by_index():
    selected = greedy_logdet_select(torch.eye(4, dtype=torch.float64), 3)
    assert selected == [0, 1, 2]


def test_weighted_kernel_is_symmetric_psd():
    stats = controlled_stats()
    kernel = importance_weighted_kernel(ExpertScorer(stats).cp(), stats.output_gram)
    assert torch.allclose(kernel, kernel.T)
    assert torch.linalg.eigvalsh(kernel).min() > -1e-8
