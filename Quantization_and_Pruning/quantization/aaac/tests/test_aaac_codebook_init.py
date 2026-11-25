import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.aaac import (
    initial_codebooks,
    quantile_codebook,
    weighted_quantiles,
)
from Quantization_and_Pruning.quantization.aaac.codebook_init import selection_group_spread


def test_weighted_quantiles_inverse_cdf_by_hand():
    values = torch.tensor([3.0, 1.0, 2.0, 4.0])
    # sorted: 1, 2, 3, 4 with weights 1, 1, 1, 1 -> cdf 1, 2, 3, 4 (total 4)
    q = weighted_quantiles(values, torch.ones(4), torch.tensor([0.1, 0.25, 0.26, 0.5, 0.75, 1.0]))
    assert torch.equal(q, torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 4.0]))


def test_weights_move_quantiles_and_zero_weight_values_are_never_picked():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    probs = torch.tensor([0.01, 0.5, 0.99])
    assert torch.equal(weighted_quantiles(values, torch.tensor([0.0, 0.0, 0.0, 5.0]), probs), torch.full((3,), 4.0))
    # heavy weight on 2.0 pulls the median there
    q = weighted_quantiles(values, torch.tensor([1.0, 10.0, 1.0, 1.0]), probs)
    assert torch.equal(q, torch.tensor([1.0, 2.0, 4.0]))


def test_all_zero_weights_fall_back_to_uniform():
    values = torch.tensor([4.0, 3.0, 2.0, 1.0])
    probs = torch.tensor([0.125, 0.375, 0.625, 0.875])
    assert torch.equal(
        weighted_quantiles(values, torch.zeros(4), probs),
        weighted_quantiles(values, torch.ones(4), probs),
    )


def test_weighted_quantiles_validation():
    with pytest.raises(ValueError, match="empty"):
        weighted_quantiles(torch.tensor([]), torch.tensor([]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        weighted_quantiles(torch.ones(2), torch.tensor([1.0, -1.0]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="probs"):
        weighted_quantiles(torch.ones(2), torch.ones(2), torch.tensor([1.5]))
    with pytest.raises(ValueError, match="shape"):
        weighted_quantiles(torch.ones(2), torch.ones(3), torch.tensor([0.5]))


def test_quantile_codebook_has_16_sorted_entries_from_the_data():
    torch.manual_seed(0)
    values = torch.randn(10_000)
    weights = torch.rand(10_000)
    table = quantile_codebook(values, weights)
    assert table.shape == (16,) and table.dtype == torch.float32
    assert torch.all(table[1:] >= table[:-1])
    assert all(bool((values == v).any()) for v in table)
    # mid-point quantiles of a symmetric sample are roughly symmetric
    assert abs(table[0].item() + table[-1].item()) < 0.1


def test_selection_group_spread_is_weighted_mean_abs():
    u = torch.tensor([[1.0, 0.0, -0.5, 0.5]])
    imp = torch.tensor([1.0, 3.0, 0.0, 0.0])  # second group has zero importance -> unweighted mean
    spread = selection_group_spread(u, imp, selection_size=2)
    assert torch.allclose(spread, torch.tensor([[0.25, 0.5]]))


def test_initial_codebooks_split_peaky_and_flat_groups():
    torch.manual_seed(0)
    s = 16
    peaky = torch.randn(8, 4 * s) * 0.05          # mass near 0
    flat = torch.rand(8, 4 * s) * 2 - 1             # spread over [-1, 1]
    u = torch.cat([peaky, flat], dim=0)
    tables = initial_codebooks(u, torch.ones(4 * s), s)
    assert tables.shape == (2, 16) and tables.dtype == torch.float32
    assert torch.all(tables[:, 1:] >= tables[:, :-1])
    # table 0 is built from the peaky half, table 1 from the flat half
    assert tables[0].abs().max() < 0.25 < tables[1].abs().max()


def test_initial_codebooks_ignore_zero_importance_columns():
    u = torch.zeros(4, 8)
    u[:, :4] = torch.linspace(-0.4, 0.4, 4)
    u[:, 4:] = 1.0  # huge values in unimportant columns
    imp = torch.tensor([1.0] * 4 + [0.0] * 4)
    tables = initial_codebooks(u, imp, selection_size=4)
    assert tables.abs().max().item() <= 0.4 + 1e-6


def test_single_selection_group_gives_identical_tables():
    u = torch.linspace(-1, 1, 16).reshape(1, 16)
    tables = initial_codebooks(u, torch.ones(16), selection_size=16)
    assert torch.equal(tables[0], tables[1])


def test_initial_codebooks_are_deterministic_and_input_is_untouched():
    torch.manual_seed(3)
    u = torch.rand(32, 64) * 2 - 1
    imp = torch.rand(64)
    before = u.clone()
    a = initial_codebooks(u, imp, 16)
    b = initial_codebooks(u.clone(), imp.clone(), 16)
    assert torch.equal(a, b) and torch.equal(u, before)
