import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.aaac import (
    assign_selection_groups,
    encode_indices,
    nearest_codeword,
)

from aaac_helpers import brute_force_group_errors


def _two_tables(first, second):
    """Pad two short tables to 16 entries by repeating their last value."""
    def pad(t):
        t = torch.tensor(t, dtype=torch.float32)
        return torch.cat([t, t[-1].repeat(16 - t.numel())])
    return torch.stack([pad(first), pad(second)])


def test_nearest_codeword_ties_go_to_lowest_index():
    table = torch.tensor([0.0, 1.0, 1.0, -1.0])
    idx, dist = nearest_codeword(torch.tensor([0.5, 1.0, -0.5, 0.9]), table)
    assert idx.tolist() == [0, 1, 0, 1]  # 0.5 and -0.5 tie between two entries -> lowest index
    assert torch.allclose(dist, torch.tensor([0.25, 0.0, 0.25, 0.01]))


def test_assignment_is_activation_weighted():
    # one row, one selection group of two weights: u = [0, 1]
    u = torch.tensor([[0.0, 1.0]])
    tables = _two_tables([0.0, -1.0], [1.0, 2.0])  # table 0 exact for u0, table 1 exact for u1
    # unweighted errors: table0 -> 0 + 1 = 1, table1 -> 1 + 0 = 1 (tie)
    sel, err = assign_selection_groups(u, torch.tensor([10.0, 1.0]), tables, 2)
    assert sel.tolist() == [[False]] and err.item() == 1.0
    sel, err = assign_selection_groups(u, torch.tensor([1.0, 10.0]), tables, 2)
    assert sel.tolist() == [[True]] and err.item() == 1.0


def test_equal_errors_choose_table_zero():
    u = torch.tensor([[0.0, 1.0, 0.3, -0.7]])
    tables = _two_tables([0.0, 0.5], [0.0, 0.5])
    sel, _ = assign_selection_groups(u, torch.ones(4), tables, 2)
    assert not sel.any()
    sel, _ = assign_selection_groups(u, torch.zeros(4), _two_tables([0.0], [1.0]), 2)
    assert not sel.any()  # zero importance -> zero error for both -> table 0


def test_group_errors_match_brute_force_and_choose_minimum():
    torch.manual_seed(0)
    u = torch.rand(37, 64) * 2 - 1
    imp = torch.rand(64) * 100
    tables = torch.sort(torch.rand(2, 16) * 2 - 1, dim=1).values
    sel, err = assign_selection_groups(u, imp, tables, 16)
    ref = brute_force_group_errors(u, imp, tables, 16)
    assert sel.shape == (37, 4) and sel.dtype == torch.bool
    assert torch.equal(sel, ref[1] < ref[0])
    assert torch.allclose(err.double(), ref.min(dim=0).values, rtol=1e-5)


def test_encode_uses_the_selected_table_per_group():
    torch.manual_seed(1)
    u = torch.rand(9, 32) * 2 - 1
    tables = torch.sort(torch.rand(2, 16) * 2 - 1, dim=1).values
    sel = torch.rand(9, 2) > 0.5
    idx, dist = encode_indices(u, tables, sel, 16)
    assert idx.dtype == torch.uint8 and idx.shape == u.shape and int(idx.max()) < 16
    table_per_weight = sel.long().repeat_interleave(16, dim=1)
    chosen = tables[table_per_weight, idx.long()]
    assert torch.allclose(dist, (u - chosen).square())
    # brute force: no entry of the selected table is closer
    for o in range(9):
        for k in range(32):
            t = tables[table_per_weight[o, k]]
            assert (u[o, k] - t).abs().min() == (u[o, k] - chosen[o, k]).abs()


def test_results_do_not_depend_on_chunk_size():
    torch.manual_seed(2)
    u = torch.rand(50, 128) * 2 - 1
    imp = torch.rand(128)
    tables = torch.sort(torch.rand(2, 16) * 2 - 1, dim=1).values
    ref_sel, ref_err = assign_selection_groups(u, imp, tables, 16, chunk_elements=1 << 30)
    ref_idx, _ = encode_indices(u, tables, ref_sel, 16, chunk_elements=1 << 30)
    for chunk in (1, 128 * 16, 128 * 16 * 7 + 5):
        sel, err = assign_selection_groups(u, imp, tables, 16, chunk_elements=chunk)
        idx, _ = encode_indices(u, tables, sel, 16, chunk_elements=chunk)
        assert torch.equal(sel, ref_sel) and torch.equal(err, ref_err) and torch.equal(idx, ref_idx)


def test_assignment_validation():
    u = torch.zeros(2, 8)
    tables = torch.zeros(2, 16)
    with pytest.raises(ValueError, match="divisible"):
        assign_selection_groups(u, torch.ones(8), tables, 3)
    with pytest.raises(ValueError, match="importance shape"):
        assign_selection_groups(u, torch.ones(4), tables, 4)
    with pytest.raises(ValueError, match="non-negative"):
        assign_selection_groups(u, -torch.ones(8), tables, 4)
    with pytest.raises(ValueError, match="codebooks shape"):
        assign_selection_groups(u, torch.ones(8), torch.zeros(2, 8), 4)
    with pytest.raises(ValueError, match="selectors"):
        encode_indices(u, tables, torch.zeros(2, 2, dtype=torch.uint8), 4)
