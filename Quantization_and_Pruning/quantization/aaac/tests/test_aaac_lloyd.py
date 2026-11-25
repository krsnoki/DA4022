import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.aaac import (
    AAACConfig,
    assign_selection_groups,
    encode_indices,
    fit_aaac_layer,
)
from Quantization_and_Pruning.quantization.aaac.lloyd import (
    finalize_codebooks,
    weighted_lloyd_update,
)

CHUNK = 1 << 24


def _tables(lo0=-0.9, hi0=0.9, lo1=-0.8, hi1=0.8):
    return torch.stack([torch.linspace(lo0, hi0, 16), torch.linspace(lo1, hi1, 16)])


def _update(u, imp, tables, selectors, s, chunk=CHUNK):
    return weighted_lloyd_update(u, imp, tables, selectors, s, chunk_elements=chunk)


def _reference(u, imp, tables, selectors, s):
    """Brute-force float64 weighted centroids, written independently of lloyd.py."""
    out, inp = u.shape
    new = tables.double().clone()
    num = torch.zeros(2, 16, dtype=torch.float64)
    den = torch.zeros(2, 16, dtype=torch.float64)
    for o in range(out):
        for k in range(inp):
            c = int(selectors[o, k // s])
            j = int((u[o, k].double() - tables[c].double()).square().argmin())
            num[c, j] += float(imp[k]) * float(u[o, k])
            den[c, j] += float(imp[k])
    mask = den > 0
    new[mask] = num[mask] / den[mask]
    return new.float()


def test_weighted_centroid_differs_from_plain_mean():
    # Entries sit at the integers -8..7, so u = -0.1 and 0.3 share the cell of entry 0.
    tables = torch.arange(-8, 8, dtype=torch.float32).repeat(2, 1)
    u = torch.tensor([[-0.1, 0.3]])
    imp = torch.tensor([1.0, 3.0])
    sel = torch.zeros(1, 1, dtype=torch.bool)
    idx, _ = encode_indices(u, tables, sel, 2)
    assert idx[0, 0] == idx[0, 1] == 8                 # same cell
    new = _update(u, imp, tables, sel, 2)
    assert new[0, 8].item() == pytest.approx((1 * -0.1 + 3 * 0.3) / 4)   # 0.2; plain mean is 0.1
    assert new[0, 8].item() != pytest.approx(0.1)
    assert torch.equal(new[1], tables[1])               # table 1 unused -> unchanged


def test_matches_brute_force_reference_both_selectors():
    g = torch.Generator().manual_seed(3)
    u = torch.randn(6, 64, generator=g) * 0.4
    imp = torch.rand(64, generator=g) * 10
    tables = _tables()
    sel = torch.rand(6, 4, generator=g) > 0.5
    new = _update(u, imp, tables, sel, 16)
    assert new.dtype == torch.float32 and new.shape == (2, 16)
    assert torch.allclose(new, _reference(u, imp, tables, sel, 16), atol=1e-6, rtol=0)
    assert sel.any() and (~sel).any()


def test_empty_cells_keep_old_value_and_input_is_not_modified():
    # All weights near 0: far-away table entries get no members.
    u = torch.full((2, 32), 0.01)
    imp = torch.ones(32)
    tables = _tables()
    original = tables.clone()
    sel = torch.zeros(2, 1, dtype=torch.bool)
    new = _update(u, imp, tables, sel, 32)
    assert torch.equal(tables, original)                     # not modified in place
    assert new.data_ptr() != tables.data_ptr()
    idx, _ = encode_indices(u, tables, sel, 32)
    used = {int(i) for i in idx.flatten()}
    empty = [j for j in range(16) if j not in used]
    assert empty
    for j in empty:
        assert new[0, j] == tables[0, j]                     # empty cell in table 0
    assert torch.equal(new[1], tables[1])                    # table 1 unused entirely
    for j in used:
        assert new[0, j].item() == pytest.approx(0.01, abs=1e-7)


def test_zero_importance_cell_keeps_old_value():
    tables = torch.arange(-8, 8, dtype=torch.float32).repeat(2, 1)
    u = torch.tensor([[-0.1, 0.3, -5.2, -4.9]])
    imp = torch.tensor([1.0, 3.0, 0.0, 0.0])               # the entry -5 cell has all I = 0
    sel = torch.zeros(1, 2, dtype=torch.bool)
    idx, _ = encode_indices(u, tables, sel, 2)
    assert idx[0, 2] == idx[0, 3] == 3                     # both in the cell of entry -5
    new = _update(u, imp, tables, sel, 2)
    assert new[0, 3] == tables[0, 3]                       # kept, not 0/0
    assert new[0, 8].item() == pytest.approx(0.2)
    assert torch.isfinite(new).all()


def test_update_is_deterministic_and_chunk_invariant():
    g = torch.Generator().manual_seed(11)
    u = torch.randn(40, 256, generator=g) * 0.3
    imp = torch.rand(256, generator=g) * 100
    tables = _tables()
    sel = torch.rand(40, 16, generator=g) > 0.5
    a = _update(u, imp, tables, sel, 16)
    assert torch.equal(a, _update(u.clone(), imp.clone(), tables.clone(), sel.clone(), 16))
    for chunk in (1, 5000):
        assert torch.equal(a, _update(u, imp, tables, sel, 16, chunk))


def test_update_rejects_bad_inputs():
    u = torch.zeros(2, 32)
    sel = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError):
        _update(u, torch.ones(31), _tables(), sel, 32)
    with pytest.raises(ValueError):
        _update(u, -torch.ones(32), _tables(), sel, 32)


def test_update_never_increases_weighted_error():
    g = torch.Generator().manual_seed(5)
    u = torch.randn(16, 128, generator=g) * 0.3
    imp = torch.rand(128, generator=g) * 50
    tables = _tables()
    sel, _ = assign_selection_groups(u, imp, tables, 16)

    def err(t):
        idx, d = encode_indices(u, t, sel, 16)
        return float((d * imp).sum())

    prev = err(tables)
    for _ in range(5):
        tables = _update(u, imp, tables, sel, 16)
        cur = err(tables)
        assert cur <= prev * (1 + 1e-6)
        prev = cur


def test_finalize_is_bf16_and_equals_cast_for_sorted_tables():
    t = _tables()
    out = finalize_codebooks(t)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 16)
    assert torch.equal(out, t.to(torch.bfloat16))
    assert out.data_ptr() != t.data_ptr()
    assert torch.equal(finalize_codebooks(t, torch.bfloat16), out)
    assert finalize_codebooks(t, torch.float16).dtype == torch.float16


def test_finalize_sorts_ascending_and_can_be_disabled():
    t = _tables().flip(-1).contiguous()
    assert torch.equal(finalize_codebooks(t), _tables().to(torch.bfloat16))
    assert torch.equal(finalize_codebooks(t, sort=False), t.to(torch.bfloat16))


def test_finalize_rejects_bad_tables():
    with pytest.raises(ValueError):
        finalize_codebooks(torch.zeros(3, 16))
    bad = _tables()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        finalize_codebooks(bad)


def test_fit_stores_indices_and_selectors_matching_the_bf16_tables():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(24, 256, generator=g) * 0.02
    w[3, 7] = 0.5
    imp = torch.rand(256, generator=g) * 1000
    for s in (128, 16):
        cfg = AAACConfig(selection_size=s)
        fit = fit_aaac_layer(w, imp, cfg)
        assert fit.codebooks.dtype == torch.bfloat16
        from Quantization_and_Pruning.quantization.aaac import compute_group_scales, normalize_weight
        scales = compute_group_scales(w, cfg.group_size, cfg.scale_dtype)
        u = normalize_weight(w, scales, cfg.group_size)
        sel, _ = assign_selection_groups(u, imp, fit.codebooks, s)
        idx, _ = encode_indices(u, fit.codebooks, fit.selectors, s)
        assert torch.equal(fit.selectors, sel)
        assert torch.equal(fit.indices, idx)
        # tables are stored ascending
        assert (fit.codebooks[:, 1:] >= fit.codebooks[:, :-1]).all()


def test_full_fit_is_bit_identical_across_runs():
    g = torch.Generator().manual_seed(9)
    w = torch.randn(16, 256, generator=g) * 0.02
    imp = torch.rand(256, generator=g) * 100
    a = fit_aaac_layer(w, imp, AAACConfig(selection_size=16))
    b = fit_aaac_layer(w.clone(), imp.clone(), AAACConfig(selection_size=16))
    for name in ("scales", "codebooks", "selectors", "indices"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.objective_history == b.objective_history
