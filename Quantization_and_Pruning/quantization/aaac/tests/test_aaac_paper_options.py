"""Tests for the paper-matching options: Eq. 9-10 initialisation and asymmetric RTN."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from Quantization_and_Pruning.quantization.aaac import (
    AAACConfig,
    RTNLinear,
    compute_group_scales,
    fit_aaac_layer,
    measure_model_storage_bytes,
    normalize_weight,
    paper_initial_codebooks,
    quantize_model_rtn,
    quantize_weight_rtn_int4_asym,
    reconstruct_rtn_weight_asym,
)


def _heavy_tailed(out=64, inp=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.distributions.StudentT(3.0).sample((out, inp)) * 0.02
    imp = torch.rand(inp, generator=g) ** 4 * 10 + 1e-3
    return w, imp


# ---------- Eq. 9-10 initialisation ----------

def test_paper_init_matches_equations_on_known_values():
    u = torch.linspace(-1, 1, 1001).reshape(7, 143)
    t = paper_initial_codebooks(u)
    assert t.shape == (2, 16) and t.dtype == torch.float32
    # T0 spans min..max at i/15
    assert torch.allclose(t[0], torch.linspace(-1, 1, 16), atol=1e-6)
    # T1 is shifted by delta = 1/30 and ends at the max
    delta = 1 / 30
    expected = -1 + 2 * (delta + (1 - delta) * torch.arange(16) / 15)
    assert torch.allclose(t[1], expected.float(), atol=1e-6)
    assert t[1, -1] == 1.0
    assert torch.all(t[:, 1:] >= t[:, :-1])


def test_paper_init_includes_group_extremes():
    w, _ = _heavy_tailed()
    scales = compute_group_scales(w, 128, torch.float16)
    u = normalize_weight(w, scales, 128)
    t = paper_initial_codebooks(u)
    assert t[0, 0] == u.min() and t[0, -1] == u.max()


def test_config_rejects_unknown_init():
    with pytest.raises(ValueError):
        AAACConfig(init_method="random")


@pytest.mark.parametrize("s", [128, 16])
def test_fit_with_paper_init_runs_and_is_deterministic(s):
    w, imp = _heavy_tailed()
    cfg = AAACConfig(group_size=128, selection_size=s, init_method="paper")
    a, b = fit_aaac_layer(w, imp, cfg), fit_aaac_layer(w, imp, cfg)
    assert torch.equal(a.indices, b.indices) and torch.equal(a.codebooks, b.codebooks)
    assert a.codebooks.dtype == torch.bfloat16
    hist = a.objective_history
    assert all(x >= y - 1e-6 * abs(x) for x, y in zip(hist[:-2], hist[1:-1]))


# ---------- asymmetric RTN ----------

def test_asym_rtn_error_is_at_most_half_a_step():
    w, _ = _heavy_tailed()
    packed, scales, zeros = quantize_weight_rtn_int4_asym(w, 128, torch.float16)
    assert packed.shape == (64, 128) and packed.dtype == torch.uint8
    assert scales.shape == (64, 2) and (scales > 0).all()
    assert zeros.shape == (64, 2) and zeros.dtype == torch.uint8 and zeros.max() <= 15
    w_hat = reconstruct_rtn_weight_asym(packed, scales, zeros, 256, 128)
    step = scales.float().repeat_interleave(128, dim=1)
    assert ((w_hat - w).abs() <= step / 2 + 1e-6).all()


def test_asym_rtn_uses_all_16_levels_and_beats_symmetric():
    w, _ = _heavy_tailed()
    lin = nn.Linear(256, 64, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w)
    asym = RTNLinear.from_linear(lin, 128, asymmetric=True)
    sym = RTNLinear.from_linear(lin, 128)
    err_a = (asym.dequantize_weight(torch.float32) - w).square().sum()
    err_s = (sym.dequantize_weight(torch.float32) - w).square().sum()
    assert err_a < err_s
    from Quantization_and_Pruning.quantization.aaac import unpack_int4_indices
    assert unpack_int4_indices(asym.packed_indices, 256).unique().numel() == 16
    assert torch.equal(asym.bias, lin.bias)


def test_asym_rtn_zero_points_are_counted_in_storage():
    model = nn.Sequential()
    model.add_module("layers", nn.ModuleList([nn.Linear(256, 64)]))
    quantize_model_rtn(model, ["layers.0"], group_size=128, asymmetric=True)
    st = measure_model_storage_bytes(model)
    assert st["packed_indices_bytes"] == 64 * 256 // 2
    assert st["scales_bytes"] == 64 * 2 * 2
    assert st["zero_point_bytes"] == 64 * 2
    assert st["total_quantized_bytes"] == 64 * 128 + 256 + 128


def test_all_zero_group_asym():
    w = torch.zeros(4, 128)
    packed, scales, zeros = quantize_weight_rtn_int4_asym(w, 128)
    assert (scales > 0).all()
    assert torch.equal(reconstruct_rtn_weight_asym(packed, scales, zeros, 128, 128), w)
