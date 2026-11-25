import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.aaac import (
    compute_group_scales,
    expand_group_scales,
    normalize_weight,
)


def test_scale_is_group_absmax_and_normalised_values_reconstruct():
    w = torch.tensor([[1.0, -4.0, 0.5, 2.0], [-0.25, 0.125, 8.0, -8.0]])
    scales = compute_group_scales(w, group_size=2, scale_dtype=torch.float32)
    assert scales.shape == (2, 2) and scales.dtype == torch.float32
    assert torch.equal(scales, torch.tensor([[4.0, 2.0], [0.25, 8.0]]))

    u = normalize_weight(w, scales, group_size=2)
    assert u.dtype == torch.float32 and u.shape == w.shape
    assert torch.equal(u, torch.tensor([[0.25, -1.0, 0.25, 1.0], [-1.0, 0.5, 1.0, -1.0]]))
    assert torch.equal(u * expand_group_scales(scales, 2), w)


def test_groups_are_contiguous_within_each_row():
    w = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    scales = compute_group_scales(w, group_size=4, scale_dtype=torch.float32)
    assert torch.equal(scales, torch.tensor([[4.0, 8.0], [12.0, 16.0]]))


def test_all_zero_group_gets_tiny_positive_scale_and_zero_values():
    w = torch.zeros(2, 4)
    w[1, 2] = 3.0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        scales = compute_group_scales(w, group_size=2, scale_dtype=dtype)
        tiny = torch.finfo(dtype).tiny
        assert (scales > 0).all()
        assert scales[0, 0].item() == tiny and scales[0, 1].item() == tiny and scales[1, 0].item() == tiny
        assert scales[1, 1].item() == 3.0
        u = normalize_weight(w, scales, 2)
        assert torch.equal(u[0], torch.zeros(4)) and u[1, 2].item() == 1.0


def test_rounded_scales_stay_positive_and_finite_at_dtype_limits():
    w = torch.tensor([[1e-30, -1e-30, 1e6, 0.0]])
    scales = compute_group_scales(w, group_size=2, scale_dtype=torch.float16)
    assert scales.dtype == torch.float16
    assert (scales > 0).all() and torch.isfinite(scales).all()
    assert scales[0, 1].item() == torch.finfo(torch.float16).max  # 1e6 overflows FP16 -> clamped


def test_fp16_scale_rounding_keeps_normalised_values_near_unit_range():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    scales = compute_group_scales(w, 128, torch.float16)
    u = normalize_weight(w, scales, 128)
    assert u.abs().max().item() <= 1 + 2.0 ** -10
    assert torch.allclose(u * expand_group_scales(scales, 128), w, rtol=1e-6, atol=0)


def test_input_validation():
    with pytest.raises(ValueError, match="divisible"):
        compute_group_scales(torch.ones(2, 6), group_size=4)
    with pytest.raises(ValueError, match="2-D"):
        compute_group_scales(torch.ones(8), group_size=4)
    with pytest.raises(ValueError, match="non-finite"):
        compute_group_scales(torch.tensor([[1.0, float("nan")]]), group_size=2)
    with pytest.raises(ValueError, match="floating"):
        compute_group_scales(torch.ones(2, 4, dtype=torch.int32), group_size=2)
    with pytest.raises(ValueError, match="positive"):
        normalize_weight(torch.ones(1, 2), torch.zeros(1, 1), group_size=2)
    with pytest.raises(ValueError, match="shape"):
        normalize_weight(torch.ones(1, 4), torch.ones(1, 1), group_size=2)


def test_input_weight_is_not_modified():
    w = torch.randn(4, 8)
    before = w.clone()
    normalize_weight(w, compute_group_scales(w, 4), 4)
    assert torch.equal(w, before)
