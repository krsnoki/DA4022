"""The closed-form ||J_softmax||_F^2 must match autograd and the explicit matrix."""

import pytest
import torch

from Quantization_and_Pruning.quantization.qat.jacobian_noise import (
    softmax_jacobian_explicit, softmax_jacobian_fro, softmax_jacobian_fro_sq,
)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 64])
def test_closed_form_matches_autograd(n):
    torch.manual_seed(n)
    for scale in (0.1, 1.0, 10.0):
        z = (torch.randn(n, dtype=torch.float64) * scale)
        J = torch.autograd.functional.jacobian(lambda x: torch.softmax(x, -1), z)
        p = torch.softmax(z, -1)
        expected = (J ** 2).sum()
        assert torch.allclose(softmax_jacobian_fro_sq(p).double(), expected, atol=1e-6)
        assert torch.allclose(softmax_jacobian_explicit(p).double(), J, atol=1e-6)


def test_batched_shapes_and_values():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(2, 3, 5, 11), -1)
    out = softmax_jacobian_fro_sq(p)
    assert out.shape == (2, 3, 5) and out.dtype == torch.float32
    ref = torch.stack([(softmax_jacobian_explicit(r) ** 2).sum() for r in p.reshape(-1, 11)]).reshape(2, 3, 5)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(softmax_jacobian_fro(p), ref.sqrt(), atol=1e-6)


def test_known_values_and_bounds():
    one_hot = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert softmax_jacobian_fro_sq(one_hot).item() == pytest.approx(0.0, abs=1e-7)
    assert softmax_jacobian_fro_sq(torch.tensor([0.5, 0.5])).item() == pytest.approx(0.25)
    for n in (2, 3, 10, 100):
        assert softmax_jacobian_fro_sq(torch.full((n,), 1.0 / n)).item() == pytest.approx(1 / n - 1 / n**2, rel=1e-5)
    rnd = torch.softmax(torch.randn(1000, 17) * 3, -1)
    v = softmax_jacobian_fro_sq(rnd)
    assert (v >= 0).all() and (v <= 0.25 + 1e-6).all()


def test_masked_entries_are_ignored():
    z = torch.tensor([0.3, -1.2, 2.0, float("-inf"), float("-inf")])
    p = torch.softmax(z, -1)
    assert torch.allclose(softmax_jacobian_fro_sq(p), softmax_jacobian_fro_sq(torch.softmax(z[:3], -1)))


def test_half_precision_input():
    p = torch.softmax(torch.randn(4, 9), -1)
    assert torch.allclose(softmax_jacobian_fro_sq(p.half()), softmax_jacobian_fro_sq(p), atol=1e-3)
