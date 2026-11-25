import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch
from chunked_ops import mm_softmax_mm, chunked_lm_head_ce, _split_bounds


@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 16])
def test_mm_softmax_mm_matches_baseline(chunks):
    torch.manual_seed(0)
    a, b = torch.randn(48, 8), torch.randn(8, 32)
    base = mm_softmax_mm(a, b, 1)
    got = mm_softmax_mm(a, b, chunks)
    assert got.shape == (32, 8) and got.dtype == a.dtype
    torch.testing.assert_close(got, base, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunks", [1, 2, 5, 12])
def test_lm_head_ce_loss_and_grads(chunks):
    torch.manual_seed(0)
    t, h, v = 24, 8, 40
    x0, w0 = torch.randn(t, h), torch.randn(v, h)
    y = torch.randint(0, v, (t,)); y[3] = -100
    x1, w1 = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    x2, w2 = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    ref = torch.nn.functional.cross_entropy(x1 @ w1.t(), y, ignore_index=-100)
    ref.backward()
    got = chunked_lm_head_ce(x2, w2, y, chunks)
    got.backward()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w2.grad, w1.grad, rtol=1e-4, atol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        _split_bounds(4, 0)
    with pytest.raises(ValueError):
        _split_bounds(4, 5)
    with pytest.raises(ValueError):
        mm_softmax_mm(torch.randn(4, 3), torch.randn(5, 2), 1)
    assert _split_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]