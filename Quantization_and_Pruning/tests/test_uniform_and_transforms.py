"""Shape/dtype and numerical tests for the shared quantizer and structured transforms (CPU)."""
import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.common import (  # noqa: E402
    QuantSpec,
    block_diag_dense,
    block_diag_matmul,
    dequantize_codes,
    hadamard_matrix,
    pick_block_size,
    quantize_dequantize,
    quantize_to_codes,
    random_block_diagonal,
    random_hadamard,
    random_orthogonal,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group", [None, 8])
def test_quantizer_levels_and_error_bound(bits, group):
    w = torch.randn(6, 32, generator=_gen())
    spec = QuantSpec(bits, group)
    wq = quantize_dequantize(w, spec)
    assert wq.shape == w.shape and wq.dtype == w.dtype
    gs = 32 if group is None else group
    x, xq = w.reshape(6, -1, gs), wq.reshape(6, -1, gs)
    scale = (x.amax(-1) - x.amin(-1)) / spec.levels
    assert ((x - xq).abs().amax(-1) <= scale / 2 + 1e-5).all()  # Eq. (2)
    for row in xq.reshape(-1, gs):
        assert row.unique().numel() <= 2 ** bits
    assert torch.allclose(xq.amin(-1), x.amin(-1), atol=1e-5)
    assert torch.allclose(xq.amax(-1), x.amax(-1), atol=1e-5)


def test_quantizer_codes_roundtrip_and_constant_groups():
    w = torch.randn(4, 16, generator=_gen(1))
    w[0] = 0.25  # constant row must be reproduced exactly
    spec = QuantSpec(3, 8)
    codes, scale, lo = quantize_to_codes(w, spec)
    assert codes.dtype == torch.int32 and int(codes.max()) <= spec.levels and int(codes.min()) >= 0
    assert torch.allclose(dequantize_codes(codes, scale, lo), quantize_dequantize(w, spec), atol=1e-6)
    assert torch.equal(quantize_dequantize(w, spec)[0], w[0])


def test_quantizer_dtype_and_validation():
    w = torch.randn(4, 16, generator=_gen(2)).to(torch.bfloat16)
    assert quantize_dequantize(w, QuantSpec(4)).dtype == torch.bfloat16
    with pytest.raises(ValueError):
        quantize_dequantize(torch.randn(4, 10), QuantSpec(4, 8))  # group does not divide
    with pytest.raises(ValueError):
        QuantSpec(0)
    with pytest.raises(ValueError):
        QuantSpec(4, -1)


def test_stochastic_rounding_is_unbiased():
    w = torch.randn(2, 16, generator=_gen(3))
    spec = QuantSpec(2)
    g = _gen(4)
    mean = torch.stack([quantize_dequantize(w, spec, stochastic=True, generator=g) for _ in range(3000)]).mean(0)
    scale = (w.amax(1, keepdim=True) - w.amin(1, keepdim=True)) / spec.levels
    assert ((mean - w).abs() <= 0.1 * scale).all()


def test_hadamard_and_random_blocks_are_orthogonal():
    h = hadamard_matrix(16)
    assert torch.allclose(h @ h.t(), 16 * torch.eye(16))
    for m in (random_hadamard(32, _gen()), random_orthogonal(24, _gen())):
        assert torch.allclose(m @ m.t(), torch.eye(m.shape[0]), atol=1e-5)
    with pytest.raises(ValueError):
        hadamard_matrix(12)


def test_block_diag_matmul_matches_dense():
    blocks = random_block_diagonal(32, 8, "rotation", _gen()) + 0.1 * torch.randn(4, 8, 8, generator=_gen(1))
    x = torch.randn(3, 5, 32, generator=_gen(2))
    assert torch.allclose(block_diag_matmul(x, blocks), x @ block_diag_dense(blocks), atol=1e-5)
    with pytest.raises(ValueError):
        block_diag_matmul(torch.randn(2, 30), blocks)


@pytest.mark.parametrize(
    "d,req,pow2,expected",
    [(896, 1024, True, 128), (9216, 1024, True, 1024), (2304, 1024, True, 256), (4864, 128, False, 128),
     (2304, 128, False, 128), (7, 4, False, 1)],
)
def test_pick_block_size(d, req, pow2, expected):
    assert pick_block_size(d, req, pow2) == expected
