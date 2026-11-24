"""Simulated quantization primitive and GPTQ."""

import pytest
import torch

from Backbones.registry import load_backbone
from Quantization_and_Pruning.quantization.activation.fake_quant import (
    FakeQuantConfig, FakeQuantLinear, apply_fake_quant, fake_quantize, quant_params,
)
from Quantization_and_Pruning.quantization.qat.jacobian_noise import use_jacobian_noise_attention
from Quantization_and_Pruning.quantization.weight_only.gptq import gptq_quantize_model, gptq_quantize_weight


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [True, False])
def test_levels_and_error_bound(bits, sym):
    torch.manual_seed(bits)
    x = torch.randn(16, 64)
    q = fake_quantize(x, bits, symmetric=sym)
    for row in q:
        assert row.unique().numel() <= 2**bits
    scale, _ = quant_params(x, bits, sym)
    assert ((q - x).abs() <= scale / 2 + 1e-6).all()


def test_none_and_16_bit_are_identity():
    x = torch.randn(3, 5)
    assert fake_quantize(x, None) is x
    assert fake_quantize(x, 16) is x


def test_group_size():
    x = torch.randn(4, 32)
    x[:, :16] *= 100
    g = fake_quantize(x, 4, group_size=16)
    full = fake_quantize(x, 4)
    assert (g - x)[:, 16:].abs().mean() < (full - x)[:, 16:].abs().mean()
    with pytest.raises(ValueError):
        fake_quantize(torch.randn(2, 10), 4, group_size=3)


def test_dtype_preserved():
    x = torch.randn(4, 8, dtype=torch.float16)
    assert fake_quantize(x, 4).dtype == torch.float16


def test_tag_parsing():
    assert FakeQuantConfig.from_tag("W4A4").tag == "W4A4"
    c = FakeQuantConfig.from_tag("w4a8qk4")
    assert (c.w_bits, c.a_bits, c.qk_bits) == (4, 8, 4)
    assert FakeQuantConfig.from_tag("W4A16").a_bits is None
    assert FakeQuantConfig.from_tag("FP16").tag == "FP16"
    with pytest.raises(ValueError):
        FakeQuantConfig.from_tag("int4")


def tiny(seed=0):
    torch.manual_seed(seed)
    return load_backbone("test.tiny_qwen2.random")[0].eval()


def test_apply_fake_quant_targets_only_block_linears():
    m = tiny()
    rep = apply_fake_quant(m, FakeQuantConfig.from_tag("W4A4"))
    assert rep["num_linear_quantized"] == 2 * 7  # q,k,v,o,gate,up,down per layer
    assert not isinstance(m.lm_head, FakeQuantLinear)
    assert isinstance(m.model.layers[0].mlp.down_proj, FakeQuantLinear)


def test_quantization_error_ordering():
    ids = torch.randint(0, 256, (1, 24), generator=torch.Generator().manual_seed(0))
    ref = tiny()(input_ids=ids).logits
    errs = {}
    for tag in ("W8A8", "W4A4"):
        m = tiny()
        apply_fake_quant(m, FakeQuantConfig.from_tag(tag))
        errs[tag] = (m(input_ids=ids).logits - ref).abs().mean().item()
    assert errs["W8A8"] < errs["W4A4"]
    assert errs["W8A8"] < 0.05 * ref.abs().mean().item()


def test_qk_quant_requires_and_uses_custom_attention():
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(0))
    a = tiny()
    use_jacobian_noise_attention(a)
    rep = apply_fake_quant(a, FakeQuantConfig(w_bits=None, a_bits=None, qk_bits=2))
    assert rep["num_attention_qk_quantized"] == 2
    ref = tiny()(input_ids=ids).logits
    assert not torch.allclose(a(input_ids=ids).logits, ref)


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    X = torch.randn(1024, 32) @ torch.randn(32, 32)
    W = torch.randn(16, 32)
    H = 2 * X.t() @ X / X.shape[0]
    err = lambda Q: ((X @ (W - Q).t()) ** 2).mean()
    for gs in (None, 8):
        assert err(gptq_quantize_weight(W, H, 3, group_size=gs)) < err(fake_quantize(W, 3, group_size=gs))
    q = gptq_quantize_weight(W, H, 3)
    assert all(r.unique().numel() <= 8 for r in q)


def test_gptq_model_end_to_end():
    m = tiny()
    before = m.model.layers[1].mlp.down_proj.weight.clone()
    calib = [torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(i)) for i in range(4)]
    cfg = FakeQuantConfig.from_tag("W4A16")
    rep = gptq_quantize_model(m, calib, lambda mod, b: mod(input_ids=b), cfg, log=lambda s: None)
    assert rep["gptq_layers"] == 14
    after = m.model.layers[1].mlp.down_proj.weight
    assert not torch.equal(before, after)
    assert all(r.unique().numel() <= 16 for r in after)
    apply_fake_quant(m, cfg, weights_already_quantized=True)
    assert torch.equal(m.model.layers[1].mlp.down_proj.weight, after)
