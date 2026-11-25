from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from Quantization_and_Pruning.quantization.aaac.codec import (
    decode_scales_and_selectors,
    pack_aaac_fit,
    pack_aaac_layer,
    pack_int4_indices,
    unpack_int4_indices,
)
from Quantization_and_Pruning.quantization.aaac.config import AAACConfig
from Quantization_and_Pruning.quantization.aaac.fit import fit_aaac_layer
from Quantization_and_Pruning.quantization.aaac.linear import (
    AAACLinear,
    measure_model_storage_bytes,
    replace_linear_layers_with_aaac,
)
from Quantization_and_Pruning.quantization.aaac.scaling import compute_group_scales


def test_int4_index_round_trip_exact():
    g = torch.Generator().manual_seed(42)
    idx = torch.randint(0, 16, (37, 256), generator=g, dtype=torch.uint8)
    packed = pack_int4_indices(idx)
    assert packed.dtype == torch.uint8
    assert packed.shape == (37, 128)
    unpacked = unpack_int4_indices(packed, in_features=256)
    assert torch.equal(idx, unpacked)


@pytest.mark.parametrize("selection_size", [128, 16])
def test_selector_round_trip_and_zero_group_convention(selection_size: int):
    torch.manual_seed(0)
    out_f, in_f, g = 8, 256, 128
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    w[0, :g] = 0.0
    w[3, g : 2 * g] = 0.0

    scales = compute_group_scales(w, group_size=g, scale_dtype=torch.float16)
    assert (scales > 0).all()
    assert scales[0, 0].item() == pytest.approx(torch.finfo(torch.float16).tiny)

    selectors = torch.randint(0, 2, (out_f, in_f // selection_size), dtype=torch.bool)
    selectors[0, 0] = True  # force table 1 on an all-zero group
    indices = torch.randint(0, 16, (out_f, in_f), dtype=torch.uint8)
    codebooks = torch.linspace(-1.0, 1.0, 32, dtype=torch.bfloat16).reshape(2, 16)

    packed = pack_aaac_layer(
        indices=indices,
        codebooks=codebooks,
        scales=scales,
        selectors=selectors,
        group_size=g,
        selection_size=selection_size,
    )
    abs_scales, decoded_selectors = decode_scales_and_selectors(
        scales=packed.scales,
        selector_bitset=packed.selector_bitset,
        out_features=out_f,
        in_features=in_f,
        group_size=g,
        selection_size=selection_size,
    )
    assert torch.equal(decoded_selectors, selectors)
    assert torch.equal(abs_scales, scales.abs())


@pytest.mark.parametrize("selection_size", [128, 16])
def test_exact_packed_byte_counts(selection_size: int):
    out_f, in_f, g = 64, 256, 128
    indices = torch.randint(0, 16, (out_f, in_f), dtype=torch.uint8)
    codebooks = torch.zeros(2, 16, dtype=torch.bfloat16)
    scales = torch.ones(out_f, in_f // g, dtype=torch.float16)
    selectors = torch.randint(0, 2, (out_f, in_f // selection_size), dtype=torch.bool)

    packed = pack_aaac_layer(
        indices=indices,
        codebooks=codebooks,
        scales=scales,
        selectors=selectors,
        group_size=g,
        selection_size=selection_size,
    )
    breakdown = packed.packed_bytes_breakdown()

    expected_idx_bytes = (out_f * in_f) // 2
    expected_cb_bytes = 64
    expected_scale_bytes = out_f * (in_f // g) * 2
    expected_sel_bytes = (
        0 if selection_size == g else math.ceil((out_f * (in_f // selection_size)) / 8)
    )

    assert breakdown["packed_indices_bytes"] == expected_idx_bytes
    assert breakdown["codebooks_bytes"] == expected_cb_bytes
    assert breakdown["scales_bytes"] == expected_scale_bytes
    assert breakdown["selector_bitset_bytes"] == expected_sel_bytes
    assert breakdown["total_packed_bytes"] == (
        expected_idx_bytes + expected_cb_bytes + expected_scale_bytes + expected_sel_bytes
    )


@pytest.mark.parametrize("selection_size", [128, 16])
def test_bit_exact_parity_with_member2_fit_and_model_integration(selection_size: int):
    torch.manual_seed(7)

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, 128)
            self.proj = nn.Linear(128, 64, bias=True)
            self.norm = nn.LayerNorm(64)
            self.lm_head = nn.Linear(64, 32, bias=False)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            h = self.embed(tokens)
            h = self.norm(self.proj(h))
            return self.lm_head(h)

    model = ToyBlock().eval()
    model.proj.weight.data[0, :128] = 0.0  # all-zero group inside real fit
    orig_embed = model.embed.weight.detach().clone()
    orig_norm_w = model.norm.weight.detach().clone()
    orig_lm_head = model.lm_head.weight.detach().clone()
    orig_proj_bias = model.proj.bias.detach().clone()

    importance = torch.rand(128, dtype=torch.float32) + 0.1
    cfg = AAACConfig(group_size=128, selection_size=selection_size, outer_iters=1, inner_iters=2)
    layer_fit = fit_aaac_layer(model.proj.weight, importance, config=cfg)

    # Packed layer must match AAACLayerFit.dequantize() bit-for-bit
    packed = pack_aaac_fit(layer_fit)
    assert packed.codebooks.element_size() * packed.codebooks.numel() == 64

    replace_linear_layers_with_aaac(model, {"proj": layer_fit})

    assert isinstance(model.proj, AAACLinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert torch.equal(model.embed.weight, orig_embed)
    assert torch.equal(model.norm.weight, orig_norm_w)
    assert torch.equal(model.lm_head.weight, orig_lm_head)
    assert torch.equal(model.proj.bias, orig_proj_bias)

    ref_w = layer_fit.dequantize(torch.float32)
    packed_w = model.proj.dequantize_weight(torch.float32)
    assert torch.equal(packed_w, ref_w)

    x = torch.randn(2, 5, 128, dtype=torch.float32)
    expected_out = F.linear(x, ref_w, orig_proj_bias)
    actual_out = model.proj(x)
    assert torch.allclose(actual_out, expected_out, atol=1e-6, rtol=1e-6)

    storage = measure_model_storage_bytes(model)
    assert storage["quantized_layer_count"] == 1
    assert storage["codebooks_bytes"] == 64