from __future__ import annotations

import torch
import torch.nn as nn

from Quantization_and_Pruning.quantization.aaac.linear import measure_model_storage_bytes
from Quantization_and_Pruning.quantization.aaac.rtn import RTNLinear, quantize_model_rtn


def test_rtn_packing_bytes_and_reconstruction():
    torch.manual_seed(123)
    lin = nn.Linear(256, 64, bias=True)
    lin.weight.data[0, :128] = 0.0

    rtn = RTNLinear.from_linear(lin, group_size=128, scale_dtype=torch.float16)
    assert rtn.packed_indices.shape == (64, 128)
    assert rtn.packed_indices.dtype == torch.uint8
    assert rtn.scales.shape == (64, 2)
    assert (rtn.scales > 0).all()

    w_hat = rtn.dequantize_weight(torch.float32)
    assert w_hat.shape == lin.weight.shape
    assert torch.allclose(w_hat[0, :128], torch.zeros(128), atol=1e-6)

    x = torch.randn(2, 4, 256)
    assert rtn(x).shape == (2, 4, 64)


def test_rtn_target_layer_parity():
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 32))
    orig_head = model[2].weight.detach().clone()
    quantize_model_rtn(model, target_layer_names=["0"], group_size=128)

    assert isinstance(model[0], RTNLinear)
    assert isinstance(model[2], nn.Linear)
    assert torch.equal(model[2].weight, orig_head)

    storage = measure_model_storage_bytes(model)
    assert storage["quantized_layer_count"] == 1
    assert storage["codebooks_bytes"] == 0
    assert storage["packed_indices_bytes"] == (128 * 128) // 2