"""CPU smoke tests for ORA-Q. No download: the model is a tiny random Depth Anything.

Run from the repo root:  pytest -q Quantization_and_Pruning/tests/test_ora_q.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Performance_Metrics.depth_metrics import align_scale_shift, depth_metrics  # noqa: E402
from Quantization_and_Pruning.quantization.ptq import ora_q  # noqa: E402
from Quantization_and_Pruning.quantization.ptq.ora_q.quantizer import (  # noqa: E402
    fake_quant, fake_quant_weight, from_groups, minmax, qparams_from_range, to_groups)

CONFIGS = ROOT / "Quantization_and_Pruning" / "configs" / "ora_q"


def tiny_model():
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config
    torch.manual_seed(0)
    bb = Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=4,
                      image_size=70, patch_size=14, out_features=["stage1", "stage2"],
                      reshape_hidden_states=False)
    cfg = DepthAnythingConfig(backbone_config=bb, reassemble_hidden_size=64, neck_hidden_sizes=[16, 32],
                              reassemble_factors=[2, 1], fusion_hidden_size=16, head_hidden_size=8,
                              patch_size=14)
    return DepthAnythingForDepthEstimation(cfg).eval()


def inputs(n=3):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(1, 3, 70, 98, generator=g) for _ in range(n)]   # 5 x 7 patches + cls = 36 tokens


# ---- quantizer math -------------------------------------------------------

def test_16bit_is_nearly_lossless():
    x = torch.randn(1000)
    s, z = qparams_from_range(x.min(), x.max(), 16)
    assert (fake_quant(x, s, z, 16) - x).abs().max() < 1e-3


def test_4bit_uses_at_most_16_levels_and_error_bounded():
    x = torch.randn(5000)
    s, z = qparams_from_range(x.min(), x.max(), 4)
    y = fake_quant(x, s, z, 4)
    assert y.unique().numel() <= 16
    assert (y - x).abs().max() <= s / 2 + 1e-6


def test_idempotent():
    x = torch.randn(200)
    s, z = qparams_from_range(x.min(), x.max(), 4)
    y = fake_quant(x, s, z, 4)
    assert torch.allclose(fake_quant(y, s, z, 4), y)


def test_constant_tensor_no_nan():
    x = torch.full((10,), 3.0)
    s, z = qparams_from_range(x.min(), x.max(), 4)
    assert torch.isfinite(fake_quant(x, s, z, 4)).all()


def test_primer_worked_example():
    x = torch.tensor([-1.0, 0.2, 0.5, 3.0])
    s, z = qparams_from_range(x.min(), x.max(), 2)
    assert z.item() == 1
    assert torch.allclose(fake_quant(x, s, z, 2), torch.tensor([-4 / 3, 0.0, 0.0, 8 / 3]), atol=1e-6)


@pytest.mark.parametrize("shape,axis,g", [((2, 5, 12), -1, 4), ((2, 6, 3, 3), 1, 2),
                                          ((1, 2, 7, 4), -2, 2), ((3, 8), -1, None)])
def test_group_roundtrip(shape, axis, g):
    x = torch.randn(shape)
    xg, info = to_groups(x, axis, g)
    assert torch.equal(from_groups(xg, info), x)


def test_groups_get_their_own_scale():
    x = torch.randn(4, 8)
    x[:, 4:] *= 100                       # second group of 4 channels is huge
    xg, _ = to_groups(x, -1, 4)
    lo, hi = minmax(xg, [-2])
    assert (hi - lo).flatten()[1] > 10 * (hi - lo).flatten()[0]


def test_weight_quant_per_channel():
    w = torch.randn(6, 5, 3, 3)
    wq = fake_quant_weight(w, 4)
    assert wq.shape == w.shape
    assert all(wq[i].unique().numel() <= 16 for i in range(6))


# ---- metrics -------------------------------------------------------------

def test_alignment_recovers_scale_and_shift():
    gt = np.random.default_rng(0).uniform(0.5, 9.0, size=(480, 640)).astype(np.float32)
    mask = np.ones_like(gt, dtype=bool)
    pred = align_scale_shift(3.0 / gt + 0.2, gt, mask)
    m = depth_metrics(pred, gt, mask)
    assert m["delta1"] == 1.0 and m["abs_rel"] < 1e-4


# ---- whole pipeline on a tiny model ---------------------------------------

def test_quantizers_off_match_fp():
    fp, x = tiny_model(), inputs(1)[0]
    ref = fp(pixel_values=x).predicted_depth
    m = ora_q.build_quant_model(fp, ora_q.load_config(CONFIGS / "minmax.yaml"), weight_bits=None)
    ora_q.set_state(m, "off")
    assert torch.allclose(m(pixel_values=x).predicted_depth, ref, atol=1e-5)


def test_16bit_close_to_fp():
    fp, xs = tiny_model(), inputs()
    ref = fp(pixel_values=xs[0]).predicted_depth
    m = ora_q.build_quant_model(fp, ora_q.load_config(CONFIGS / "minmax.yaml"), act_bits=16, weight_bits=16)
    ora_q.calibrate(m, xs)
    out = m(pixel_values=xs[0]).predicted_depth
    assert (out - ref).abs().max() / ref.abs().max() < 1e-2


def test_counts_and_4bit_changes_output():
    fp, xs = tiny_model(), inputs()
    m = ora_q.build_quant_model(fp, ora_q.load_config(CONFIGS / "percentile.yaml"))
    counts = ora_q.count_by_op(m)
    # 2 blocks: 3 qkv, 1 proj, 1 fc1, 1 fc2, 2 matmul1, 2 matmul2 each; 17 decoder convs (2 reassemble + 2 neck + 10 fusion + 3 head)
    assert counts == {"qkv": 6, "proj": 2, "fc1": 2, "fc2": 2, "matmul1": 4, "matmul2": 4, "conv": 17}
    ora_q.calibrate(m, xs)
    out = m(pixel_values=xs[0]).predicted_depth
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, fp(pixel_values=xs[0]).predicted_depth)


def test_full_ora_q_pipeline(tmp_path):
    fp, xs = tiny_model(), inputs()
    cfg = ora_q.load_config(CONFIGS / "ora_q.yaml")
    report = ora_q.diagnose(fp, xs, cfg)
    ops = report["operators"]
    assert set(ops) == {"matmul1", "fc2", "proj", "conv", "matmul2"}
    assert ops["matmul2"]["mode"] == "dynamic"
    for op in ("matmul1", "fc2", "proj", "conv"):
        assert ops[op]["role"] in ("clip-tolerant", "clip-sensitive")
        assert ops[op]["clip_score"] >= 0
        assert np.isfinite(ops[op]["S"])
    m = ora_q.build_quant_model(fp, cfg, report)
    ora_q.calibrate(m, xs)
    out = m(pixel_values=torch.randn(1, 3, 84, 112)).predicted_depth   # a different size works too
    assert out.shape == (1, 84, 112) and torch.isfinite(out).all()
