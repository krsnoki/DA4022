"""CPU tests for the NYUv2 loader, preprocessing, config checks and repeatability.

No download: the dataset files are tiny synthetic stand-ins.
Run from the repo root:  pytest -q Quantization_and_Pruning/tests
"""

import copy
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Backbones.Task_Finetuned import depth_anything_v2 as dav2  # noqa: E402
from Dataloaders.evaluation import nyuv2  # noqa: E402
from Quantization_and_Pruning.quantization.ptq import ora_q  # noqa: E402
from Quantization_and_Pruning.tests.test_ora_q import CONFIGS, inputs, tiny_model  # noqa: E402


@pytest.fixture
def fake_mat(tmp_path):
    """Three frames stored the way the real v7.3 file stores them (transposed)."""
    rgb = np.random.default_rng(0).integers(0, 256, size=(3, 480, 640, 3), dtype=np.uint8)
    depth = np.random.default_rng(1).uniform(0.5, 9.0, size=(3, 480, 640)).astype(np.float32)
    path = tmp_path / "fake.mat"
    with h5py.File(path, "w") as f:
        f["images"] = rgb.transpose(0, 3, 2, 1)     # (N, 3, W, H)
        f["depths"] = depth.transpose(0, 2, 1)      # (N, W, H)
    return path, rgb, depth


# ---- loader ----------------------------------------------------------------

def test_load_nyu_shapes_dtypes_orientation(fake_mat):
    path, rgb, depth = fake_mat
    out = nyuv2.load_nyu(path, [2, 0])
    assert len(out) == 2
    for (x, d), i in zip(out, [2, 0]):              # order follows the indices
        assert x.shape == (480, 640, 3) and x.dtype == np.uint8
        assert d.shape == (480, 640) and d.dtype == np.float32
        assert np.array_equal(x, rgb[i]) and np.array_equal(d, depth[i])


def test_load_nyu_boundaries(fake_mat):
    path = fake_mat[0]
    assert nyuv2.load_nyu(path, []) == []
    assert len(nyuv2.load_nyu(path, [0])) == 1
    assert len(nyuv2.load_nyu(path, [0, 1, 2])) == 3
    with pytest.raises((IndexError, ValueError)):
        nyuv2.load_nyu(path, [3])


def test_split_indices_become_zero_based(tmp_path):
    path = tmp_path / "splits.mat"
    scipy.io.savemat(path, {"testNdxs": np.array([[1], [4]]), "trainNdxs": np.array([[2], [3]])})
    assert nyuv2.load_split_indices(path) == {"test": [0, 3], "train": [1, 2]}


def test_calibration_sampling_is_repeatable_and_train_only():
    train, test = list(range(0, 200, 2)), list(range(1, 200, 2))
    a = nyuv2.sample_calibration(train, 16, 0)
    assert a == nyuv2.sample_calibration(train, 16, 0)
    assert a != nyuv2.sample_calibration(train, 16, 1)
    assert len(set(a)) == 16 and a == sorted(a)
    assert set(a) <= set(train) and not set(a) & set(test)


def test_valid_mask_crop_and_depth_range():
    d = np.ones((480, 640), np.float32)
    d[100, 100], d[100, 101], d[100, 102] = 0.0, 10.5, 10.0
    m = nyuv2.valid_mask(d)
    assert m.dtype == bool and m.shape == d.shape
    assert not m[100, 100] and not m[100, 101] and m[100, 102]
    assert m[45, 41] and m[470, 600]
    assert not m[44, 41] and not m[45, 40] and not m[471, 600] and not m[470, 601]


def test_test_time_preprocessing_has_no_randomness():
    from transformers import DPTImageProcessor
    # the values in preprocessor_config.json at the pinned revision
    proc = DPTImageProcessor(do_resize=True, size={"height": 518, "width": 518}, keep_aspect_ratio=True,
                             ensure_multiple_of=14, resample=3, do_rescale=True, do_normalize=True,
                             image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    rgb = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    a, b = dav2.preprocess(proc, rgb, "cpu"), dav2.preprocess(proc, rgb, "cpu")
    assert a.shape == (1, 3, 518, 686) and a.dtype == torch.float32
    assert torch.equal(a, b)


# ---- model pin -------------------------------------------------------------

def test_default_revision_is_pinned_and_matches_registry():
    reg = yaml.safe_load((ROOT / "Backbones" / "registry.yaml").read_text())
    entry = reg["vision.depth-anything-v2-small"]
    assert dav2.resolve_revision() == dav2.REVISION == entry["revision"]
    assert dav2.resolve_revision(revision="abc") == "abc"


# ---- config checks ---------------------------------------------------------

@pytest.mark.parametrize("name", ["minmax", "percentile", "ora_q"])
def test_shipped_configs_are_valid(name):
    cfg = ora_q.load_config(CONFIGS / f"{name}.yaml")
    assert cfg["method"] == name
    assert cfg["calibration"]["split"] == "train"


@pytest.mark.parametrize("path,value", [
    (("calibration", "split"), "test"),
    (("calibration", "num_images"), 0),
    (("calibration", "seed"), None),
    (("bits", "act"), 1),
    (("bits", "weight"), "4"),
    (("ops", "fc2", "g"), 0),
    (("ops", "fc2", "calib"), "median"),
    (("ops", "qkv", "mode"), "sometimes"),
    (("ops", "fc3"), {"g": None}),
    (("tau",), None),
])
def test_bad_config_is_rejected(path, value):
    cfg = copy.deepcopy(ora_q.load_config(CONFIGS / "ora_q.yaml"))
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="bad config"):
        ora_q.validate_config(cfg)


@pytest.mark.parametrize("key", ["method", "bits", "calibration", "ops"])
def test_missing_section_is_rejected(key):
    cfg = ora_q.load_config(CONFIGS / "ora_q.yaml")
    del cfg[key]
    with pytest.raises(ValueError, match="bad config"):
        ora_q.validate_config(cfg)


# ---- repeatability ---------------------------------------------------------

def run_once(name):
    fp, xs = tiny_model(), inputs()
    cfg = ora_q.load_config(CONFIGS / f"{name}.yaml")
    torch.manual_seed(0)                      # percentile calibration samples at random
    report = ora_q.diagnose(fp, xs, cfg) if name == "ora_q" else None
    m = ora_q.build_quant_model(fp, cfg, report)
    ora_q.calibrate(m, xs)
    return report, m(pixel_values=xs[0]).predicted_depth


@pytest.mark.parametrize("name", ["minmax", "percentile", "ora_q"])
def test_same_seed_gives_same_result(name):
    r1, y1 = run_once(name)
    r2, y2 = run_once(name)
    assert r1 == r2
    assert torch.equal(y1, y2)
