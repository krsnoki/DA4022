"""Config validation tests (no torch needed)."""
import copy

import pytest
import yaml

from conftest import CONFIG_DIR

from Quantization_and_Pruning.quantization.weight_only.cafeq.config import CafeQConfig, ConfigError

ALL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _base():
    return yaml.safe_load((CONFIG_DIR / "tiny_smoke.yaml").read_text())


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = CafeQConfig.from_yaml(path)
    assert cfg.calibration_dataset is None
    assert cfg.quantization.method in ("none", "rtn", "random_hadamard", "cafeq")


def test_calibration_dataset_must_be_explicit_and_null():
    data = _base()
    del data["calibration_dataset"]
    with pytest.raises(ConfigError, match="explicitly"):
        CafeQConfig.from_dict(data)
    data = _base()
    data["calibration_dataset"] = "text.c4.train"
    with pytest.raises(ConfigError, match="calibration-free"):
        CafeQConfig.from_dict(data)


@pytest.mark.parametrize(
    "path,value",
    [
        (("quantization", "bits"), 1),
        (("quantization", "method"), "gptq"),
        (("quantization", "group_size"), 0),
        (("block_transform", "loss"), "l1"),
        (("block_transform", "lr"), 0),
        (("paired_transform", "temperature"), -1.0),
        (("paired_transform", "pseudo_loss"), "max"),
        (("evaluation", "seq_len"), 0),
        (("backbone", "dtype"), "int8"),
    ],
)
def test_invalid_values_rejected(path, value):
    data = copy.deepcopy(_base())
    data[path[0]][path[1]] = value
    with pytest.raises(ConfigError):
        CafeQConfig.from_dict(data)


def test_unknown_keys_rejected():
    data = _base()
    data["quantization"]["bitz"] = 4
    with pytest.raises(ConfigError, match="unknown"):
        CafeQConfig.from_dict(data)
    data = _base()
    data["extra"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        CafeQConfig.from_dict(data)


def test_nested_targets_parse():
    data = _base()
    data["quantization"]["targets"] = {"ff": True, "qk": False, "vo": False}
    cfg = CafeQConfig.from_dict(data)
    assert cfg.quantization.targets.ff and not cfg.quantization.targets.qk
