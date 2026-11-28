"""Turn a YAML config (and, for ORA-Q, a role report) into a quantized model."""

import copy

import yaml

from .qmodules import quantize_model

OPS = {"qkv", "matmul1", "matmul2", "proj", "fc1", "fc2", "conv"}


def _is_int(v, low, high=None):
    return type(v) is int and v >= low and (high is None or v <= high)


def validate_config(cfg):
    """Raise ValueError on a config that would be misread or silently ignored."""
    def check(ok, msg):
        if not ok:
            raise ValueError(f"bad config: {msg}")

    check(isinstance(cfg, dict), "not a mapping")
    for key in ("method", "bits", "calibration", "ops"):
        check(key in cfg, f"missing '{key}'")
    for key in ("weight", "act"):
        check(_is_int(cfg["bits"].get(key), 2, 16), f"bits.{key} must be an integer from 2 to 16")
    cal = cfg["calibration"]
    for key in ("dataset_id", "split", "preprocessing_id", "num_images", "seed"):
        check(key in cal, f"missing calibration.{key}")
    check(cal["split"] == "train", "calibration.split must be 'train', never the test split")
    check(_is_int(cal["num_images"], 1), "calibration.num_images must be a positive integer")
    check(_is_int(cal["seed"], 0), "calibration.seed must be a non-negative integer")
    check(isinstance(cfg["ops"], dict) and cfg["ops"], "ops must be a non-empty mapping")
    for op, c in cfg["ops"].items():
        check(op in OPS, f"unknown operator '{op}'")
        check(c.get("g") is None or _is_int(c["g"], 1), f"{op}.g must be null or a positive integer")
        check(c.get("calib", "minmax") in ("minmax", "percentile"), f"{op}.calib is unknown")
        check(c.get("mode", "static") in ("static", "dynamic"), f"{op}.mode is unknown")
        if c.get("diagnose"):
            check(isinstance(cfg.get("tau"), float), "tau is required for diagnosed operators")
    return cfg


def load_config(path):
    with open(path) as f:
        return validate_config(yaml.safe_load(f))


def op_configs(cfg, role_report=None):
    """Final per-operator settings.

    Baselines: the YAML says everything. ORA-Q: the YAML gives the group size,
    the role report gives calibrator and static/dynamic for diagnosed operators.
    """
    ops = {}
    for op, c in cfg["ops"].items():
        c = dict(c)
        if c.get("diagnose"):
            if role_report is None:
                raise ValueError(f"{op} needs a role report (run the diagnosis first)")
            r = role_report["operators"][op]
            c["calib"], c["mode"] = r["calibrator"], r["mode"]
        c.setdefault("calib", "minmax")
        c.setdefault("mode", "static")
        ops[op] = c
    return ops


def build_quant_model(fp_model, cfg, role_report=None, act_bits=None, weight_bits=None):
    """Fresh quantized copy of fp_model. The FP model is left untouched."""
    model = copy.deepcopy(fp_model)
    bits = cfg["bits"]
    quantize_model(model, op_configs(cfg, role_report),
                   act_bits=act_bits or bits["act"],
                   weight_bits=weight_bits or bits["weight"])
    return model
