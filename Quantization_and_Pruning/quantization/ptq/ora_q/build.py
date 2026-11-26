"""Turn a YAML config (and, for ORA-Q, a role report) into a quantized model."""

import copy

import yaml

from .qmodules import quantize_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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
