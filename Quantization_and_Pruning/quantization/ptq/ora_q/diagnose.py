"""ORA-Q diagnosis: decide each operator's outlier role from calibration images.

1. Structural analysis (Sec. 2.2, eq. 1-2): stability score S from how much each
   group's range moves between images.
2. Distributional analysis (Sec. 3.2, eq. 5): clip score from how much the FP
   depth output changes when one operator's input is clipped at its 99.5th
   percentile.

Everything here runs on the full-precision model (FP weights, quantizers only
observing or clipping, never quantizing).
"""

import copy

import torch

from .qmodules import calibrate, quantizers, quantize_model, set_state


def stability_scores(model, eps=1e-8):
    """S per operator: mean over groups of std over images of log(range).

    Each layer instance gives one S; the operator's S is the mean over its
    layers (the paper does not say how layers are combined).
    """
    per_op = {}
    for q in quantizers(model):
        if not q.ranges:
            continue
        r = torch.stack(q.ranges)                       # (images, groups)
        s = torch.log(r.clamp(min=eps)).std(dim=0, correction=0).mean().item()
        per_op.setdefault(q.op, []).append(s)
    return {op: sum(v) / len(v) for op, v in per_op.items()}


@torch.no_grad()
def clip_scores(model, calib_inputs, ops):
    """Relative L1 change of the depth output when one operator class is clipped.

    All layer instances of that operator are clipped together, per image, top
    side only, at their own 99.5th percentile.
    """
    set_state(model, "off")
    y_fp = [model(pixel_values=pv).predicted_depth for pv in calib_inputs]
    denom = sum(y.abs().sum().item() for y in y_fp) / len(y_fp)

    scores = {}
    for op in ops:
        set_state(model, "clip", ops=[op])
        diff = 0.0
        for pv, y in zip(calib_inputs, y_fp):
            diff += (model(pixel_values=pv).predicted_depth - y).abs().sum().item()
        scores[op] = (diff / len(calib_inputs)) / denom
    set_state(model, "off")
    return scores


def diagnose(fp_model, calib_inputs, cfg):
    """Run both analyses and return the role report (a JSON-ready dict)."""
    ops_cfg = cfg["ops"]
    diag_ops = [op for op, c in ops_cfg.items() if c.get("diagnose")]

    # observe-only copy with ORA-Q's grouping, static everywhere so S can be computed
    model = copy.deepcopy(fp_model)
    observe_cfg = {op: {**ops_cfg[op], "mode": "static", "calib": "minmax"} for op in diag_ops}
    quantize_model(model, observe_cfg, act_bits=cfg["bits"]["act"], weight_bits=None)
    calibrate(model, calib_inputs, record_ranges=True)
    set_state(model, "off")

    S = stability_scores(model)
    stable_ops = [op for op in diag_ops if ops_cfg[op].get("architectural") != "input_dependent"]
    clip = clip_scores(model, calib_inputs, stable_ops)

    tau = cfg["tau"]
    operators = {}
    for op in diag_ops:
        c = ops_cfg[op]
        entry = {"S": round(S.get(op, float("nan")), 4), "granularity": _gran(c)}
        if op not in stable_ops:
            # the paper treats matmul2 as input-dependent by architecture (Sec. 2.2, Table 2)
            entry.update(clip_score=None, role="input-dependent", calibrator="minmax", mode="dynamic")
        else:
            score = clip[op]
            tolerant = score < tau
            entry.update(clip_score=round(score, 4),
                         role="clip-tolerant" if tolerant else "clip-sensitive",
                         calibrator="percentile" if tolerant else "minmax",
                         mode="static")
        operators[op] = entry
    del model
    return {"tau": tau, "operators": operators}


def _gran(c):
    if c.get("g") is None:
        return {"type": "tensor"}
    g = {"type": "group", "g": c["g"]}
    if c.get("per_head"):
        g["per_head"] = True
    return g
