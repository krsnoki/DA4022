"""Why does our diagnosis call fc2 clip-sensitive? Try other readings of the clip test.

Uses only the 16 train-split calibration images. The variant list is fixed here, before any run.

Run from the repo root (CPU). A second file, OUT with _layers added, gets the per-layer scores:
  python Quantization_and_Pruning/experiments/ora_q/diag_fc2_variants.py . MAT SPLITS OUT.csv
"""

import copy
import csv
import sys

import numpy as np
import torch

repo, mat, splits, out = sys.argv[1:5]
sys.path.append(repo)

from Backbones.Task_Finetuned import depth_anything_v2 as dav2  # noqa: E402
from Dataloaders.evaluation import nyuv2  # noqa: E402
from Performance_Metrics.depth_metrics import align_scale_shift, depth_metrics  # noqa: E402
from Quantization_and_Pruning.quantization.ptq import ora_q  # noqa: E402
from Quantization_and_Pruning.quantization.ptq.ora_q import qmodules as qm  # noqa: E402
from Quantization_and_Pruning.quantization.ptq.ora_q.quantizer import subsample, to_groups  # noqa: E402

OPS = ["matmul1", "fc2", "proj", "conv"]
Q = 0.995
VARIANTS = [
    "baseline",        # our code: input, top side, per layer and image, all layers of a class at once
    "two_sided",       # also clip the bottom 0.5%
    "magnitude",       # clip |x| at the 99.5th percentile of |x|
    "per_group",       # one top threshold per ORA-Q group
    "pooled_images",   # one top threshold per layer, from all 16 images
    "one_layer",       # clip one layer at a time, average the scores
    "output",          # clip the operator output instead of its input (not defined for matmul1)
]
mode = "baseline"
pooled = {}
orig_forward = qm.ActQuantizer.forward


def top(x, q=Q):
    return torch.quantile(subsample(x.detach().float(), 2_000_000), q)


def forward(self, x):
    if self.state == "collect":
        pooled.setdefault(self, []).append(subsample(x.detach().float(), 125_000).clone())
        return x
    if self.state != "clip" or mode in ("baseline", "one_layer"):
        return orig_forward(self, x)
    if mode == "output":
        return x
    if mode == "two_sided":
        return x.clamp(min=top(x, 1 - Q), max=top(x))
    if mode == "magnitude":
        t = top(x.abs())
        return x.clamp(-t, t)
    if mode == "pooled_images":
        return x.clamp(max=pooled[self])
    if mode == "per_group":
        xg, info = to_groups(x, self.axis, self.g)
        rows = qm._rows(xg, self._keep()).float()
        if rows.shape[1] > 200_000:
            rows = rows[:, torch.randint(0, rows.shape[1], (200_000,))]
        step = max(1, 16_000_000 // rows.shape[1])
        hi = torch.cat([torch.quantile(rows[i:i + step], Q, dim=1) for i in range(0, rows.shape[0], step)])
        keep = sorted(d % xg.dim() for d in self._keep())
        shape = [xg.shape[d] if d in keep else 1 for d in range(xg.dim())]
        return ora_q.from_groups(torch.minimum(xg, hi.reshape(shape)), info)
    raise ValueError(mode)


qm.ActQuantizer.forward = forward


@torch.no_grad()
def main():
    global mode
    torch.manual_seed(0)
    fp, processor, revision = dav2.load("cpu")
    split = nyuv2.load_split_indices(splits)
    idx = nyuv2.sample_calibration(split["train"], 16, 0)
    assert not set(idx) & set(split["test"])
    data = nyuv2.load_nyu(mat, idx)
    pvs = [dav2.preprocess(processor, rgb, "cpu") for rgb, _ in data]

    cfg = ora_q.load_config(f"{repo}/Quantization_and_Pruning/configs/ora_q/ora_q.yaml")
    model = copy.deepcopy(fp)
    ops_cfg = {o: {**cfg["ops"][o], "mode": "static", "calib": "minmax"} for o in OPS + ["matmul2"]}
    ora_q.quantize_model(model, ops_cfg, weight_bits=None)
    ora_q.set_state(model, "off")

    def run(m):
        return [m(pixel_values=pv).predicted_depth for pv in pvs]

    def d1(ys):
        vals = []
        for y, (_, gt) in zip(ys, data):
            disp = torch.nn.functional.interpolate(y[None], size=gt.shape, mode="bilinear",
                                                   align_corners=False)[0, 0].numpy()
            m = nyuv2.valid_mask(gt)
            vals.append(depth_metrics(align_scale_shift(disp, gt, m), gt, m)["delta1"])
        return float(np.mean(vals))

    ref = run(fp)
    off = run(model)
    gap = max((a - b).abs().max().item() for a, b in zip(ref, off))
    print(f"sanity: wrapped model with clipping off vs original, max abs diff {gap:.2e}", flush=True)
    denom = sum(y.abs().sum().item() for y in ref) / len(ref)
    ref_d1 = d1(ref)
    print(f"FP delta1 on the 16 calibration images: {ref_d1:.4f}", flush=True)

    def score(ys):
        rel = sum((a - b).abs().sum().item() for a, b in zip(ys, ref)) / len(ref) / denom
        return rel, d1(ys) - ref_d1

    # thresholds pooled over images
    ora_q.set_state(model, "collect")
    run(model)
    for q in list(pooled):
        pooled[q] = top(torch.cat(pooled[q]))
    ora_q.set_state(model, "off")

    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["variant", "operator", "layers", "rel_l1_score", "delta_delta1"])
    layer_f = open(out.replace(".csv", "_layers.csv"), "w", newline="")
    lw = csv.writer(layer_f)
    lw.writerow(["operator", "layer", "rel_l1_score", "delta_delta1"])

    def emit(variant, op, n, rel, dd):
        w.writerow([variant, op, n, round(rel, 4), round(dd, 4)])
        f.flush()
        print(f"{variant:14s} {op:8s} score {rel:.4f}  d(delta1) {dd:+.4f}", flush=True)

    names = {m: n for n, m in model.named_modules() if isinstance(m, qm.ActQuantizer)}
    for mode in VARIANTS:
        for op in OPS:
            qs = ora_q.quantizers(model, [op])
            ora_q.set_state(model, "off")
            torch.manual_seed(0)
            if mode == "one_layer":
                res = []
                for q in qs:
                    q.state = "clip"
                    res.append(score(run(model)))
                    q.state = "off"
                    lw.writerow([op, names[q], round(res[-1][0], 4), round(res[-1][1], 4)])
                    layer_f.flush()
                emit(mode, op, len(qs), float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res])))
                continue
            if mode == "output" and op == "matmul1":
                continue
            hooks = []
            if mode == "output":
                for mod in model.modules():
                    if isinstance(mod, (qm.QuantLinear, qm.QuantConv2d)) and mod.act.op == op:
                        hooks.append(mod.register_forward_hook(lambda m_, i_, o_: o_.clamp(max=top(o_))))
            else:
                ora_q.set_state(model, "clip", ops=[op])
            emit(mode, op, len(hooks) or len(qs), *score(run(model)))
            for h in hooks:
                h.remove()
    print("done", flush=True)


main()
