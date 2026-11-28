"""Saved model size and save/load drift on the real checkpoint, on CPU.

Run from the repo root:
  python Quantization_and_Pruning/experiments/ora_q/measure_size.py --mat MAT --splits SPLITS --out OUT.csv

The quantized models are fake quantized, so their weights are saved as float32.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Backbones.Task_Finetuned import depth_anything_v2 as dav2  # noqa: E402
from Dataloaders.evaluation import nyuv2  # noqa: E402
from Quantization_and_Pruning.quantization.ptq import ora_q  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = ROOT / "Quantization_and_Pruning" / "configs" / "ora_q"


def saved_bytes(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--role-report", default=str(HERE / "role_report.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fp, processor, revision = dav2.load("cpu")
    split = nyuv2.load_split_indices(args.splits)
    calib_idx = nyuv2.sample_calibration(split["train"], 16, args.seed)
    calib = [dav2.preprocess(processor, rgb, "cpu") for rgb, _ in nyuv2.load_nyu(args.mat, calib_idx)]
    x = dav2.preprocess(processor, nyuv2.load_nyu(args.mat, split["test"][:1])[0][0], "cpu")
    print("processor:", type(processor).__name__, "| input", tuple(x.shape),
          "| output", tuple(fp(pixel_values=x).predicted_depth.shape))
    print("torch", torch.__version__, "| transformers", transformers.__version__)
    with open(args.role_report) as f:
        report = json.load(f)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        params = sum(p.numel() for p in fp.parameters())
        rows.append({"method": "fp32", "revision": revision, "num_parameters": params,
                     "state_dict_bytes": saved_bytes(fp, path), "static_scale_zeropoint_bytes": 0,
                     "reload_max_abs_diff": ""})
        for method in ("minmax", "percentile", "ora_q"):
            cfg = ora_q.load_config(CONFIGS / f"{method}.yaml")
            rep = report if method == "ora_q" else None
            torch.manual_seed(args.seed)
            model = ora_q.build_quant_model(fp, cfg, rep)
            ora_q.calibrate(model, calib)
            ref = model(pixel_values=x).predicted_depth
            size = saved_bytes(model, path)
            qbytes = sum(v.numel() * v.element_size() for k, v in model.state_dict().items()
                         if k.endswith((".scale", ".zero_point")))

            # reload into a second model; two flipped images give it different scales first
            other = ora_q.build_quant_model(fp, cfg, rep)
            ora_q.calibrate(other, [c.flip(-1) * 1.5 for c in calib[:2]])
            assert not torch.equal(other(pixel_values=x).predicted_depth, ref)
            other.load_state_dict(torch.load(path, weights_only=True))
            drift = (other(pixel_values=x).predicted_depth - ref).abs().max().item()

            rows.append({"method": method, "revision": revision, "num_parameters": params,
                         "state_dict_bytes": size, "static_scale_zeropoint_bytes": qbytes,
                         "reload_max_abs_diff": drift})
            print(rows[-1])
            del model, other

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
