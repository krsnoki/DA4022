"""Reproduce FP32, W4A4 MinMax, W4A4 Percentile and ORA-Q on NYUv2 with DAv2-Small.

Example (from the repo root):
  python Quantization_and_Pruning/experiments/ora_q/run_ora_q.py \
      --mat /path/nyu_depth_v2_labeled.mat --splits /path/splits.mat \
      --methods fp32 minmax percentile ora_q --num-images 654

Each method appends one row to results.csv. ORA-Q also writes role_report.json.
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from Backbones.Task_Finetuned import depth_anything_v2 as dav2  # noqa: E402
from Dataloaders.evaluation import nyuv2  # noqa: E402
from Performance_Metrics.depth_metrics import evaluate, latency_and_memory  # noqa: E402
from Quantization_and_Pruning.quantization.ptq import ora_q  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = ROOT / "Quantization_and_Pruning" / "configs" / "ora_q"
CONFIG_OF = {"minmax": "minmax.yaml", "percentile": "percentile.yaml", "ora_q": "ora_q.yaml"}
CSV_FIELDS = ["experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision",
              "dataset_id", "split", "seed", "device", "framework_version", "dtype", "batch_size",
              "input_length", "output_length", "quality_metric", "quality_value", "abs_rel",
              "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb", "energy_j",
              "calib_seconds", "status", "notes"]


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--methods", nargs="+", default=["fp32", "minmax", "percentile", "ora_q"])
    ap.add_argument("--num-images", type=int, default=654, help="first N test images (654 = all)")
    ap.add_argument("--calib-images", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-bits", type=int, default=None, help="override, e.g. 16 for a sanity check")
    ap.add_argument("--weight-bits", type=int, default=None)
    ap.add_argument("--role-report", default=str(HERE / "role_report.json"))
    ap.add_argument("--rediagnose", action="store_true", help="redo the diagnosis even if the JSON exists")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    fp_model, processor, revision = dav2.load(device, revision=args.revision)
    print(f"model revision: {revision}")

    split = nyuv2.load_split_indices(args.splits)
    calib_idx = nyuv2.sample_calibration(split["train"], args.calib_images, args.seed)
    test_idx = split["test"][:args.num_images]
    print(f"calibration (train) indices: {calib_idx}")
    calib = nyuv2.load_nyu(args.mat, calib_idx)
    test = nyuv2.load_nyu(args.mat, test_idx)
    calib_pv = [dav2.preprocess(processor, rgb, device) for rgb, _ in calib]
    print(f"loaded {len(calib)} calibration and {len(test)} test images")

    for method in args.methods:
        print(f"\n=== {method} ===")
        t0 = time.time()
        if method == "fp32":
            model, notes = fp_model, "unmodified checkpoint"
        else:
            cfg = ora_q.load_config(CONFIGS / CONFIG_OF[method])
            report = None
            if method == "ora_q":
                if args.rediagnose or not os.path.exists(args.role_report):
                    print("diagnosing operator roles ...")
                    report = ora_q.diagnose(fp_model, calib_pv, cfg)
                    report.update(backbone_id="vision.depth-anything-v2-small", revision=revision,
                                  calibration={"dataset_id": "vision.nyuv2.calibration",
                                               "split": "train", "indices": calib_idx,
                                               "seed": args.seed})
                    with open(args.role_report, "w") as f:
                        json.dump(report, f, indent=2)
                    print(f"wrote {args.role_report}")
                else:
                    with open(args.role_report) as f:
                        report = json.load(f)
                print(json.dumps(report["operators"], indent=2))
            model = ora_q.build_quant_model(fp_model, cfg, report, args.act_bits, args.weight_bits)
            print("quantizers per operator:", ora_q.count_by_op(model))
            ora_q.calibrate(model, calib_pv)
            notes = f"fake quantization; config {CONFIG_OF[method]}"
        calib_s = time.time() - t0

        metrics = evaluate(lambda rgb: dav2.predict_disparity(model, dav2.preprocess(processor, rgb, device),
                                                              rgb.shape[:2]), test)
        print(f"{method}: delta1 {metrics['delta1']:.4f}  abs_rel {metrics['abs_rel']:.4f}")

        runtime = {}
        if not args.no_latency:
            runtime = latency_and_memory(model, dav2.preprocess(processor, test[0][0], device))
            print(f"latency median {runtime['latency_median_ms']:.1f} ms, peak {runtime['peak_memory_mb']} MB")

        a_bits = args.act_bits or 4
        w_bits = args.weight_bits or 4
        row = {
            "experiment_id": f"oraq-nyuv2-{method}-w{w_bits}a{a_bits}-s{args.seed}" if method != "fp32"
                             else f"oraq-nyuv2-fp32-s{args.seed}",
            "paper_id": "ora-q", "method": method, "commit": git_commit(),
            "backbone_id": "vision.depth-anything-v2-small", "backbone_revision": revision,
            "dataset_id": "vision.nyuv2.test", "split": f"eigen_test[:{len(test)}]", "seed": args.seed,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
            "framework_version": f"torch {torch.__version__}",
            "dtype": "float32" if method == "fp32" else f"fake-quant W{w_bits}A{a_bits} (float32 compute)",
            "batch_size": 1, "input_length": "x".join(map(str, calib_pv[0].shape[-2:])),
            "output_length": "480x640", "quality_metric": "delta1",
            "quality_value": round(metrics["delta1"], 4), "abs_rel": round(metrics["abs_rel"], 4),
            "latency_median_ms": round(runtime["latency_median_ms"], 2) if runtime else None,
            "latency_p95_ms": round(runtime["latency_p95_ms"], 2) if runtime else None,
            "throughput": round(runtime["throughput"], 2) if runtime else None,
            "peak_memory_mb": round(runtime["peak_memory_mb"], 1) if runtime and runtime["peak_memory_mb"] else None,
            "energy_j": None, "calib_seconds": round(calib_s, 1) if method != "fp32" else None,
            "status": "partial", "notes": notes,
        }
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        if model is not fp_model:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
