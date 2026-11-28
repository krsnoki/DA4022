"""Reproduce FP32, W4A4 MinMax, W4A4 Percentile and ORA-Q on NYUv2 with DAv2-Small.

Example (from the repo root):
  python Quantization_and_Pruning/experiments/ora_q/run_ora_q.py \
      --mat /path/nyu_depth_v2_labeled.mat --splits /path/splits.mat \
      --methods fp32 minmax percentile ora_q --num-images 654

Each method appends one row to results.csv. ORA-Q also writes role_report.json.
Run details (versions, hardware, data hashes, timings, raw latency samples) go to
a JSON file next to the CSV (--meta), one entry per call.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

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


def git(*cmd):
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *cmd], text=True).strip()
    except Exception:
        return "unknown"


def git_commit():
    return git("rev-parse", "--short", "HEAD")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(device):
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": np.__version__,
           "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "commit": git("rev-parse", "HEAD"),
           "uncommitted_changes": git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown")}
    if device == "cuda":
        prop = torch.cuda.get_device_properties(0)
        env.update(gpu=prop.name, gpu_memory_mb=round(prop.total_memory / 2**20),
                   cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        try:
            env["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                text=True).strip()
        except Exception:
            env["nvidia_smi"] = "unavailable"
    return env


def median_ms(xs):
    return round(float(np.median(xs)) * 1000, 3) if xs else None


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
    ap.add_argument("--meta", default=None, help="run details JSON (default: next to --out)")
    ap.add_argument("--hash-data", action="store_true", help="store SHA-256 of --mat and --splits")
    ap.add_argument("--seed-per-method", action="store_true",
                    help="reset the seed before each method, so a row does not depend on run order")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    meta_path = args.meta or str(Path(args.out).with_suffix("")) + "_meta.json"
    run = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": vars(args),
           "environment": environment(device), "methods": {}}
    print(json.dumps(run["environment"], indent=1))
    if run["environment"]["uncommitted_changes"]:
        print("WARNING: tracked files differ from the commit; the commit field will not describe this run")
    if args.hash_data:
        run["data_sha256"] = {"mat": sha256(args.mat), "splits": sha256(args.splits)}
        print("data sha256:", run["data_sha256"])

    t0 = time.time()
    fp_model, processor, revision = dav2.load(device, revision=args.revision)
    if device == "cuda":
        torch.cuda.synchronize()
    run["model_load_seconds"] = round(time.time() - t0, 2)
    print(f"model revision: {revision}")

    split = nyuv2.load_split_indices(args.splits)
    calib_idx = nyuv2.sample_calibration(split["train"], args.calib_images, args.seed)
    test_idx = split["test"][:args.num_images]
    print(f"calibration (train) indices: {calib_idx}")
    t0 = time.time()
    calib = nyuv2.load_nyu(args.mat, calib_idx)
    test = nyuv2.load_nyu(args.mat, test_idx)
    run["data_load_seconds"] = round(time.time() - t0, 2)
    calib_pv = [dav2.preprocess(processor, rgb, device) for rgb, _ in calib]
    run.update(model_revision=revision, calibration_indices=calib_idx, num_test_images=len(test))
    print(f"loaded {len(calib)} calibration and {len(test)} test images")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for method in args.methods:
        print(f"\n=== {method} ===")
        if args.seed_per_method:
            torch.manual_seed(args.seed)
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

        # per image: processor on CPU, copy to device, then forward + resize + copy back
        times = {"preprocess": [], "transfer": [], "forward_and_postprocess": []}

        def predict(rgb):
            a = time.perf_counter()
            pv = dav2.preprocess(processor, rgb, "cpu")
            b = time.perf_counter()
            pv = pv.to(device)
            sync()
            c = time.perf_counter()
            out = dav2.predict_disparity(model, pv, rgb.shape[:2])
            d = time.perf_counter()
            times["preprocess"].append(b - a)
            times["transfer"].append(c - b)
            times["forward_and_postprocess"].append(d - c)
            return out

        t1 = time.time()
        metrics = evaluate(predict, test)
        eval_s = time.time() - t1
        print(f"{method}: delta1 {metrics['delta1']:.4f}  abs_rel {metrics['abs_rel']:.4f}")

        runtime = {}
        if not args.no_latency:
            runtime = latency_and_memory(model, dav2.preprocess(processor, test[0][0], device))
            print(f"latency median {runtime['latency_median_ms']:.1f} ms, peak {runtime['peak_memory_mb']} MB")
        run["methods"][method] = {
            "setup_seconds": round(calib_s, 2), "eval_seconds": round(eval_s, 2),
            "per_image_median_ms": {k: median_ms(v) for k, v in times.items()},
            "latency_samples_ms": [round(t, 3) for t in runtime.get("samples_ms", [])],
            "delta1": metrics["delta1"], "abs_rel": metrics["abs_rel"],
        }
        print("per-image median ms:", run["methods"][method]["per_image_median_ms"])

        a_bits = args.act_bits or 4
        w_bits = args.weight_bits or 4
        row = {
            "experiment_id": f"oraq-nyuv2-{method}-w{w_bits}a{a_bits}-s{args.seed}" if method != "fp32"
                             else f"oraq-nyuv2-fp32-s{args.seed}",
            "paper_id": "ora-q", "method": method, "commit": git_commit(),
            "backbone_id": "vision.depth-anything-v2-small", "backbone_revision": revision,
            "dataset_id": "vision.nyuv2.test", "split": f"eigen_test[:{len(test)}]", "seed": args.seed,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
            "framework_version": f"torch {torch.__version__}; transformers {transformers.__version__}",
            "dtype": "float32" if method == "fp32" else f"fake-quant W{w_bits}A{a_bits} (float32 compute)",
            "batch_size": 1, "input_length": "x".join(map(str, calib_pv[0].shape[-2:])),
            "output_length": "480x640", "quality_metric": "delta1",
            "quality_value": round(metrics["delta1"], 4), "abs_rel": round(metrics["abs_rel"], 4),
            "latency_median_ms": round(runtime["latency_median_ms"], 2) if runtime else None,
            "latency_p95_ms": round(runtime["latency_p95_ms"], 2) if runtime else None,
            "throughput": round(runtime["throughput"], 2) if runtime else None,
            "peak_memory_mb": round(runtime["peak_memory_mb"], 1) if runtime and runtime["peak_memory_mb"] else None,
            "energy_j": None, "calib_seconds": round(calib_s, 1) if method != "fp32" else None,
            "status": "partial", "notes": f"{notes}; details in {Path(meta_path).name}",
        }
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        save_meta(meta_path, run)
        if model is not fp_model:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nresults appended to {args.out}, run details in {meta_path}")


def save_meta(path, run):
    """One JSON list per file; this call's entry is replaced as methods finish."""
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
    runs = [r for r in runs if r.get("started") != run["started"]] + [run]
    with open(path, "w") as f:
        json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
