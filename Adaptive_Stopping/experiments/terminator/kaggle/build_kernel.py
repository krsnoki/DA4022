"""Build a self-contained Kaggle script kernel from repo files.

The kernel script embeds the repo files listed in FILES, writes them to
/kaggle/tmp/src on the Kaggle machine, and then runs a stage driver
(one of the stage_*.py files next to this script).

Example:
    python Adaptive_Stopping/experiments/terminator/kaggle/build_kernel.py \
        --stage stage_traces.py --slug terminator-traces --out-dir E:/kernels/traces \
        --env SMOKE=1
    kaggle kernels push -p E:/kernels/traces --accelerator NvidiaTeslaT4

The Kaggle user name is read from the local Kaggle API config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

FILES = [
    "Dataloaders/evaluation/math_reasoning.py",
    "Adaptive_Stopping/methods/learned_exit/terminator/__init__.py",
    "Adaptive_Stopping/methods/learned_exit/terminator/generation.py",
    "Adaptive_Stopping/methods/learned_exit/terminator/labels.py",
    "Adaptive_Stopping/methods/learned_exit/terminator/policy.py",
    "Adaptive_Stopping/methods/learned_exit/terminator/probe.py",
    "Adaptive_Stopping/experiments/terminator/generate_traces.py",
    "Adaptive_Stopping/experiments/terminator/build_labels.py",
    "Adaptive_Stopping/experiments/terminator/train_probe.py",
    "Adaptive_Stopping/experiments/terminator/evaluate.py",
    "Adaptive_Stopping/experiments/terminator/latency.py",
]

HEADER = '''import os, sys
FILES = {files}
SRC = "/kaggle/tmp/src"
for rel, text in FILES.items():
    path = os.path.join(SRC, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
os.environ.update({env})
sys.path.insert(0, SRC)
os.chdir(SRC)
'''


def kaggle_user() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="stage driver file in this folder")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for the kernel")
    ap.add_argument("--kernel-sources", nargs="*", default=[],
                    help="slugs of earlier kernels whose output is mounted as input")
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="slugs of Kaggle datasets mounted as input (e.g. the traces)")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    files = {}
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            files[rel] = p.read_text(encoding="utf-8")
    env = dict(kv.split("=", 1) for kv in args.env)
    script = HEADER.format(files=repr(files), env=repr(env))
    script += "\n" + (HERE / args.stage).read_text(encoding="utf-8")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(script, encoding="utf-8")

    user = kaggle_user()
    meta = {
        "id": f"{user}/{args.slug}",
        "title": args.slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_internet": True,
        "dataset_sources": [s if "/" in s else f"{user}/{s}" for s in args.datasets],
        "competition_sources": [],
        "kernel_sources": [s if "/" in s else f"{user}/{s}" for s in args.kernel_sources],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out / 'kernel.py'} ({len(script)} chars, {len(files)} files)")


if __name__ == "__main__":
    main()
