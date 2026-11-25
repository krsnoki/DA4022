"""Run the frozen 30-example synthetic length-stratified evaluation on CUDA.

Run only after reviewing the new profile's pilot. Three data seeds are three
disjoint example groups, not repeated decoding trials on the same examples.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .run import source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--out", default="artifacts/agentkv-evaluation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verification = json.loads(Path(args.verification).read_text())
    if not verification.get("passed") or verification.get("smoke") or verification.get("source_sha256") != source_hash():
        raise SystemExit("A passing pretrained integration gate for this source is required.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in (101, 202, 303):
        command = [sys.executable, "-u", "-m", "Attention.experiments.agentkv.run",
                   "--config", f"Attention/configs/agentkv/evaluation_{seed}.json",
                   "--out", str(out / f"seed_{seed}")]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)
    print(f"Completed three strata; retain their separate summaries under {out}.")


if __name__ == "__main__":
    main()
