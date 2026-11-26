"""Run the full ST-ES pipeline for one config (each stage resumes from its cached outputs).

Stages: (1) cached generation on the calibration and evaluation subsets,
(2) calibration replay over the full grid + config selection, (3) replay of the
selected configs on the evaluation subset, (4) live latency/memory benchmark.

    python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from Adaptive_Stopping.experiments.st_es.generate import cache_path
from Adaptive_Stopping.experiments.st_es.io_utils import DEFAULT_RUN_DIR

STAGES = ["generate", "calibrate", "evaluate", "live"]


def run(mod: str, *args) -> None:
    cmd = [sys.executable, "-u", "-m", f"Adaptive_Stopping.experiments.st_es.{mod}", *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def check_config(cfg: dict) -> None:
    """The config documents the grid; refuse values the code does not implement (no silent mismatch)."""
    from Adaptive_Stopping.common.forced_answer import ANSWER_BUDGET
    from Adaptive_Stopping.common.steps import STEP_DELIMITER
    from Adaptive_Stopping.methods.step_tagging.calibration import PAPER_DELTAS, PAPER_TARGETS
    from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger

    lo, hi = cfg["grid"]["deltas"]
    assert tuple(range(lo, hi + 1)) == PAPER_DELTAS, "grid.deltas differs from calibration.PAPER_DELTAS"
    assert tuple(cfg["targets"]) == PAPER_TARGETS, "targets differ from calibration.PAPER_TARGETS"
    assert cfg["grid"]["step_counts"] == "default" and cfg["grid"]["tags"] == "all_except_other"
    assert cfg["tagger"] == KeywordStepTagger.name
    assert cfg["answer_budget"] == ANSWER_BUDGET and cfg["step_delimiter"] == STEP_DELIMITER
    assert cfg["calibration"]["dataset_id"] == cfg["calibration_dataset"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    check_config(cfg)
    seed, cap, prompt = cfg["seed"], cfg["max_new_tokens"], cfg["prompt_id"]
    traces = {}
    for split in ("calibration", "evaluation"):
        s = cfg[split]
        traces[split] = cache_path(args.run_dir, s["dataset_id"], s["num_samples"], s["offset"], seed, cap, prompt)
        if "generate" in args.stages:
            run("generate", "--dataset-id", s["dataset_id"], "--num-samples", s["num_samples"], "--offset",
                s["offset"], "--seed", seed, "--max-new-tokens", cap, "--prompt-id", prompt,
                "--run-dir", args.run_dir)
    cal_dir = args.run_dir / "replay" / traces["calibration"].parent.name / traces["calibration"].stem / "grid"
    if "calibrate" in args.stages:
        run("replay_sweep", "--traces", traces["calibration"], "--select-targets",
            "--exit-prompt-id", cfg["exit_prompt_id"], "--run-dir", args.run_dir)
    if "evaluate" in args.stages:
        run("replay_sweep", "--traces", traces["evaluation"], "--selected", cal_dir / "selected.json",
            "--exit-prompt-id", cfg["exit_prompt_id"], "--run-dir", args.run_dir)
    if "live" in args.stages:
        lv = cfg["live"]
        run("live_benchmark", "--dataset-id", cfg["evaluation"]["dataset_id"], "--num-samples", lv["num_samples"],
            "--seed", seed, "--max-new-tokens", cap, "--prompt-id", prompt, "--selected", cal_dir / "selected.json",
            "--targets", *lv["targets"], "--kinds", *lv["kinds"], "--run-dir", args.run_dir)


if __name__ == "__main__":
    main()
