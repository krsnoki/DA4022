"""Offline replay of ST-ES and the step-count baseline over cached traces.

For one cached trace file (from ``generate.py``) this script

1. segments every trace into steps and tags them (``KeywordStepTagger``);
2. scores the standard (full) generation with Math-Verify;
3. computes, for every configuration, where generation would have stopped;
4. forces an answer at every distinct cut point (100-token budget, greedy),
   caching the results so later runs are cheap;
5. writes per-configuration accuracy / mean generated tokens / stop rate.

With ``--select-targets`` (calibration split) it also writes the configurations
selected at 85/90/95% of the standard accuracy; with ``--selected`` (test split)
it evaluates exactly those configurations.

Token counts here come from replay. Runtimes derived from them are *estimates*;
measured latency comes from ``live_benchmark.py``.

Example::

    python -m Adaptive_Stopping.experiments.st_es.replay_sweep \\
        --traces <run_dir>/traces/text.gsm8k.train/math_cot_boxed.v1_n200_o0_s42_cap4096.jsonl \\
        --select-targets
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from Adaptive_Stopping.common.forced_answer import EXIT_PROMPTS
from Adaptive_Stopping.experiments.st_es.io_utils import (
    DEFAULT_RUN_DIR, environment_metadata, limit_gpu_memory, read_jsonl, sha256_file, write_json,
)
from Adaptive_Stopping.experiments.st_es.replay_lib import (  # noqa: F401  (prepare/cut_points re-exported)
    ForcedAnswerCache, aggregate, cut_points, prepare,
)
from Adaptive_Stopping.methods.step_tagging.calibration import (
    PAPER_TARGETS, Config, build_grid, select_configs,
)
from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Backbones.registry import get_backbone_entry
from Performance_Metrics.quality.math_answer import SCORER_ID


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--selected", type=Path, default=None, help="selected.json from a calibration run")
    ap.add_argument("--select-targets", action="store_true", help="write selected.json (calibration split)")
    ap.add_argument("--exit-prompt-id", default="exit_boxed.v1", choices=sorted(EXIT_PROMPTS))
    ap.add_argument("--backbone-id", default="text.deepseek-r1-distill-qwen-1.5b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-examples", type=int, default=None, help="debug: only the first N traces")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = ap.parse_args(argv)

    # <run_dir>/replay/<dataset>/<traces stem>/{grid|selected}; the forced-answer cache sits one level up,
    # so it is shared by the grid and selected runs of the same trace file (and only of that file).
    mode = "selected" if args.selected else "grid"
    out_dir = args.out_dir or (args.run_dir / "replay" / args.traces.parent.name / args.traces.stem / mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(args.traces))[: args.max_examples]
    bb = HFCausalLMBackbone(get_backbone_entry(args.backbone_id))
    tok = bb.load_tokenizer()
    tagger = KeywordStepTagger()

    t0 = time.time()
    prepared = prepare(records, tok, tagger)
    print(f"prepared {len(prepared)} traces in {time.time() - t0:.1f}s")

    if args.selected:
        sel = json.loads(args.selected.read_text(encoding="utf-8"))
        configs = [Config.from_dict(c) for c in sel["configs"]]
    else:
        configs = build_grid()
    cuts = {p["rec"]["uid"]: cut_points(p, configs) for p in prepared}

    # ---------------- forced answers (cached per trace file) ----------------
    forced = ForcedAnswerCache(out_dir.parent / f"forced_answers.{args.exit_prompt_id}.jsonl")
    todo = forced.missing(prepared, cuts)
    n_todo = sum(len(v) for v in todo.values())
    print(f"forced answers: {len(forced)} cached, {n_todo} to compute over {len(todo)} traces")
    forced_wall = 0.0
    if n_todo:
        bb.load(device=args.device)
        if args.device.startswith("cuda"):
            limit_gpu_memory(0.85)
        forced_wall = forced.fill(prepared, todo, bb, args.exit_prompt_id)

    # ---------------- aggregate ----------------
    rows, per_example = aggregate(prepared, configs, cuts, forced)
    n, std_acc, std_tokens = rows[0]["n"], rows[0]["accuracy"], rows[0]["mean_tokens"]

    fields = list(rows[0].keys())
    with open(out_dir / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "per_example.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_example[0].keys()) if per_example else ["key"])
        w.writeheader()
        w.writerows(per_example)

    tag_hist = defaultdict(int)
    for p in prepared:
        for s, t in zip(p["steps"], p["tags"]):
            if s.closed and s.in_thinking:
                tag_hist[t] += 1
    summary = {
        "traces": str(args.traces.name), "traces_sha256": sha256_file(args.traces), "n": n,
        "standard_accuracy": std_acc, "standard_mean_tokens": std_tokens,
        "finish_reasons": {fr: sum(p["rec"]["finish_reason"] == fr for p in prepared) for fr in ("eos", "length")},
        "mean_reasoning_steps": sum(p["n_eligible"] for p in prepared) / n,
        "tag_histogram": dict(sorted(tag_hist.items(), key=lambda kv: -kv[1])),
        "tagger": tagger.name, "scorer": SCORER_ID, "exit_prompt_id": args.exit_prompt_id,
        "exit_prompt": EXIT_PROMPTS[args.exit_prompt_id], "answer_budget": 100, "answer_decoding": "greedy",
        "n_configs": len(configs), "forced_answers_cached": len(forced), "forced_answer_wall_s": forced_wall,
        "environment": environment_metadata(),
    }
    if args.select_targets:
        sel = {"targets": list(PAPER_TARGETS), "calibration_traces": str(args.traces.name),
               "standard_accuracy": std_acc, "selected": {}, "configs": []}
        for kind in ("st_es", "step_count"):
            picks = select_configs(rows, std_acc, kind)
            sel["selected"][kind] = {str(t): r for t, r in picks.items()}
            for r in picks.values():
                if r is not None:
                    cfg = Config(kind=r["kind"], tag=r["tag"], delta=r["delta"], k=r["k"]).to_dict()
                    if cfg not in sel["configs"]:
                        sel["configs"].append(cfg)
        write_json(out_dir / "selected.json", sel)
        summary["selected"] = sel["selected"]
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("n", "standard_accuracy", "standard_mean_tokens", "finish_reasons",
                                              "mean_reasoning_steps")}, indent=1))
    if args.select_targets:
        for kind, picks in summary["selected"].items():
            for t, r in picks.items():
                print(kind, t, None if r is None else {k: r[k] for k in ("key", "accuracy", "mean_tokens", "saved_pct")})


if __name__ == "__main__":
    main()
