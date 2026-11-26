"""Ablation: ST-ES with a minimal step size k (proceedings-version Algorithm 1, |s_i| > k).

For each k, the cached traces are re-segmented with short steps merged
(``common.step_merge``). Then, exactly as in the main experiment, the grid is
replayed on the calibration split, the 85/90/95% configurations are selected,
and they are evaluated on the held-out split. ``k = none`` is plain splitting at
every "\\n\\n" and reproduces the main results. Forced answers are shared with
the main replay through the per-trace-file cache; only new cut points need the GPU.

    python -m Adaptive_Stopping.experiments.st_es.ablation_min_step --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from Adaptive_Stopping.common.step_merge import segment_with_min_size
from Adaptive_Stopping.experiments.st_es.generate import cache_path
from Adaptive_Stopping.experiments.st_es.io_utils import DEFAULT_RUN_DIR, limit_gpu_memory, read_jsonl, write_json
from Adaptive_Stopping.experiments.st_es.replay_lib import ForcedAnswerCache, aggregate, cut_points, prepare, trace_dir
from Adaptive_Stopping.methods.step_tagging.calibration import PAPER_TARGETS, Config, build_grid, select_configs
from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Backbones.registry import get_backbone_entry

REPO = Path(__file__).resolve().parents[3]


def parse_k(s: str) -> int | None:
    return None if s.lower() in ("none", "-") else int(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ks", nargs="+", default=["none", "8", "16", "32", "64"])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed, cap, prompt, exit_id = cfg["seed"], cfg["max_new_tokens"], cfg["prompt_id"], cfg["exit_prompt_id"]
    tr = {s: cache_path(args.run_dir, cfg[s]["dataset_id"], cfg[s]["num_samples"], cfg[s]["offset"], seed, cap, prompt)
          for s in ("calibration", "evaluation")}
    recs = {s: list(read_jsonl(tr[s])) for s in tr}
    bb = HFCausalLMBackbone(get_backbone_entry(cfg["backbone_id"]))
    tok = bb.load_tokenizer()
    tagger = KeywordStepTagger()
    caches = {s: ForcedAnswerCache(trace_dir(tr[s], args.run_dir) / f"forced_answers.{exit_id}.jsonl") for s in tr}

    def run(split, k, configs):
        prepared = prepare(recs[split], tok, tagger, steps_fn=segment_with_min_size(k))
        cuts = {p["rec"]["uid"]: cut_points(p, configs) for p in prepared}
        todo = caches[split].missing(prepared, cuts)
        if todo:
            if bb.model is None:
                bb.load(device=args.device)
                if args.device.startswith("cuda"):
                    limit_gpu_memory(0.85)
            print(f"k={k} {split}: {sum(map(len, todo.values()))} new forced answers over {len(todo)} traces")
            caches[split].fill(prepared, todo, bb, exit_id)
        rows, _ = aggregate(prepared, configs, cuts, caches[split])
        steps = sum(p["n_eligible"] for p in prepared) / len(prepared)
        return rows, steps

    out_rows = []
    for k_s in args.ks:
        k = parse_k(k_s)
        cal_rows, cal_steps = run("calibration", k, build_grid())
        std_cal = cal_rows[0]
        picks = {kind: select_configs(cal_rows, std_cal["accuracy"], kind) for kind in ("st_es", "step_count")}
        chosen = []
        for kind in picks:
            for t, r in picks[kind].items():
                if r is not None:
                    c = Config(kind=r["kind"], tag=r["tag"], delta=r["delta"], k=r["k"])
                    if c not in chosen:
                        chosen.append(c)
        test_rows, test_steps = run("evaluation", k, chosen)
        by_key = {r["key"]: r for r in test_rows}
        std_test = test_rows[0]
        out_rows.append({"min_step_tokens": k_s, "kind": "standard", "target": "", "config": "standard",
                         "cal_accuracy": std_cal["accuracy"], "cal_mean_tokens": std_cal["mean_tokens"],
                         "test_accuracy": std_test["accuracy"], "test_mean_tokens": std_test["mean_tokens"],
                         "test_saved_pct": 0.0, "test_stop_rate": 0.0,
                         "mean_reasoning_steps_cal": cal_steps, "mean_reasoning_steps_test": test_steps})
        for kind in ("st_es", "step_count"):
            for t in PAPER_TARGETS:
                r = picks[kind][t]
                if r is None:
                    out_rows.append({"min_step_tokens": k_s, "kind": kind, "target": t, "config": "none reached"})
                    continue
                tr_row = by_key[r["key"]]
                out_rows.append({"min_step_tokens": k_s, "kind": kind, "target": t, "config": r["key"],
                                 "cal_accuracy": r["accuracy"], "cal_mean_tokens": r["mean_tokens"],
                                 "test_accuracy": tr_row["accuracy"], "test_mean_tokens": tr_row["mean_tokens"],
                                 "test_saved_pct": tr_row["saved_pct"], "test_stop_rate": tr_row["stop_rate"],
                                 "mean_reasoning_steps_cal": cal_steps, "mean_reasoning_steps_test": test_steps})
        print(f"k={k_s}: steps/trace cal {cal_steps:.1f} test {test_steps:.1f}; "
              + "; ".join(f"{r['kind']}@{r['target']} {r['config']} test acc {r.get('test_accuracy', float('nan')):.3f} "
                          f"tok {r.get('test_mean_tokens', float('nan')):.0f}"
                          for r in out_rows if r["min_step_tokens"] == k_s and r["kind"] == "st_es"))

    out = REPO / "Adaptive_Stopping" / "experiments" / "st_es" / "results" / cfg["evaluation"]["dataset_id"]
    out.mkdir(parents=True, exist_ok=True)
    fields = ["min_step_tokens", "kind", "target", "config", "cal_accuracy", "cal_mean_tokens", "test_accuracy",
              "test_mean_tokens", "test_saved_pct", "test_stop_rate", "mean_reasoning_steps_cal",
              "mean_reasoning_steps_test"]
    with open(out / "ablation_min_step.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    write_json(out / "ablation_min_step.meta.json", {
        "ks": args.ks, "definition": "|s_i| > k generated tokens, delimiter included; k=none = split at every \\n\\n",
        "protocol": "per k: calibrate on the calibration split (full grid), evaluate the selected configs on the test split",
        "tagger": tagger.name, "traces": {s: tr[s].name for s in tr}, "config": str(args.config.as_posix()),
    })
    print(json.dumps({"wrote": str(out / "ablation_min_step.csv")}))


if __name__ == "__main__":
    main()
