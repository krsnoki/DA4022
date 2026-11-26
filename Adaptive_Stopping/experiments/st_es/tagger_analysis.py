"""Error analysis of the rule-based step tagger, and v1 vs v2.

1. Held-out agreement with GPT-4o-mini on the authors' 99 labelled steps
   (``validate_tagger`` data). Gives the confusion matrices and the main
   failure modes with short examples.
2. Tag distributions of v1 and v2 on our calibration traces.
3. Effect on ST-ES. Per config file, the grid is replayed on the calibration
   split with each tagger, the 85/90/95% configs are selected, and they are
   evaluated on the test split. Forced answers are shared through the
   per-trace-file cache, so only new cut points need the GPU.

Outputs go to ``Adaptive_Stopping/experiments/st_es/results/tagger_analysis/``.

    python -m Adaptive_Stopping.experiments.st_es.tagger_analysis --configs Adaptive_Stopping/configs/st_es/gsm8k.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from Adaptive_Stopping.experiments.st_es.generate import cache_path
from Adaptive_Stopping.experiments.st_es.io_utils import DEFAULT_RUN_DIR, limit_gpu_memory, read_jsonl, write_json
from Adaptive_Stopping.experiments.st_es.replay_lib import ForcedAnswerCache, aggregate, cut_points, prepare, trace_dir
from Adaptive_Stopping.experiments.st_es.validate_tagger import binary_scores, fetch
from Adaptive_Stopping.methods.step_tagging.calibration import PAPER_TARGETS, Config, build_grid, select_configs
from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Adaptive_Stopping.methods.step_tagging.tagger_v2 import KeywordStepTaggerV2
from Adaptive_Stopping.methods.step_tagging.taxonomy import ID_TO_NUMBER, TAG_IDS, to_tag_id
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Backbones.registry import get_backbone_entry

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "Adaptive_Stopping" / "experiments" / "st_es" / "results" / "tagger_analysis"
TAGGERS = {"v1": KeywordStepTagger(), "v2": KeywordStepTaggerV2()}
# sequential single-hue ramp (light -> dark) for counts
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def validation(run_dir: Path) -> tuple[dict, list[dict]]:
    items = {}
    for p in fetch(run_dir / "external"):
        for row in json.loads(p.read_text(encoding="utf-8")):
            items[(row["sample_id"], row["step_id"])] = row
    rows = list(items.values())
    gold = [to_tag_id(r["category"]) for r in rows]
    res = {}
    for name, tg in TAGGERS.items():
        pred = tg.tag_all(r["step"] for r in rows)
        per_tag = {t: binary_scores(gold, pred, t) for t in TAG_IDS if t != "other"}
        res[name] = {"tagger": tg.name, "n_steps": len(rows),
                     "exact_agreement": sum(g == p for g, p in zip(gold, pred)) / len(rows),
                     "mean_binary_macro_f1": sum(s["binary_macro_f1"] for s in per_tag.values()) / len(per_tag),
                     "per_tag": per_tag, "pred": pred}
    return res, [dict(r, gold=g) for r, g in zip(rows, gold)]


def plot_confusion(gold, pred, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = [t for t in TAG_IDS]
    idx = {t: i for i, t in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150, facecolor="#fcfcfb")
    vmax = max(max(r) for r in m)
    ax.imshow(m, cmap=ListedColormap(RAMP), vmin=0, vmax=vmax)
    names = [f"{ID_TO_NUMBER[t]}. {t}" for t in labels]
    ax.set_xticks(range(len(labels)), names, rotation=60, ha="right", fontsize=8, color="#52514e")
    ax.set_yticks(range(len(labels)), names, fontsize=8, color="#52514e")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i][j]:
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=8,
                        color="#ffffff" if m[i][j] > vmax * 0.55 else "#0b0b0b")
    ax.set_xlabel("rule-based tagger", color="#52514e")
    ax.set_ylabel("GPT-4o-mini label (authors' release)", color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def failure_examples(rows, pred, top=8, per_pair=2) -> str:
    pairs = Counter((r["gold"], p) for r, p in zip(rows, pred) if r["gold"] != p)
    lines = ["# v1 tagger: most frequent disagreements with GPT-4o-mini", "",
             "Steps: DeepSeek-R1-Distill-Qwen-14B on MATH500, labels by GPT-4o-mini, released by the authors "
             "(yannisbelkhiter/ReasonType_Human_validation, MIT). Excerpts are truncated to 200 characters.", ""]
    for (g, p), c in pairs.most_common(top):
        lines += [f"## GPT-4o-mini `{g}` → v1 `{p}` ({c} steps)", ""]
        for r, q in [(r, q) for r, q in zip(rows, pred) if r["gold"] == g and q == p][:per_pair]:
            excerpt = " ".join(r["step"].split())[:200].replace("|", "\\|")
            lines += [f"- {excerpt}"]
        lines.append("")
    return "\n".join(lines)


def calibration_effect(config: Path, run_dir: Path, device: str) -> list[dict]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    seed, cap, prompt, exit_id = cfg["seed"], cfg["max_new_tokens"], cfg["prompt_id"], cfg["exit_prompt_id"]
    tr = {s: cache_path(run_dir, cfg[s]["dataset_id"], cfg[s]["num_samples"], cfg[s]["offset"], seed, cap, prompt)
          for s in ("calibration", "evaluation")}
    recs = {s: list(read_jsonl(tr[s])) for s in tr}
    bb = HFCausalLMBackbone(get_backbone_entry(cfg["backbone_id"]))
    tok = bb.load_tokenizer()
    caches = {s: ForcedAnswerCache(trace_dir(tr[s], run_dir) / f"forced_answers.{exit_id}.jsonl") for s in tr}
    out = []
    for name, tg in TAGGERS.items():
        def run(split, configs):
            prepared = prepare(recs[split], tok, tg)
            cuts = {p["rec"]["uid"]: cut_points(p, configs) for p in prepared}
            todo = caches[split].missing(prepared, cuts)
            if todo:
                if bb.model is None:
                    bb.load(device=device)
                    if device.startswith("cuda"):
                        limit_gpu_memory(0.85)
                print(f"{name} {split}: {sum(map(len, todo.values()))} new forced answers")
                caches[split].fill(prepared, todo, bb, exit_id)
            rows, _ = aggregate(prepared, configs, cuts, caches[split])
            hist = Counter(t for p in prepared for s, t in zip(p["steps"], p["tags"]) if s.closed and s.in_thinking)
            return rows, hist

        cal_rows, hist = run("calibration", build_grid())
        picks = select_configs(cal_rows, cal_rows[0]["accuracy"], "st_es")
        chosen = [Config("st_es", tag=r["tag"], delta=r["delta"]) for r in picks.values() if r]
        test_rows, _ = run("evaluation", list(dict.fromkeys(chosen)))
        by_key = {r["key"]: r for r in test_rows}
        total = sum(hist.values())
        for t in PAPER_TARGETS:
            r = picks[t]
            row = {"dataset": cfg["evaluation"]["dataset_id"], "tagger": name, "target": t,
                   "other_share_calibration": round(hist.get("other", 0) / total, 4)}
            if r is None:
                out.append({**row, "config": "none reached"})
                continue
            te = by_key[r["key"]]
            out.append({**row, "config": r["key"], "cal_accuracy": round(r["accuracy"], 4),
                        "cal_mean_tokens": round(r["mean_tokens"], 1), "test_accuracy": round(te["accuracy"], 4),
                        "test_mean_tokens": round(te["mean_tokens"], 1), "test_saved_pct": round(te["saved_pct"], 2),
                        "test_standard_accuracy": round(test_rows[0]["accuracy"], 4)})
        out.append({"dataset": cfg["evaluation"]["dataset_id"], "tagger": name, "target": "tag_histogram",
                    "config": json.dumps(dict(hist.most_common()))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", type=Path, nargs="*", default=[])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    val, rows = validation(args.run_dir)
    gold = [r["gold"] for r in rows]
    for name in TAGGERS:
        plot_confusion(gold, val[name]["pred"], f"Tagger {name} vs GPT-4o-mini on {len(rows)} held-out steps "
                       f"(exact agreement {val[name]['exact_agreement']:.1%})", OUT / f"confusion_{name}.png")
    (OUT / "failure_examples_v1.md").write_text(failure_examples(rows, val["v1"]["pred"]), encoding="utf-8")
    write_json(OUT / "validation_v1_vs_v2.json", {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                                                  for k, v in val.items()})
    for name in TAGGERS:
        print(f"{name}: exact {val[name]['exact_agreement']:.3f}  mean binary macro-F1 "
              f"{val[name]['mean_binary_macro_f1']:.3f}")

    eff = []
    for c in args.configs:
        eff += calibration_effect(c, args.run_dir, args.device)
    if eff:
        fields = ["dataset", "tagger", "target", "config", "other_share_calibration", "cal_accuracy", "cal_mean_tokens",
                  "test_accuracy", "test_mean_tokens", "test_saved_pct", "test_standard_accuracy"]
        with open(OUT / "calibration_effect_v1_vs_v2.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(eff)
        for r in eff:
            if r["target"] != "tag_histogram":
                print(r)


if __name__ == "__main__":
    main()
