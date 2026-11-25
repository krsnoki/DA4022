"""Build token labels from generated traces (step B).

For each trace: the final answer is the last \\boxed{} in the answer section;
its first appearance in the thinking text gives the label position (see
methods/learned_exit/terminator/labels.py). Traces without a boxed answer or
without a match are marked as dropped.

Writes one small JSON line per trace (no token arrays) and a summary JSON.
Train traces are split into train/val by a fixed seed.

Example:
    python Adaptive_Stopping/experiments/terminator/build_labels.py \
        --traces traces_train.jsonl traces_math500.jsonl --out labels.jsonl \
        --summary labels_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Dataloaders.evaluation.math_reasoning import extract_boxed  # noqa: E402
from Adaptive_Stopping.methods.learned_exit.terminator.labels import (  # noqa: E402
    answer_token_position,
    token_char_starts,
)

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def label_row(tok, r):
    ans = extract_boxed(r["answer_text"])
    out = {
        "uid": r["uid"],
        "split": r["split"],
        "thinking_tokens": len(r["think_ids"]),
        "final_answer": ans,
        "label_pos": None,
        "drop_reason": None,
    }
    if ans is None:
        out["drop_reason"] = "no_boxed_answer"
        return out
    text, starts = token_char_starts(tok, r["think_ids"])
    pos = answer_token_position(text, starts, ans, r.get("problem"))
    if pos is None:
        out["drop_reason"] = "answer_not_in_thinking"
    out["label_pos"] = pos
    return out


def summarize(rows):
    n = len(rows)
    kept = [r for r in rows if r["label_pos"] is not None]
    rel = [r["label_pos"] / max(1, r["thinking_tokens"]) for r in kept]
    return {
        "n": n,
        "kept": len(kept),
        "drop_rate": 1 - len(kept) / n if n else None,
        "drop_no_boxed": sum(r["drop_reason"] == "no_boxed_answer" for r in rows),
        "drop_not_found": sum(r["drop_reason"] == "answer_not_in_thinking" for r in rows),
        "median_rel_label_pos": statistics.median(rel) if rel else None,
        "mean_rel_label_pos": statistics.mean(rel) if rel else None,
        "frac_label_pos_lt_10pct": sum(x < 0.1 for x in rel) / len(rel) if rel else None,
        "label1_token_frac": (
            sum(r["thinking_tokens"] - r["label_pos"] for r in kept)
            / max(1, sum(r["thinking_tokens"] for r in kept))
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rows = []
    for fn in args.traces:
        with open(fn) as f:
            for line in f:
                if line.strip():
                    rows.append(label_row(tok, json.loads(line)))

    train_uids = sorted(r["uid"] for r in rows if r["split"] == "train")
    rng = random.Random(args.seed)
    val = set(rng.sample(train_uids, int(round(args.val_frac * len(train_uids))))) if train_uids else set()
    for r in rows:
        if r["split"] == "train":
            r["fold"] = "val" if r["uid"] in val else "train"
        else:
            r["fold"] = "test"

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {fold: summarize([r for r in rows if r["fold"] == fold])
               for fold in ("train", "val", "test") if any(r["fold"] == fold for r in rows)}
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
