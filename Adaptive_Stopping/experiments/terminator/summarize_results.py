"""Collect the outputs of the train, eval and latency kernels into one
results folder.

Writes in --out-dir:
  summary.csv, summary_val.csv, threshold_selection.json   (from evaluate.py)
  latency_summary.csv     no_stop vs TERMINATOR on the problems both runs finished
  train_summary.json      label statistics and the probe's validation curve
  per_problem.jsonl       test rows of evaluate.py (gzipped if over 1 MB)
  latency_rows.jsonl      per-problem latency rows

Example:
    python Adaptive_Stopping/experiments/terminator/summarize_results.py \
        --train out/train --eval out/eval/eval --latency out/latency \
        --out-dir Adaptive_Stopping/experiments/terminator/results
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import statistics
from pathlib import Path

LIMIT = 1_000_000


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def copy_small(src, dst):
    if src.stat().st_size <= LIMIT:
        shutil.copy(src, dst)
        return dst.name
    gz = dst.with_name(dst.name + ".gz")
    with open(src, "rb") as f, gzip.open(gz, "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    return gz.name


def latency_table(lat_dir):
    runs = {}
    for p in sorted(Path(lat_dir).glob("latency_*.jsonl")):
        rows = load_jsonl(p)
        if rows:
            runs[rows[0]["method"]] = {r["uid"]: r for r in rows}
    if len(runs) < 2:
        return [], []
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    table = []
    for method, by_uid in runs.items():
        rs = [by_uid[u] for u in common]
        lat = [r["latency_s"] for r in rs]
        think = sum(r["thinking_tokens"] for r in rs)
        toks = sum(r["thinking_tokens"] + r["answer_tokens"] for r in rs)
        table.append({
            "method": method,
            "threshold": rs[0]["threshold"],
            "n": len(rs),
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            "mean_thinking_tokens": round(think / len(rs), 1),
            "stop_rate": round(sum(r["stopped_early"] for r in rs) / len(rs), 4),
            "latency_mean_s": round(statistics.mean(lat), 2),
            "latency_median_s": round(statistics.median(lat), 2),
            "latency_p95_s": round(pct(lat, 0.95), 2),
            "tokens_per_s": round(toks / sum(lat), 2),
            "probe_ms_per_token": (round(1000 * sum(r["probe_s"] for r in rs) / think, 3)
                                   if method == "terminator" and think else None),
            "peak_mem_mb_max": round(max(r["peak_mem_mb"] for r in rs), 1),
        })
    rows = [r for by_uid in runs.values() for u, r in sorted(by_uid.items())]
    return table, rows


def token_metrics(labels, probs, threshold=0.5):
    """Token-level metrics of the probe on traces with a label (same rule as
    train_probe.evaluate: label 1 from label_pos on, prediction prob > threshold)."""
    tp = fp = fn = tn = n_traces = 0
    for uid, ps in probs.items():
        pos = labels.get(uid)
        if pos is None:
            continue
        n_traces += 1
        pos = max(0, min(pos, len(ps)))
        for t, p in enumerate(ps):
            pred, y = p > threshold, t >= pos
            tp += pred and y
            fp += pred and not y
            fn += (not pred) and y
            tn += (not pred) and not y

    def f1(a, b, c):
        return 2 * a / max(1, 2 * a + b + c)

    n = tp + fp + fn + tn
    return {"n_traces": n_traces, "n_tokens": n, "threshold": threshold,
            "macro_f1": round(0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp)), 4),
            "f1_label1": round(f1(tp, fp, fn), 4), "f1_label0": round(f1(tn, fn, fp), 4),
            "accuracy": round((tp + tn) / max(1, n), 4)}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="output folder of the train kernel")
    ap.add_argument("--eval", required=True, help="eval folder with summary.csv")
    ap.add_argument("--latency", default=None, help="output folder of the latency kernel")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = Path(args.eval)
    for name in ("summary.csv", "summary_val.csv", "threshold_selection.json"):
        shutil.copy(ev / name, out / name)
    written = [copy_small(ev / "per_problem.jsonl", out / "per_problem.jsonl")]

    tr = Path(args.train)
    log = json.loads((tr / "probe" / "train_log.json").read_text())
    train = {
        "labels": json.loads((tr / "labels_summary.json").read_text()),
        "probe_params": log["n_params"],
        "n_train_traces": log["n_train"],
        "n_val_traces": log["n_val"],
        "class_weights": {"w0": log["w0"], "w1": log["w1"]},
        "total_steps": log.get("total_steps"),
        "train_time_s": log.get("train_time_s"),
        "skipped_optimizer_steps": log.get("skipped_optimizer_steps"),
        "nonfinite_loss_skipped": len(log.get("nonfinite_loss_skipped", [])),
        "oom_skipped": len(log.get("oom_skipped", [])),
        "base_hidden_nonfinite": log.get("base_hidden_nonfinite"),
        "val_evals": log["evals"],
        "best_val_macro_f1": log.get("best_val_macro_f1"),
    }
    test_probs = tr / "probe" / "probs_test.jsonl"
    if test_probs.exists():
        labels = {r["uid"]: r["label_pos"] for r in load_jsonl(tr / "labels.jsonl")
                  if r.get("fold") == "test"}
        probs = {r["uid"]: r["probs"] for r in load_jsonl(test_probs)}
        train["test_token_metrics"] = token_metrics(labels, probs)
        print("test token metrics", json.dumps(train["test_token_metrics"]))
    (out / "train_summary.json").write_text(json.dumps(train, indent=2))

    if args.latency:
        table, rows = latency_table(args.latency)
        if table:
            write_csv(out / "latency_summary.csv", table)
            with open(out / "latency_rows.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            written.append("latency_rows.jsonl")
            for t in table:
                print(json.dumps(t))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
