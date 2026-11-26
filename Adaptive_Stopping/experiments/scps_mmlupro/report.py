"""Render the markdown result tables in the SCPS paper card from ``summary.json``.

    python -m Adaptive_Stopping.experiments.scps_mmlupro.report <out-dir> [--latency-subset <dir> --subset-n 32]

Printing tables from the saved summary (instead of copying numbers by
hand) keeps the documentation consistent with the run outputs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ORDER = ["scps", "self_consistency", "tree_prefix", "semantic_pruning", "option_topk"]
NAMES = {"scps": "SCPS", "self_consistency": "Self-consistency (16 samples)", "tree_prefix": "Tree-prefix",
         "semantic_pruning": "Semantic pruning (0.75)", "option_topk": "Option top-8 (no sampling)"}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci(d):
    return f"{100 * d['delta']:+.1f} pp [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] ({d['wins']} wins / {d['losses']} losses)"


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--latency-subset", default=None, help="dir with a rerun of SCPS on the first N questions")
    ap.add_argument("--subset-n", type=int, default=32)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m = s["methods"]
    lines = ["| Method | Pass@16 (gold in portfolio) | Majority-vote accuracy | Distinct answers | Rollouts | Total tokens (decode + aux) | Dup. token frac. | Null completions | Latency median / p95 per question | Peak CUDA memory | Budget violations |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ORDER:
        if k not in m:
            continue
        v = m[k]
        mem = ((s["run_meta"].get(k) or {}).get("peak_memory") or {}).get("cuda_peak_allocated_mb")
        dup = "n/a" if v["mean_dup_token_frac"] is None else f"{100 * v['mean_dup_token_frac']:.1f}%"
        lines.append(
            f"| {NAMES[k]} | {pct(v['pass_at_n'])} ({v['pass_count']}/{v['n']}) | {pct(v['majority_accuracy'])} | "
            f"{v['mean_unique_answers']:.2f} | {v['mean_rollouts']:.2f} | {v['mean_total_tokens']:.1f} | {dup} | "
            f"{v['null_completions']} | {v['latency_median_ms'] / 1000:.1f} s / {v['latency_p95_ms'] / 1000:.1f} s | "
            f"{'n/a' if mem is None else f'{mem:,.0f} MB'} | {v['budget_violations']} |")
    print("\n".join(lines))
    print("\nPaired bootstrap, SCPS minus baseline (10,000 resamples of questions, 95% percentile interval):\n")
    print("| Baseline | Pass@16 delta | Majority-vote accuracy delta |\n|---|---|---|")
    for b, d in s.get("paired_vs_scps", {}).items():
        print(f"| {NAMES[b]} | {ci(d['pass'])} | {ci(d['majority'])} |")
    c = s.get("option_topk_control")
    if c:
        print(f"\nOption top-K control (n = {c['n']}): per-question K matched to SCPS's distinct-answer count gives "
              f"{pct(c['matched_k_pass'])}; SCPS minus matched-K: {ci(c['scps_vs_matched_k'])}.\n")
        print("| K | " + " | ".join(str(k) for k in c["gold_in_topk_curve"]) + " |")
        print("|---|" + "---:|" * len(c["gold_in_topk_curve"]))
        print("| Gold in top-K | " + " | ".join(pct(v) for v in c["gold_in_topk_curve"].values()) + " |")
    cat = s.get("per_category_pass_counts")
    if cat:
        cols = [k for k in ORDER if any(f"{k}_pass" in d for d in cat.values()) or k == "scps"]
        print("\n| Category | n | " + " | ".join(NAMES[k] for k in cols) + " |")
        print("|---|---:|" + "---:|" * len(cols))
        for name, d in cat.items():
            print(f"| {name} | {d['n']} | " + " | ".join(str(d.get(f"{k}_pass", "")) for k in cols) + " |")
    if a.latency_subset:
        n = a.subset_n
        print(f"\nLatency on the first {n} questions (seconds, median / p95):\n")
        print("| Method | Session | Median | p95 |\n|---|---|---:|---:|")
        rows = [("scps", out / "per_question_scps.jsonl", "main run (session 1)"),
                ("scps", Path(a.latency_subset) / "per_question_scps.jsonl", "rerun (session 3)")]
        rows += [(k, out / f"per_question_{k}.jsonl", "session 2") for k in ORDER[1:]]
        for k, p, sess in rows:
            w = sorted(r["wall_time_s"] for r in load(p)[:n])
            p95 = w[min(len(w) - 1, int(round(0.95 * (len(w) - 1))))]
            print(f"| {NAMES[k]} | {sess} | {statistics.median(w):.1f} | {p95:.1f} |")


if __name__ == "__main__":
    main()
