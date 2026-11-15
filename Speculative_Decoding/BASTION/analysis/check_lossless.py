#!/usr/bin/env python
"""E3: losslessness check at T=0 — compare DFlash / BASTION output tokens with AR.

Greedy speculative decoding is lossless in exact arithmetic. In bf16 the
target's logits depend on the shape of the forward pass (1 token for AR, a
16-token block for DFlash, an N-node tree with a custom mask and the ``sdpa``
kernel for BASTION; AR/DFlash may also use a different attention kernel), so
near-ties in the argmax can flip and outputs can diverge. This script
measures how often and where that happens; it does not assume the outcome.

Comparison unit: one (model, dataset, sample, turn) whose AR record has the
same prompt hash. Reported per (model, dataset, method):
  exact-match rate, first-divergence position distribution (0-based index into
  the generated tokens; a pure length difference counts as divergence at the
  shorter length), and token agreement up to divergence = prefix length / AR
  length.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import ARIndex  # noqa: E402
from common import VARIANT_ORDER, dataset_sort_key, load_records, method_label, model_short, rec_profile_variant  # noqa: E402


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, len(shorter) on pure length mismatch, None if identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def compare(records: list[dict], ar_index: ARIndex | None = None) -> tuple[list[dict], list[dict]]:
    if ar_index is None:
        ar_index = ARIndex(records)
    per_record = []
    for rec in records:
        if rec["mode"] == "ar":
            continue
        ar = ar_index.lookup(rec)
        row = {
            "model": rec["model"], "dataset": rec["dataset"], "sample_idx": rec["sample_idx"], "turn": rec["turn"],
            "mode": rec["mode"], "fixed_budget": rec.get("fixed_budget"), "profile_variant": rec_profile_variant(rec),
            "method": method_label(rec["mode"], rec.get("fixed_budget"), rec_profile_variant(rec)),
        }
        if ar is None:
            row["status"] = "no_ar"
        elif ar.get("input_hash") != rec.get("input_hash"):
            row["status"] = "prompt_mismatch"
        else:
            a, b = ar["output_ids"], rec["output_ids"]
            div = first_divergence(a, b)
            row.update(
                status="compared",
                exact=div is None,
                first_divergence=div,
                ar_len=len(a),
                method_len=len(b),
                prefix_agreement=1.0 if div is None else (div / len(a) if a else 0.0),
                ar_hit_max=len(a) >= ar.get("max_new_tokens", 10**9),
            )
        per_record.append(row)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_record:
        groups[(row["model"], row["dataset"], row["mode"], row["fixed_budget"], row["profile_variant"])].append(row)
    summary = []
    for (model, dataset, mode, budget, variant), rows in groups.items():
        comp = [r for r in rows if r["status"] == "compared"]
        divs = [r["first_divergence"] for r in comp if not r["exact"]]
        summary.append({
            "model": model, "dataset": dataset, "mode": mode, "fixed_budget": budget, "profile_variant": variant,
            "method": method_label(mode, budget, variant),
            "n_records": len(rows),
            "n_compared": len(comp),
            "n_no_ar": sum(r["status"] == "no_ar" for r in rows),
            "n_prompt_mismatch": sum(r["status"] == "prompt_mismatch" for r in rows),
            "n_exact": sum(r["exact"] for r in comp),
            "exact_rate": (sum(r["exact"] for r in comp) / len(comp)) if comp else None,
            "n_diverged": len(divs),
            "div_pos_min": int(min(divs)) if divs else None,
            "div_pos_p25": float(np.percentile(divs, 25)) if divs else None,
            "div_pos_median": float(np.median(divs)) if divs else None,
            "div_pos_p75": float(np.percentile(divs, 75)) if divs else None,
            "div_pos_max": int(max(divs)) if divs else None,
            "mean_prefix_agreement": float(np.mean([r["prefix_agreement"] for r in comp])) if comp else None,
            "mean_prefix_agreement_diverged": float(np.mean([r["prefix_agreement"] for r in comp if not r["exact"]])) if divs else None,
            "n_length_only_diff": sum(1 for r in comp if not r["exact"] and r["first_divergence"] == min(r["ar_len"], r["method_len"])),
        })
    summary.sort(key=lambda s: (s["model"], s["mode"], -1 if s["fixed_budget"] is None else s["fixed_budget"],
                                VARIANT_ORDER.get(s["profile_variant"], 9), dataset_sort_key(s["dataset"])))
    # Overall rows per (model, method)
    overall = defaultdict(list)
    for row in per_record:
        if row["status"] == "compared":
            overall[(row["model"], row["mode"], row["fixed_budget"], row["profile_variant"])].append(row)
    for (model, mode, budget, variant), comp in overall.items():
        divs = [r["first_divergence"] for r in comp if not r["exact"]]
        summary.append({
            "model": model, "dataset": "ALL", "mode": mode, "fixed_budget": budget, "profile_variant": variant,
            "method": method_label(mode, budget, variant), "n_records": None, "n_compared": len(comp),
            "n_no_ar": None, "n_prompt_mismatch": None,
            "n_exact": sum(r["exact"] for r in comp), "exact_rate": sum(r["exact"] for r in comp) / len(comp),
            "n_diverged": len(divs), "div_pos_min": int(min(divs)) if divs else None,
            "div_pos_p25": float(np.percentile(divs, 25)) if divs else None,
            "div_pos_median": float(np.median(divs)) if divs else None,
            "div_pos_p75": float(np.percentile(divs, 75)) if divs else None,
            "div_pos_max": int(max(divs)) if divs else None,
            "mean_prefix_agreement": float(np.mean([r["prefix_agreement"] for r in comp])),
            "mean_prefix_agreement_diverged": float(np.mean([r["prefix_agreement"] for r in comp if not r["exact"]])) if divs else None,
            "n_length_only_diff": sum(1 for r in comp if not r["exact"] and r["first_divergence"] == min(r["ar_len"], r["method_len"])),
        })
    return summary, per_record


def to_markdown(summary: list[dict]) -> str:
    def f(x, nd=2):
        return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines = ["| Model | Method | Dataset | compared | exact | exact rate | first-div. pos (min / median / max) | prefix agreement (all / diverged) | no AR | prompt≠ |",
             "|---|---|---|---:|---:|---:|---|---|---:|---:|"]
    for s in summary:
        lines.append(
            f"| {model_short(s['model'])} | {s['method']} | {s['dataset']} | {s['n_compared']} | {s['n_exact']} | "
            f"{f(None if s['exact_rate'] is None else 100 * s['exact_rate'], 1)}% | "
            f"{f(s['div_pos_min'])} / {f(s['div_pos_median'], 0)} / {f(s['div_pos_max'])} | "
            f"{f(s['mean_prefix_agreement'], 3)} / {f(s['mean_prefix_agreement_diverged'], 3)} | "
            f"{f(s['n_no_ar'])} | {f(s['n_prompt_mismatch'])} |"
        )
    lines.append("")
    lines.append("Exact = generated token ids identical to the AR output for the same prompt (T=0). "
                 "First-divergence position is a 0-based index into the generated tokens.")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ar-jsonl", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args(argv)
    records = load_records(args.runs)
    ar_index = ARIndex(records, load_records(args.ar_jsonl) if args.ar_jsonl else [])
    summary, per_record = compare(records, ar_index)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}lossless_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(out / f"{args.prefix}lossless_per_record.csv", "w", newline="") as fh:
        keys = sorted({k for r in per_record for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per_record)
    md = "# Lossless check (T=0): speculative outputs vs AR\n\n" + to_markdown(summary) + "\n"
    (out / f"{args.prefix}lossless.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
