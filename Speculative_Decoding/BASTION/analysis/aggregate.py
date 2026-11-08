#!/usr/bin/env python
"""Aggregate harness JSONL records into Table-1-style summaries.

Definitions (mirroring upstream ``bastion/benchmark.py::_print_decode_summary``):

* unit of averaging: one *record* = one (dataset, sample, turn, mode, budget)
  generation; MT-Bench contributes two records (turns) per sample.
* mean TPOT: arithmetic mean over records of each record's
  ``time_per_output_token`` (decode time / output tokens, prefill excluded,
  as computed inside upstream's generate functions).
* throughput (tok/s): ``1 / mean TPOT`` (what upstream prints).
* speedup: ``mean TPOT(AR) / mean TPOT(method)``, both means taken over the
  SAME set of (model, dataset, sample, turn) keys (records without an AR
  counterpart, or whose AR counterpart had a different prompt hash, are
  excluded and counted).
* AAL (tau): mean over all decoding steps pooled across the group's records
  (upstream ``_mean_acceptance_length``); includes the bonus token.
* mean tree size: mean realized tree size over all BASTION steps pooled
  across records (per-record means also reported).
* "Average" rows: arithmetic mean over datasets of the per-dataset speedups /
  AALs (this is how the paper's Table 1 "Average" column is computed; checked
  against the paper's numbers). "Pooled" rows: the same formulas applied to
  all records of all datasets at once (what upstream prints for a group alias).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASET_LABELS, DATASET_SOURCES, PAPER_DATASET_ORDER, PAPER_ID, dataset_sort_key, load_meta,
    load_records, method_label, model_short, turn_key,
)

DA4022_COLUMNS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision", "dataset_id", "split",
    "seed", "device", "framework_version", "dtype", "batch_size", "input_length", "output_length",
    "quality_metric", "quality_value", "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]

MODE_ORDER = {"ar": 0, "dflash": 1, "bastion": 2}


def _budget_sort(b):
    return -1 if b is None else int(b)


class ARIndex:
    """AR reference lookup by (model, dataset, sample_idx, turn)."""

    def __init__(self, primary: list[dict], external: list[dict] | None = None) -> None:
        self.by_key: dict[tuple, list[dict]] = defaultdict(list)
        for rec in primary:
            if rec["mode"] == "ar":
                self.by_key[turn_key(rec)].append(rec)
        self.external_keys = 0
        for rec in external or []:
            if rec["mode"] == "ar":
                rec = dict(rec, _external=True)
                self.by_key[turn_key(rec)].append(rec)
                self.external_keys += 1

    def lookup(self, rec: dict) -> dict | None:
        cands = self.by_key.get(turn_key(rec), [])
        if not cands:
            return None
        same_run = [c for c in cands if c.get("run_name") == rec.get("run_name") and not c.get("_external")]
        if same_run:
            return same_run[0]
        external = [c for c in cands if c.get("_external")]
        return external[0] if external else cands[0]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _summ_group(recs: list[dict], ar_index: ARIndex, hash_check: bool) -> dict:
    mode = recs[0]["mode"]
    tpots = np.array([r["time_per_output_token"] for r in recs], dtype=np.float64)
    matched_ar, matched_m = [], []
    n_no_ar = n_hash_mismatch = 0
    ar_runs = set()
    for r in recs:
        ar = r if mode == "ar" else ar_index.lookup(r)
        if ar is None:
            n_no_ar += 1
            continue
        if hash_check and ar.get("input_hash") != r.get("input_hash"):
            n_hash_mismatch += 1
            continue
        matched_ar.append(ar["time_per_output_token"])
        matched_m.append(r["time_per_output_token"])
        ar_runs.add(ar.get("run_name"))
    speedup = None
    if matched_m:
        speedup = float(np.mean(matched_ar) / np.mean(matched_m))
    steps = [a for r in recs for a in (r.get("acceptance_lengths") or [])]
    sizes = [s for r in recs for s in (r.get("tree_sizes") or [])]
    per_rec_sizes = [r.get("mean_tree_size") for r in recs if r.get("tree_sizes")]
    mismatch = [r.get("tree_size_mismatch_steps") for r in recs if r.get("tree_size_mismatch_steps") is not None]
    out = {
        "model": recs[0]["model"],
        "dataset": recs[0]["dataset"],
        "mode": mode,
        "fixed_budget": recs[0].get("fixed_budget"),
        "method": method_label(mode, recs[0].get("fixed_budget")),
        "run_names": sorted({r.get("run_name") for r in recs}),
        "n_records": len(recs),
        "n_samples": len({r["sample_idx"] for r in recs}),
        "n_matched_ar": len(matched_m),
        "n_no_ar": n_no_ar,
        "n_hash_mismatch": n_hash_mismatch,
        "ar_runs_used": sorted(x for x in ar_runs if x),
        "mean_tpot_ms": float(tpots.mean() * 1e3),
        "tokens_per_s": float(1.0 / tpots.mean()) if tpots.mean() > 0 else None,
        "median_tpot_ms": float(np.median(tpots) * 1e3),
        "p95_tpot_ms": float(np.percentile(tpots, 95) * 1e3),
        "speedup": speedup,
        "ar_mean_tpot_matched_ms": float(np.mean(matched_ar) * 1e3) if matched_ar else None,
        "mode_mean_tpot_matched_ms": float(np.mean(matched_m) * 1e3) if matched_m else None,
        "aal": None if mode == "ar" or not steps else float(np.mean(steps)),
        "n_steps": len(steps),
        "mean_tree_size": float(np.mean(sizes)) if sizes else None,
        "mean_tree_size_per_record": _mean(per_rec_sizes),
        "tree_size_mismatch_steps": int(sum(mismatch)) if mismatch else (0 if mode == "bastion" and recs[0].get("fixed_budget") is not None else None),
        "mean_input_tokens": float(np.mean([r["num_input_tokens"] for r in recs])),
        "mean_output_tokens": float(np.mean([r["num_output_tokens"] for r in recs])),
        "total_output_tokens": int(sum(r["num_output_tokens"] for r in recs)),
        "mean_ttft_ms": _mean([None if r.get("time_to_first_token") is None else r["time_to_first_token"] * 1e3 for r in recs]),
        "peak_mem_mib": max((r.get("peak_mem_allocated_mib") or 0.0) for r in recs) or None,
        "frac_hit_max_new_tokens": float(np.mean([r["num_output_tokens"] >= r.get("max_new_tokens", math.inf) for r in recs])),
        "_matched_ar_tpots": matched_ar,
        "_matched_mode_tpots": matched_m,
        "_steps": steps,
        "_sizes": sizes,
    }
    return out


def summarize(records: list[dict], ar_index: ARIndex | None = None, hash_check: bool = True) -> list[dict]:
    """Per-(model, dataset, mode, budget) rows plus Average and Pooled rows."""
    if ar_index is None:
        ar_index = ARIndex(records)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["model"], r["dataset"], r["mode"], r.get("fixed_budget"))].append(r)
    rows = [_summ_group(v, ar_index, hash_check) for v in groups.values()]
    rows.sort(key=lambda x: (x["model"], MODE_ORDER.get(x["mode"], 9), _budget_sort(x["fixed_budget"]), dataset_sort_key(x["dataset"])))

    by_method: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[(row["model"], row["mode"], row["fixed_budget"])].append(row)
    extra = []
    for (model, mode, budget), mrows in by_method.items():
        datasets = sorted((r["dataset"] for r in mrows), key=dataset_sort_key)
        sp = [r["speedup"] for r in mrows if r["speedup"] is not None]
        aal = [r["aal"] for r in mrows if r["aal"] is not None]
        sizes = [r["mean_tree_size"] for r in mrows if r["mean_tree_size"] is not None]
        base = {
            "model": model, "mode": mode, "fixed_budget": budget, "method": method_label(mode, budget),
            "run_names": sorted({n for r in mrows for n in r["run_names"]}),
            "n_records": sum(r["n_records"] for r in mrows),
            "n_samples": sum(r["n_samples"] for r in mrows),
            "n_matched_ar": sum(r["n_matched_ar"] for r in mrows),
            "n_no_ar": sum(r["n_no_ar"] for r in mrows),
            "n_hash_mismatch": sum(r["n_hash_mismatch"] for r in mrows),
            "datasets_included": datasets,
            "missing_paper_datasets": [d for d in PAPER_DATASET_ORDER if d not in datasets],
        }
        avg = dict(base, dataset="Average",
                   speedup=float(np.mean(sp)) if sp else None,
                   n_datasets_with_speedup=len(sp),
                   aal=float(np.mean(aal)) if aal else None,
                   mean_tree_size=float(np.mean(sizes)) if sizes else None,
                   tokens_per_s=float(np.mean([r["tokens_per_s"] for r in mrows])),
                   average_kind="unweighted mean over datasets of per-dataset values")
        all_ar = [x for r in mrows for x in r["_matched_ar_tpots"]]
        all_m = [x for r in mrows for x in r["_matched_mode_tpots"]]
        all_steps = [x for r in mrows for x in r["_steps"]]
        all_sizes = [x for r in mrows for x in r["_sizes"]]
        tp_all = [x for r in mrows for x in [r["mean_tpot_ms"]] * r["n_records"]]
        pooled = dict(base, dataset="Pooled",
                      speedup=float(np.mean(all_ar) / np.mean(all_m)) if all_m else None,
                      aal=None if mode == "ar" or not all_steps else float(np.mean(all_steps)),
                      mean_tree_size=float(np.mean(all_sizes)) if all_sizes else None,
                      tokens_per_s=float(1e3 / np.mean(tp_all)) if tp_all else None,
                      average_kind="all records of all datasets pooled (upstream formulas)")
        extra += [avg, pooled]
    extra.sort(key=lambda x: (x["model"], MODE_ORDER.get(x["mode"], 9), _budget_sort(x["fixed_budget"]), x["dataset"]))
    return rows + extra


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _fmt(x, nd=2, suffix=""):
    return "–" if x is None else f"{x:.{nd}f}{suffix}"


def table1_markdown(rows: list[dict], datasets: list[str] | None = None) -> str:
    per = [r for r in rows if r["dataset"] not in ("Average", "Pooled")]
    if datasets is None:
        datasets = sorted({r["dataset"] for r in per}, key=dataset_sort_key)
    lut = {(r["model"], r["mode"], r["fixed_budget"], r["dataset"]): r for r in rows}
    methods = sorted({(r["model"], r["mode"], r["fixed_budget"]) for r in rows if r["mode"] != "ar"},
                     key=lambda m: (m[0], MODE_ORDER.get(m[1], 9), _budget_sort(m[2])))
    head = "| Model | Method | " + " | ".join(f"{DATASET_LABELS.get(d, d)} Speedup | τ" for d in datasets) + " | Average Speedup | τ | Pooled Speedup | τ |"
    sep = "|---|---|" + "---:|---:|" * (len(datasets) + 2)
    lines = [head, sep]
    for model, mode, budget in methods:
        cells = []
        for d in datasets + ["Average", "Pooled"]:
            r = lut.get((model, mode, budget, d))
            cells += [_fmt(r and r["speedup"], 2, "×"), _fmt(r and r["aal"])]
        lines.append(f"| {model_short(model)} | {method_label(mode, budget)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Speedup = mean TPOT(AR) / mean TPOT(method) over records with a matching AR record "
                 "(same dataset, sample, turn and prompt hash). τ = AAL pooled over steps. "
                 "Average = unweighted mean over the datasets shown (paper convention); "
                 "Pooled = all records pooled (upstream group-alias convention).")
    return "\n".join(lines)


def details_markdown(rows: list[dict]) -> str:
    lines = ["| Model | Method | Dataset | n rec | AR-matched | no AR | hash≠ | mean TPOT ms | tok/s | speedup | AAL | mean tree | mean out tok | hit max_new |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["dataset"] in ("Average", "Pooled"):
            continue
        lines.append(
            f"| {model_short(r['model'])} | {r['method']} | {r['dataset']} | {r['n_records']} | {r['n_matched_ar']} | "
            f"{r['n_no_ar']} | {r['n_hash_mismatch']} | {_fmt(r['mean_tpot_ms'])} | {_fmt(r['tokens_per_s'], 1)} | "
            f"{_fmt(r['speedup'], 2, '×')} | {_fmt(r['aal'])} | {_fmt(r['mean_tree_size'], 1)} | "
            f"{_fmt(r['mean_output_tokens'], 0)} | {_fmt(100 * r['frac_hit_max_new_tokens'], 0, '%')} |"
        )
    return "\n".join(lines)


def load_quality(path: str | None) -> dict:
    """score_math.py CSV -> {(model, dataset, mode, budget): (accuracy, n)}."""
    if not path:
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            budget = None if row["fixed_budget"] in ("", "None") else int(row["fixed_budget"])
            out[(row["model"], row["dataset"], row["mode"], budget)] = (float(row["accuracy"]), int(row["n"]))
    return out


def da4022_rows(rows: list[dict], metas: dict[str, dict], quality: dict, status: str) -> list[dict]:
    out = []
    for r in rows:
        meta = {}
        for name in r["run_names"]:
            if name in metas:
                meta = metas[name]
                break
        env = meta.get("env", {})
        pk = env.get("packages", {})
        rev = (meta.get("hf_revisions", {}).get("target", {}) or {})
        ds = r["dataset"]
        if ds in ("Average", "Pooled"):
            dataset_id = f"{ds.lower()}({','.join(r.get('datasets_included', []))})"
            split = ""
        else:
            dataset_id, split = DATASET_SOURCES.get(ds, (ds, ""))
        q = quality.get((r["model"], ds, r["mode"], r["fixed_budget"]))
        notes = [
            f"run={'+'.join(r['run_names'])}",
            f"upstream={str(meta.get('upstream_commit', ''))[:7]}",
            "latency=per-output-token decode latency (TPOT; prefill excluded; median/p95 over records)",
            "throughput=1/mean(TPOT) decode tokens/s",
            f"speedup_vs_AR={_fmt(r.get('speedup'), 3)} (n_matched={r['n_matched_ar']}, no_AR={r['n_no_ar']}, hash_mismatch={r['n_hash_mismatch']})",
            f"AAL={_fmt(r.get('aal'), 3)}",
            f"n_records={r['n_records']}",
            "energy=not_measured",
            "peak_memory=torch.cuda.max_memory_allocated incl. weights (max over records)",
        ]
        if r.get("mean_tree_size") is not None:
            notes.append(f"mean_tree_size={r['mean_tree_size']:.1f}")
        if r.get("average_kind"):
            notes.append(f"aggregate={r['average_kind']}")
        if q is None:
            notes.append("quality=not_scored")
        if meta.get("gpu_key") and "b4500" in str(meta.get("gpu_key")):
            notes.append("cost-model GPU spec registered at runtime (not in upstream table)")
        per_dataset = ds not in ("Average", "Pooled")
        out.append({
            "experiment_id": f"{r['run_names'][0] if r['run_names'] else 'run'}:{model_short(r['model'])}:{ds}:{r['method']}",
            "paper_id": PAPER_ID,
            "method": r["method"],
            "commit": meta.get("da4022_commit", ""),
            "backbone_id": r["model"],
            "backbone_revision": rev.get("config_commit_hash") or rev.get("refs_main") or "",
            "dataset_id": dataset_id,
            "split": split,
            "seed": (meta.get("args") or {}).get("seed", ""),
            "device": env.get("gpu_name", ""),
            "framework_version": (f"torch {pk.get('torch')}; transformers {pk.get('transformers')}; CUDA {env.get('torch_cuda')}"
                                  if pk else ""),
            "dtype": "bf16",
            "batch_size": 1,
            "input_length": f"{r['mean_input_tokens']:.1f}" if per_dataset else "",
            "output_length": f"{r['mean_output_tokens']:.1f}" if per_dataset else "",
            "quality_metric": "boxed-answer accuracy" if q else "",
            "quality_value": f"{q[0]:.4f}" if q else "",
            "latency_median_ms": f"{r['median_tpot_ms']:.3f}" if per_dataset else "",
            "latency_p95_ms": f"{r['p95_tpot_ms']:.3f}" if per_dataset else "",
            "throughput": f"{r['tokens_per_s']:.2f}" if r.get("tokens_per_s") else "",
            "peak_memory_mb": f"{r['peak_mem_mib']:.0f}" if per_dataset and r.get("peak_mem_mib") else "",
            "energy_j": "",
            "status": status,
            "notes": "; ".join(notes),
        })
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="harness JSONL files")
    ap.add_argument("--ar-jsonl", nargs="*", default=[], help="external runs whose AR records serve as reference")
    ap.add_argument("--no-hash-check", action="store_true", help="match AR records on keys only (not recommended)")
    ap.add_argument("--quality", default=None, help="score_math.py CSV to fill quality columns")
    ap.add_argument("--status", default="partial", help="DA4022 status column value")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="", help="filename prefix for outputs")
    args = ap.parse_args(argv)

    records = load_records(args.runs)
    external = load_records(args.ar_jsonl) if args.ar_jsonl else []
    ar_index = ARIndex(records, external)
    rows = summarize(records, ar_index, hash_check=not args.no_hash_check)
    metas = {}
    for p in list(args.runs) + list(args.ar_jsonl):
        m = load_meta(p)
        if m.get("run_name"):
            metas[m["run_name"]] = m
        for r in load_records([p])[:1]:
            metas.setdefault(r.get("run_name"), m)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pre = args.prefix
    pub = [public(r) for r in rows]
    (out / f"{pre}summary.json").write_text(json.dumps(pub, indent=1), encoding="utf-8")
    keys = sorted({k for r in pub for k in r})
    with open(out / f"{pre}summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in pub:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    with open(out / f"{pre}da4022_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DA4022_COLUMNS)
        w.writeheader()
        for r in da4022_rows(rows, metas, load_quality(args.quality), args.status):
            w.writerow(r)
    md = ["# BASTION reproduction summary", "",
          f"Runs: {', '.join(args.runs)}", f"External AR runs: {', '.join(args.ar_jsonl) or 'none'}",
          f"External AR records loaded: {ar_index.external_keys}", "",
          table1_markdown(rows), "", "## Per-group details", "", details_markdown(rows), ""]
    (out / f"{pre}table1.md").write_text("\n".join(md), encoding="utf-8")

    # Console report incl. matching counts.
    print("\n".join(md[:6]))
    print(table1_markdown(rows))
    tot = [r for r in rows if r["dataset"] not in ("Average", "Pooled") and r["mode"] != "ar"]
    print(f"\nAR matching: {sum(r['n_matched_ar'] for r in tot)} matched, {sum(r['n_no_ar'] for r in tot)} without AR, "
          f"{sum(r['n_hash_mismatch'] for r in tot)} prompt-hash mismatches (excluded)")
    print(f"wrote {out}/{pre}summary.csv, {pre}summary.json, {pre}da4022_results.csv, {pre}table1.md")


if __name__ == "__main__":
    main()
