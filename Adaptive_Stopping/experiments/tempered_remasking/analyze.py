"""Compute the paper's metrics from sampling files and emit Performance_Metrics rows.

Metrics: pass@k and pass@NFE (Fig. 2, 3, 6-8), Best@k / Best@NFE with
self-consistency and ORM selection (Fig. 4, 9), Entropy@k (Fig. 10).

    python Adaptive_Stopping/experiments/tempered_remasking/analyze.py \
        results/tempered_remasking/llada_gsm8k_*.jsonl --json summary.json --csv rows.csv

NFEs follow the paper's footnote 3 (``nfe_paper``): AR counts tokens before
<eos>; confidence-based samplers count NFEs until all L positions are unmasked.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from common import read_jsonl
from tempered_remasking import metrics as M

CSV_FIELDS = (
    "experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,"
    "framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,"
    "latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes"
).split(",")


def summarize(path: str, num_trials: int = 200) -> dict:
    recs = list(read_jsonl(path))
    if not recs:
        raise ValueError(f"{path} is empty")
    run = next((r["run"] for r in recs if r.get("run")), {}) or {}
    correct = [[s["correct"] for s in r["samples"]] for r in recs]
    nfes = [[s.get("nfe_paper", s["nfe"]) for s in r["samples"]] for r in recs]
    out = {
        "file": path,
        "run": run,
        "num_problems": len(recs),
        "n_samples": min(len(c) for c in correct),
        "pass_at_k": M.pass_at_k_curve(correct),
        "pass_at_nfe": M.pass_at_nfe_curve(correct, nfes),
        "mean_nfe_per_sample": sum(sum(f) / len(f) for f in nfes) / len(nfes),
    }
    answers = [[s.get("answer") for s in r["samples"]] for r in recs]
    if any(a is not None for ans in answers for a in ans):  # math tasks
        refs = [r["reference"] for r in recs]
        ks = list(out["pass_at_k"])
        out["best_at_nfe_self_consistency"] = M.best_at_nfe_curve(answers, refs, nfes, ks, method="self_consistency",
                                                                  correct=correct, num_trials=num_trials)
        out["entropy_at_k"] = {k: M.entropy_at_k(answers, k, num_trials) for k in ks}
        if all("orm_score" in s for r in recs for s in r["samples"]):
            scores = [[s["orm_score"] for s in r["samples"]] for r in recs]
            out["best_at_nfe_orm"] = M.best_at_nfe_curve(answers, refs, nfes, ks, method="orm", scores=scores,
                                                         correct=correct, num_trials=num_trials)
    walls = sorted(r["wall_time_s"] / len(r["samples"]) for r in recs if r.get("wall_time_s") is not None)
    if walls:
        out["latency_median_ms_per_sample"] = 1000 * walls[len(walls) // 2]
        out["latency_p95_ms_per_sample"] = 1000 * walls[min(len(walls) - 1, int(0.95 * len(walls)))]
    mems = [r["peak_memory_mb"] for r in recs if r.get("peak_memory_mb") is not None]
    out["peak_memory_mb"] = max(mems) if mems else None
    return out


def csv_rows(summary: dict, status: str = "reproduced") -> list:
    run = summary["run"]
    env = run.get("env", {})
    sampler = run.get("sampler", {})
    pol = run.get("policy", {})
    method = pol.get("name", "unknown") + "".join(f"_{k}{v}" for k, v in pol.items() if k != "name")
    rows = []
    for k, v in summary["pass_at_k"].items():
        rows.append({
            "experiment_id": f"{os.path.basename(summary['file'])}:pass@{k}",
            "paper_id": "tale_of_two_temperatures_2026",
            "method": method,
            "commit": env.get("commit"),
            "backbone_id": run.get("model_id"),
            "backbone_revision": run.get("revision"),
            "dataset_id": run.get("dataset"),
            "split": "test",
            "seed": run.get("seed"),
            "device": env.get("device"),
            "framework_version": f"torch {env.get('torch')}",
            "dtype": run.get("dtype"),
            "batch_size": run.get("batch_size"),
            "input_length": None,
            "output_length": sampler.get("gen_length"),
            "quality_metric": f"pass@{k}",
            "quality_value": v,
            "latency_median_ms": summary.get("latency_median_ms_per_sample"),
            "latency_p95_ms": summary.get("latency_p95_ms_per_sample"),
            "throughput": None,
            "peak_memory_mb": summary.get("peak_memory_mb"),
            "energy_j": None,
            "status": status,
            "notes": f"mean_nfe_per_sample={summary['mean_nfe_per_sample']:.1f}; T_token={sampler.get('token_temperature')}; energy not_measured",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--num-trials", type=int, default=200)
    ap.add_argument("--status", default="reproduced",
                    choices=["reproduced", "partial", "approximate", "not reproduced"],
                    help="status label written to every CSV row (root README rule 6)")
    args = ap.parse_args()
    summaries = [summarize(f, args.num_trials) for f in args.files]
    for s in summaries:
        print(f"{s['file']}: problems={s['num_problems']} mean_NFE={s['mean_nfe_per_sample']:.1f}")
        print("  pass@k:", {k: round(v, 4) for k, v in s["pass_at_k"].items()})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for s in summaries:
                w.writerows(csv_rows(s, args.status))


if __name__ == "__main__":
    main()
