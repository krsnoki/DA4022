#!/usr/bin/env python
"""E1 diagnostics that go into the reproduction report (CPU only).

Inputs: E1 harness JSONL files (AR, DFlash and adaptive BASTION on the same
prompts) and the ``aggregate.py`` summary JSON of the same runs.
Outputs ``<prefix>paper_comparison.{md,json}`` with:

* per-dataset speedup / tau, local vs paper Table 1 (A100, T=0), and the
  BASTION/DFlash speedup ratio for both;
* implied cycle cost in AR-step units, tau / speedup per dataset. A
  speculative cycle commits about tau tokens and TPOT(method) is about
  cycle time / tau, so cycle time / TPOT(AR) is about tau / speedup. This is
  approximate: TPOT is averaged per record and tau is pooled over steps;
* realized adaptive tree-size distribution (pooled over all steps);
* 95% percentile-bootstrap intervals (2000 resamples, seed 0) for the average
  speedups and their ratio, resampling prompts within each dataset (the
  AR / DFlash / BASTION triple of a prompt is kept together). Only prompt
  sampling variability is covered, not run-to-run timing noise;
* number of prompts on which BASTION's TPOT is lower than DFlash's;
* per-session mean TPOT when a run was resumed (session boundaries from
  ``<run>.meta.json``), as a check that the restart did not shift timing.

Only warm-profile adaptive BASTION records (the E1 configuration) are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAPER_DATASET_ORDER, PAPER_TABLE1_T0, load_records, model_short  # noqa: E402

MODES = ("ar", "dflash", "bastion")


def _e1_record(rec: dict) -> bool:
    if rec["mode"] == "bastion":
        return rec.get("fixed_budget") is None and rec.get("profile_mode") == "warm"
    return rec["mode"] in ("ar", "dflash")


def bootstrap(triples: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ds_names = list(triples)
    avg = np.zeros((n_boot, 3))
    per_ds = np.zeros((n_boot, len(ds_names)))
    for b in range(n_boot):
        sd, sb = [], []
        for i, ds in enumerate(ds_names):
            a = triples[ds]
            m = a[rng.integers(0, len(a), len(a))].mean(axis=0)
            sd.append(m[0] / m[1])
            sb.append(m[0] / m[2])
            per_ds[b, i] = m[1] / m[2]
        avg[b] = [np.mean(sd), np.mean(sb), np.mean(sb) / np.mean(sd)]
    lo, hi = np.percentile(avg, [2.5, 97.5], axis=0)
    plo, phi = np.percentile(per_ds, [2.5, 97.5], axis=0)
    return {
        "dflash_avg_speedup": [lo[0], hi[0]], "bastion_avg_speedup": [lo[1], hi[1]],
        "ratio_of_averages": [lo[2], hi[2]],
        "per_dataset_ratio": {ds: [plo[i], phi[i]] for i, ds in enumerate(ds_names)},
    }


def session_check(path: Path, recs: list[dict]) -> list[dict]:
    meta_path = path.with_name(path.name[: -len(".jsonl")] + ".meta.json")
    if not meta_path.exists():
        return []
    starts = sorted(s["start"] for s in json.loads(meta_path.read_text())["sessions"])
    if len(starts) < 2:
        return []
    out = []
    by = defaultdict(lambda: defaultdict(list))  # (dataset, mode) -> session -> tpot
    for r in recs:
        sess = sum(r["timestamp"] >= s for s in starts) - 1
        by[(r["dataset"], r["mode"])][sess].append(r["time_per_output_token"] * 1e3)
    for (ds, mode), sessions in sorted(by.items()):
        if len(sessions) > 1:
            out.append({"dataset": ds, "mode": mode,
                        "sessions": {str(k): {"n": len(v), "mean_tpot_ms": float(np.mean(v))} for k, v in sorted(sessions.items())}})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--summary", required=True, help="aggregate.py summary.json of the same runs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="e1_")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    summary = {(r["model"], r["dataset"], r["mode"]): r for r in json.load(open(args.summary))
               if r["mode"] != "bastion" or (r.get("fixed_budget") is None and r.get("profile_variant") in (None, "warm"))}
    result, md = [], ["# E1: local results vs paper Table 1 (A100, T=0)", ""]
    for path in map(Path, args.runs):
        recs = [r for r in load_records([path]) if _e1_record(r)]
        if not recs:
            continue
        model = recs[0]["model"]
        if (model, "dflash") not in PAPER_TABLE1_T0:  # Table 1 covers the Qwen3 targets only
            continue
        name = model_short(model)
        datasets = [d for d in PAPER_DATASET_ORDER if any(r["dataset"] == d for r in recs)]
        res: dict = {"model": model, "run": str(path)}

        rows = []
        for ds in datasets + ["Average"]:
            d, b = summary[(model, ds, "dflash")], summary[(model, ds, "bastion")]
            pd, pb = PAPER_TABLE1_T0[(model, "dflash")][ds], PAPER_TABLE1_T0[(model, "bastion")][ds]
            rows.append({"dataset": ds, "dflash": d["speedup"], "dflash_tau": d["aal"], "bastion": b["speedup"],
                         "bastion_tau": b["aal"], "mean_tree": b.get("mean_tree_size"),
                         "paper_dflash": pd[0], "paper_dflash_tau": pd[1], "paper_bastion": pb[0], "paper_bastion_tau": pb[1],
                         "ratio": b["speedup"] / d["speedup"], "paper_ratio": pb[0] / pd[0]})
        res["per_dataset"] = rows
        per = [r for r in rows if r["dataset"] != "Average"]
        res["implied_cycle_cost_ar_steps"] = {
            k: {"local_mean": float(np.mean([r[f"{p}_tau"] / r[p] for r in per])),
                "paper_mean": float(np.mean([r[f"paper_{p}_tau"] / r[f"paper_{p}"] for r in per]))}
            for k, p in (("DFlash", "dflash"), ("BASTION", "bastion"))}

        sizes = np.concatenate([np.asarray(r["tree_sizes"]) for r in recs if r["mode"] == "bastion"])
        cnt = Counter(sizes.tolist())
        res["tree_sizes"] = {"n_steps": int(sizes.size), "pooled_mean": float(sizes.mean()), "median": float(np.median(sizes)),
                             "max": int(sizes.max()), "distribution": {str(k): v for k, v in sorted(cnt.items())}}

        tp = defaultdict(dict)
        for r in recs:
            tp[(r["dataset"], r["orig_idx"], r["turn"])][r["mode"]] = r["time_per_output_token"]
        triples = {ds: np.array([[v[m] for m in MODES] for k, v in tp.items() if k[0] == ds and len(v) == 3]) for ds in datasets}
        res["bootstrap95"] = bootstrap(triples, args.n_boot, args.seed)
        res["bastion_faster_than_dflash"] = {ds: [int((a[:, 2] < a[:, 1]).sum()), int(len(a))] for ds, a in triples.items()}
        res["session_check"] = session_check(path, recs)
        result.append(res)

        md += [f"## {name}", "",
               "| Benchmark | DFlash local | DFlash paper | BASTION local | BASTION paper | BASTION/DFlash local | BASTION/DFlash paper "
               "| τ DFlash local / paper | τ BASTION local / paper | mean tree (local) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['dataset']} | {r['dflash']:.2f}× | {r['paper_dflash']:.2f}× | {r['bastion']:.2f}× | {r['paper_bastion']:.2f}× "
                      f"| {r['ratio']:.3f} | {r['paper_ratio']:.3f} | {r['dflash_tau']:.2f} / {r['paper_dflash_tau']:.2f} "
                      f"| {r['bastion_tau']:.2f} / {r['paper_bastion_tau']:.2f} | {r['mean_tree']:.1f} |")
        bs = res["bootstrap95"]
        ts = res["tree_sizes"]
        md += ["", f"Mean tree in the Average row = unweighted mean of per-dataset means; pooled over all {ts['n_steps']} steps: "
               f"{ts['pooled_mean']:.1f} (median {ts['median']:.0f}, max {ts['max']}). Distribution of realized sizes: "
               + ", ".join(f"{k}: {v} ({100 * v / ts['n_steps']:.1f}%)" for k, v in ts["distribution"].items()) + ".", "",
               "Implied cycle cost τ/speedup in AR-step units, mean over datasets (approximate): "
               + "; ".join(f"{k} local {v['local_mean']:.2f}, paper {v['paper_mean']:.2f}"
                           for k, v in res["implied_cycle_cost_ar_steps"].items()) + ".", "",
               f"95% bootstrap intervals (prompts resampled within dataset, {args.n_boot} resamples): DFlash average "
               f"[{bs['dflash_avg_speedup'][0]:.2f}, {bs['dflash_avg_speedup'][1]:.2f}], BASTION average "
               f"[{bs['bastion_avg_speedup'][0]:.2f}, {bs['bastion_avg_speedup'][1]:.2f}], ratio of averages "
               f"[{bs['ratio_of_averages'][0]:.3f}, {bs['ratio_of_averages'][1]:.3f}].", "",
               "| Benchmark | BASTION/DFlash 95% interval | prompts where BASTION TPOT < DFlash TPOT |", "|---|---|---:|"]
        for ds in datasets:
            lo, hi = bs["per_dataset_ratio"][ds]
            w, n = res["bastion_faster_than_dflash"][ds]
            md.append(f"| {ds} | [{lo:.3f}, {hi:.3f}] | {w}/{n} |")
        if res["session_check"]:
            md += ["", "Resumed run: mean TPOT (ms) per session for datasets that span sessions:", ""]
            for s in res["session_check"]:
                md.append(f"- {s['dataset']} {s['mode']}: " + ", ".join(
                    f"session {k}: {v['mean_tpot_ms']:.2f} ms (n={v['n']})" for k, v in s["sessions"].items()))
        md.append("")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}paper_comparison.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    (out / f"{args.prefix}paper_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
