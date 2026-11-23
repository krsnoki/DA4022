#!/usr/bin/env python
"""E4 diagnostics for a budget sweep (CPU only).

Inputs: a sweep JSONL (BASTION fixed budgets plus adaptive with the warm and
upstream-cold profiles) and the E1 JSONL of the same target, which supplies
AR (the speedup reference) and, on the same prompts, DFlash and the E1
adaptive-BASTION records.

Everything is computed on the prompt/turn keys present in every compared
configuration (the sweep's 16 prompts per benchmark). Speedup follows
``aggregate.py``: mean TPOT(AR) / mean TPOT(method) per dataset, then the
unweighted mean over datasets. Outputs ``<prefix>extras.{md,json}`` with:

* per-dataset speedup of every configuration, the per-dataset best fixed N
  and the "Oracle" average (paper Fig. 4: best fixed N per dataset,
  averaged);
* realized tree-size distributions of the adaptive variants;
* 95% percentile-bootstrap intervals (prompts resampled within dataset,
  configurations of one prompt kept together) for ratios of averages:
  adaptive-warm vs the oracle choice, vs each fixed N, vs the cold-profile
  adaptive, vs DFlash, and vs the E1 adaptive records of the same prompts
  (a repeat of the same configuration in a different process, i.e. a check of
  run-to-run timing reproducibility). Prompt sampling is the only resampled
  source of variation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAPER_DATASET_ORDER, load_records, model_short  # noqa: E402


def config_name(rec: dict, source: str) -> str | None:
    mode = rec["mode"]
    if mode == "ar":
        return "AR" if source == "e1" else None
    if mode == "dflash":
        return "DFlash (E1)" if source == "e1" else None
    if rec.get("fixed_budget") is not None:
        return f"Fixed-{rec['fixed_budget']}" if source == "sweep" else None
    pm = rec.get("profile_mode") or "upstream"
    if source == "e1":
        return "Adaptive warm (E1 repeat)" if pm == "warm" else None
    return "Adaptive warm" if pm == "warm" else "Adaptive upstream-cold"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default=None, help="default: <sweep run name>_")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prefix = args.prefix if args.prefix is not None else Path(args.sweep).stem + "_"

    tpot: dict = defaultdict(dict)  # (dataset, orig_idx, turn) -> config -> tpot
    hashes: dict = defaultdict(set)
    trees: dict = {}  # (config, key) -> realized tree sizes
    model = None
    for source, path in (("sweep", args.sweep), ("e1", args.e1)):
        for r in load_records([path]):
            name = config_name(r, source)
            if name is None:
                continue
            model = model or r["model"]
            key = (r["dataset"], r["orig_idx"], r["turn"])
            tpot[key][name] = r["time_per_output_token"]
            hashes[key].add(r.get("input_hash"))
            if name.startswith("Adaptive"):
                trees[(name, key)] = r.get("tree_sizes") or []
    fixed = sorted({c for v in tpot.values() for c in v if c.startswith("Fixed-")}, key=lambda c: int(c.split("-")[1]))
    configs = ["AR", "DFlash (E1)", "Adaptive warm (E1 repeat)", *fixed, "Adaptive warm", "Adaptive upstream-cold"]
    # Only keys where every configuration exists and all records saw the same prompt.
    keys = [k for k, v in tpot.items() if all(c in v for c in configs) and len(hashes[k]) == 1]
    n_skipped = sum(1 for k, v in tpot.items() if any(c in v for c in configs[3:]) and k not in keys)
    datasets = [d for d in PAPER_DATASET_ORDER if any(k[0] == d for k in keys)]
    arr = {d: np.array([[tpot[k][c] for c in configs] for k in keys if k[0] == d]) for d in datasets}
    ci = {c: i for i, c in enumerate(configs)}

    def speedups(sample: dict[str, np.ndarray]) -> np.ndarray:
        """datasets x configs speedup matrix."""
        return np.array([sample[d].mean(axis=0)[0] / sample[d].mean(axis=0) for d in datasets])

    sp = speedups(arr)
    avg = sp.mean(axis=0)
    fixed_idx = [ci[c] for c in fixed]
    best = {d: fixed[int(np.argmax(sp[i, fixed_idx]))] for i, d in enumerate(datasets)}
    oracle = float(np.mean([sp[i, ci[best[d]]] for i, d in enumerate(datasets)]))

    comparisons = [("Adaptive warm", "Oracle (best fixed N per dataset)")] + [("Adaptive warm", c) for c in fixed] + [
        ("Adaptive warm", "Adaptive upstream-cold"), ("Adaptive warm", "DFlash (E1)"),
        ("Adaptive warm", "Adaptive warm (E1 repeat)"), ("Adaptive upstream-cold", "DFlash (E1)")]
    rng = np.random.default_rng(args.seed)
    boot = np.zeros((args.n_boot, len(comparisons)))
    for b in range(args.n_boot):
        s = speedups({d: a[rng.integers(0, len(a), len(a))] for d, a in arr.items()})
        a = s.mean(axis=0)
        o = np.mean([s[i, ci[best[d]]] for i, d in enumerate(datasets)])  # oracle choice fixed from the full data
        for j, (x, y) in enumerate(comparisons):
            boot[b, j] = a[ci[x]] / (o if y.startswith("Oracle") else a[ci[y]])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    point = [avg[ci[x]] / (oracle if y.startswith("Oracle") else avg[ci[y]]) for x, y in comparisons]

    tree_stats = {}
    key_set = set(keys)
    for name in [c for c in configs if c.startswith("Adaptive")]:
        t = np.concatenate([np.asarray(v) for (n, k), v in trees.items() if n == name and k in key_set])
        tree_stats[name] = {"n_steps": int(t.size), "pooled_mean": float(t.mean()),
                            "distribution": {str(k): v for k, v in sorted(Counter(t.tolist()).items())}}

    result = {
        "model": model, "sweep": args.sweep, "e1": args.e1, "n_keys": len(keys), "n_keys_skipped": n_skipped,
        "keys_per_dataset": {d: len(a) for d, a in arr.items()}, "configs": configs,
        "speedup_per_dataset": {d: {c: float(sp[i, ci[c]]) for c in configs[1:]} for i, d in enumerate(datasets)},
        "speedup_average": {c: float(avg[ci[c]]) for c in configs[1:]},
        "best_fixed_per_dataset": best, "oracle_average": oracle,
        "ratios": [{"num": x, "den": y, "point": float(p), "ci95": [float(l), float(h)]}
                   for (x, y), p, l, h in zip(comparisons, point, lo, hi)],
        "adaptive_tree_sizes": tree_stats,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}extras.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    cols = configs[1:]
    md = [f"# E4 budget sweep diagnostics: {model_short(model)}", "",
          f"{len(keys)} prompt/turn keys present in every configuration ({', '.join(f'{d} {len(a)}' for d, a in arr.items())}); "
          f"{n_skipped} sweep keys skipped. AR, DFlash and 'Adaptive warm (E1 repeat)' are the E1 records of the same prompts. "
          "Speedup = mean TPOT(AR) / mean TPOT(config) per dataset; Average = unweighted mean over datasets.", "",
          "| Dataset | " + " | ".join(cols) + " | best fixed |", "|---|" + "---:|" * len(cols) + "---|"]
    for i, d in enumerate(datasets):
        md.append(f"| {d} | " + " | ".join(f"{sp[i, ci[c]]:.2f}" for c in cols) + f" | {best[d]} |")
    md.append("| **Average** | " + " | ".join(f"**{avg[ci[c]]:.3f}**" for c in cols) + f" | Oracle {oracle:.3f} |")
    md += ["", f"Ratios of averages with 95% bootstrap intervals ({args.n_boot} resamples of prompts within dataset; "
           "the oracle's per-dataset N is fixed from the full data):", "", "| Ratio | point | 95% interval |", "|---|---:|---|"]
    for r in result["ratios"]:
        md.append(f"| {r['num']} / {r['den']} | {r['point']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    md += ["", "Realized adaptive tree sizes (pooled over steps):", ""]
    for name, t in tree_stats.items():
        md.append(f"- {name}: mean {t['pooled_mean']:.1f} over {t['n_steps']} steps; "
                  + ", ".join(f"{k}: {100 * v / t['n_steps']:.1f}%" for k, v in t["distribution"].items()))
    md.append("")
    (out / f"{prefix}extras.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
