"""Summarise the GSM8K sweep written by ``run_sweep.py`` (one JSON per cell).

    python -m Mixture_of_Experts.experiments.xshare.summarize RESULTS_DIR [--out-csv PATH]

Prints a Markdown table and writes ``RESULTS_DIR/summary.csv`` (or ``--out-csv``).
One row per (batch size, method), pooled over the subsets found. What the
columns mean, in plain words:

* accuracy: total correct answers / total questions over all subsets, with a
  95% Wilson score interval. The interval is the range of true accuracies
  that are consistent with what we saw; it is wide when few questions were
  asked. Wilson is used instead of the plain normal interval because it
  behaves well near 0% and 100%.
* change (pt): pooled method accuracy minus pooled baseline accuracy (same
  batch size), in percentage points.
* McNemar b/c: the methods answer the same questions, so they are compared
  question by question. b = questions the baseline got right and the method
  got wrong; c = the reverse. Questions both got right (or both wrong) carry
  no information. p is the exact two-sided McNemar p-value: if the method and
  baseline were equally good, b and c would split like fair coin flips, and p
  is the chance of a split at least this lopsided. A large p means the
  difference is indistinguishable from noise.
* tokens/s: decode throughput, mean and sample standard deviation over
  subsets (blank with one subset). speed-up %: per subset,
  method tokens/s / baseline tokens/s - 1 on the same subset, then mean and
  standard deviation over subsets.
* step median / p95 ms, request latency median / p95 s, kept and activated
  experts, live sequences per step, J/token (and above idle): means over
  subsets. J/token is blank if any subset has no energy measurement.
* peak MB: the largest peak allocated memory over subsets.
* routing overhead (also written to routing_overhead.csv): for every keep-all
  control (a method named xshare_keep_all*), mean over subsets of (keep-all - baseline) median
  decode step time, plus how many generations of xshare_keep_all differ from
  the baseline's (0 means it reproduces the baseline exactly).

Differences that fall inside the confidence interval (or have a large
McNemar p) are noise: do not read them as an effect of the method.
Missing values are left empty, never filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

Z95 = 1.96

CSV_COLUMNS = [
    "batch_size", "method", "n_subsets", "n_questions", "accuracy", "acc_ci_low",
    "acc_ci_high", "acc_change_pt", "mcnemar_b", "mcnemar_c", "mcnemar_p",
    "tps_mean", "tps_sd", "speedup_pct_mean", "speedup_pct_sd", "step_median_ms",
    "step_p95_ms", "req_latency_median_s", "req_latency_p95_s", "kept_experts",
    "activated_experts", "live_seq_per_step", "j_per_token",
    "j_per_token_above_idle", "peak_memory_mb",
]


# ---------------------------------------------------------------- statistics

def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (fractions in 0-1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mean_or_none(values):
    """Mean, or None if there are no values or any is missing."""
    if not values or any(v is None for v in values):
        return None
    return statistics.fmean(values)


def sd_or_none(values):
    """Sample standard deviation (n-1), or None for fewer than two values."""
    if len(values) < 2 or any(v is None for v in values):
        return None
    return statistics.stdev(values)


# ------------------------------------------------------------------- loading

def load_runs(folder: Path) -> dict[tuple[int, str], dict[int, dict]]:
    """Return {(batch_size, method): {subset: run_json}}."""
    groups: dict[tuple[int, str], dict[int, dict]] = {}
    for path in sorted(folder.glob("xshare-gsm8k-*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        key = (run["batch_size"], run["method"])
        groups.setdefault(key, {})[run["subset"]] = run
    return groups


def correctness(runs: dict[int, dict]) -> dict[tuple[int, int], bool]:
    """Map (subset, question index) -> correct flag."""
    return {
        (s, rec["index"]): bool(rec["correct"])
        for s, run in runs.items()
        for rec in run.get("records") or []
    }


def generations(runs: dict[int, dict]) -> dict[tuple[int, int], str]:
    return {
        (s, rec["index"]): rec.get("generation")
        for s, run in runs.items()
        for rec in run.get("records") or []
    }


def field(run: dict, *path):
    """Nested lookup that returns None when any level is missing."""
    cur = run
    for key in path:
        if not isinstance(cur, dict) or cur.get(key) is None:
            return None
        cur = cur[key]
    return cur


def per_subset(runs: dict[int, dict], *path) -> list:
    return [field(runs[s], *path) for s in sorted(runs)]


# ------------------------------------------------------------------ one row

def paired_counts(method_runs: dict, base_runs: dict) -> tuple[int, int]:
    m, b = correctness(method_runs), correctness(base_runs)
    shared = m.keys() & b.keys()
    base_only = sum(1 for k in shared if b[k] and not m[k])
    method_only = sum(1 for k in shared if m[k] and not b[k])
    return base_only, method_only


def speedups(method_runs: dict, base_runs: dict) -> list[float]:
    out = []
    for s in sorted(method_runs.keys() & base_runs.keys()):
        mt = field(method_runs[s], "decode_tokens_per_s")
        bt = field(base_runs[s], "decode_tokens_per_s")
        if mt is not None and bt:
            out.append(100 * (mt / bt - 1))
    return out


def summarise_group(batch: int, method: str, runs: dict, base_runs: dict | None) -> dict:
    corr = correctness(runs)
    n_q = len(corr)
    n_ok = sum(corr.values())
    acc = n_ok / n_q if n_q else None
    lo, hi = wilson_interval(n_ok, n_q) if n_q else (None, None)
    tps = per_subset(runs, "decode_tokens_per_s")
    tps_clean = [t for t in tps if t is not None]
    mem = [m for m in per_subset(runs, "peak_memory_allocated_mb") if m is not None]
    row = {
        "batch_size": batch, "method": method, "n_subsets": len(runs),
        "n_questions": n_q, "accuracy": acc, "acc_ci_low": lo, "acc_ci_high": hi,
        "tps_mean": mean_or_none(tps_clean), "tps_sd": sd_or_none(tps_clean),
        "step_median_ms": mean_or_none(per_subset(runs, "decode_step_median_ms")),
        "step_p95_ms": mean_or_none(per_subset(runs, "decode_step_p95_ms")),
        "req_latency_median_s": mean_or_none(per_subset(runs, "request_latency_median_s")),
        "req_latency_p95_s": mean_or_none(per_subset(runs, "request_latency_p95_s")),
        "kept_experts": mean_or_none(per_subset(runs, "routing", "mean_kept_experts")),
        "activated_experts": mean_or_none(per_subset(runs, "routing", "mean_activated_experts")),
        "live_seq_per_step": mean_or_none(per_subset(runs, "routing", "mean_tokens_per_step")),
        "j_per_token": mean_or_none(per_subset(runs, "energy", "j_per_token")),
        "j_per_token_above_idle": mean_or_none(per_subset(runs, "energy", "j_per_token_above_idle")),
        "peak_memory_mb": max(mem) if mem else None,
    }
    if base_runs is not None and method != "baseline":
        b, c = paired_counts(runs, base_runs)
        base_corr = correctness(base_runs)
        base_acc = sum(base_corr.values()) / len(base_corr) if base_corr else None
        sp = speedups(runs, base_runs)
        row.update({
            "acc_change_pt": None if acc is None or base_acc is None else 100 * (acc - base_acc),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": mcnemar_exact(b, c),
            "speedup_pct_mean": mean_or_none(sp), "speedup_pct_sd": sd_or_none(sp),
        })
    return row


def build_rows(groups: dict) -> list[dict]:
    rows = []
    for (batch, method) in sorted(groups, key=lambda k: (k[0], k[1] != "baseline", k[1])):
        rows.append(summarise_group(batch, method, groups[(batch, method)], groups.get((batch, "baseline"))))
    return rows


KEEP_ALL_PREFIX = "xshare_keep_all"
OVERHEAD_COLUMNS = ["batch_size", "method", "overhead_ms", "n_matched", "n_differ"]


def routing_overhead(groups: dict) -> list[dict]:
    """Selection overhead from every keep-all control (methods named ``xshare_keep_all*``).

    A keep-all method runs the XShare code path but prunes nothing. Per batch
    size and keep-all method: mean over subsets of its median decode-step time
    minus the baseline's, and how many generations differ from the baseline
    (0 means it reproduces the baseline exactly).
    """
    out = []
    keep_methods = sorted({m for _, m in groups if m.startswith(KEEP_ALL_PREFIX)})
    for batch in sorted({b for b, _ in groups}):
        base = groups.get((batch, "baseline"))
        if base is None:
            continue
        for method in keep_methods:
            keep = groups.get((batch, method))
            if keep is None:
                continue
            diffs = [
                field(keep[s], "decode_step_median_ms") - field(base[s], "decode_step_median_ms")
                for s in sorted(keep.keys() & base.keys())
                if field(keep[s], "decode_step_median_ms") is not None
                and field(base[s], "decode_step_median_ms") is not None
            ]
            gk, gb = generations(keep), generations(base)
            shared = gk.keys() & gb.keys()
            out.append({
                "batch_size": batch, "method": method, "overhead_ms": mean_or_none(diffs),
                "n_matched": len(shared), "n_differ": sum(1 for k in shared if gk[k] != gb[k]),
            })
    return out


def write_overhead_csv(overhead: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERHEAD_COLUMNS)
        writer.writeheader()
        for o in overhead:
            writer.writerow({c: csv_value(o.get(c)) for c in OVERHEAD_COLUMNS})


# ---------------------------------------------------------------- formatting

def num(x, nd: int = 2) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def csv_value(x):
    if x is None:
        return ""
    return round(x, 4) if isinstance(x, float) else x


def pm(mean, sd, nd: int) -> str:
    if mean is None:
        return ""
    return f"{mean:.{nd}f}" if sd is None else f"{mean:.{nd}f} ± {sd:.{nd}f}"


def markdown_row(r: dict) -> str:
    acc = "" if r["accuracy"] is None else (
        f"{100 * r['accuracy']:.1f} [{100 * r['acc_ci_low']:.1f}, {100 * r['acc_ci_high']:.1f}]"
    )
    change = "" if r.get("acc_change_pt") is None else f"{r['acc_change_pt']:+.1f}"
    mc = "" if r.get("mcnemar_b") is None else f"{r['mcnemar_b']}/{r['mcnemar_c']}"
    jpt = ""
    if r["j_per_token"] is not None:
        jpt = num(r["j_per_token"], 3)
        if r["j_per_token_above_idle"] is not None:
            jpt += f" ({num(r['j_per_token_above_idle'], 3)})"
    cells = [
        r["batch_size"], r["method"], r["n_subsets"], acc, change, mc,
        num(r.get("mcnemar_p"), 3), pm(r["tps_mean"], r["tps_sd"], 1),
        pm(r.get("speedup_pct_mean"), r.get("speedup_pct_sd"), 1),
        num(r["step_median_ms"], 2), num(r["req_latency_median_s"], 2),
        num(r["kept_experts"], 2), num(r["activated_experts"], 2),
        num(r["live_seq_per_step"], 2), jpt, num(r["peak_memory_mb"], 0),
    ]
    return "| " + " | ".join(str(c) for c in cells) + " |"


MD_HEADER = [
    "batch", "method", "subsets", "accuracy % [95% CI]", "change (pt)", "McNemar b/c", "p",
    "tokens/s mean ± sd", "speed-up % mean ± sd", "step median ms",
    "request latency median s", "kept", "activated", "live seq./step",
    "J/token (above idle)", "peak MB",
]


def print_markdown(rows: list[dict], overhead: list[dict]) -> None:
    print("| " + " | ".join(MD_HEADER) + " |")
    print("|" + "---|" * len(MD_HEADER))
    for r in rows:
        print(markdown_row(r))
    print()
    for o in overhead:
        print(
            f"Routing overhead, batch {o['batch_size']}: {o['method']} minus baseline "
            f"median decode step = {num(o['overhead_ms'], 3)} ms; generations differing from "
            f"baseline: {o['n_differ']} of {o['n_matched']} matched questions."
        )


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: csv_value(r.get(c)) for c in CSV_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    groups = load_runs(args.results_dir)
    if not groups:
        raise SystemExit(f"no xshare-gsm8k-*.json files in {args.results_dir}")
    rows = build_rows(groups)
    overhead = routing_overhead(groups)
    print_markdown(rows, overhead)
    out = args.out_csv or args.results_dir / "summary.csv"
    write_csv(rows, out)
    print(f"\nWrote {out}")
    if overhead:
        overhead_path = out.parent / "routing_overhead.csv"
        write_overhead_csv(overhead, overhead_path)
        print(f"Wrote {overhead_path}")


if __name__ == "__main__":
    main()
