#!/usr/bin/env python
"""Boxed-answer accuracy for gsm8k / math500 / aime25 per (model, mode, budget).

Quality sanity check: at T=0, lossless speculative decoding should give the
same accuracy as AR (differences can only come from bf16 divergences, see
check_lossless.py). This is NOT the paper's evaluation (the paper reports
speed only) and not a standard harness such as lm-eval / Qwen math eval.

Ground truth: upstream's cached prompt JSONL (``upstream/cache/<name>.jsonl``)
stores only the formatted prompt, so answers are taken from the same HF
dataset/config/split, iterated in the same order as upstream
``_prepare_dataset``; each record's ``orig_idx`` indexes that order. The
prompt text re-formatted with upstream's own ``DATASETS[name]['format']`` is
compared with the cached JSONL line to verify the alignment.

Answer extraction: content of the LAST ``\\boxed{...}`` (or ``\\fbox{...}``)
with balanced braces. Outputs without a box count as wrong (reported
separately as n_no_boxed). Normalization: strip whitespace, ``$``,
``\\text{}``/``\\mathrm{}``/``\\textbf{}`` wrappers, ``\\left``/``\\right``,
spacing commands, degree signs, ``\\%``, trailing period, thousands separators
in numbers, a leading ``x =``; ``\\dfrac``/``\\tfrac`` -> ``\\frac``. Strings
compare equal after normalization, or numerically (|a-b| <= 1e-6 max(1,|b|))
when both parse as numbers / simple fractions. Limitations: no symbolic
equivalence (e.g. ``\\frac{\\sqrt3}{2}`` vs ``\\frac{1}{2}\\sqrt{3}``), no
interval/set reordering, no unit handling beyond the listed strips. So
absolute MATH-500 accuracy is a lower bound; the comparison across modes is
the point.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, dataset_sort_key, load_records, method_label, model_short  # noqa: E402

MATH_DATASETS = ("gsm8k", "math500", "aime25")


# ---------------------------------------------------------------------------
# extraction + normalization
# ---------------------------------------------------------------------------

def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} with balanced braces."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            best = text[i : j - 1]
    if best is None:
        m = re.findall(r"\\boxed\s+([^\s$]+)", text)  # \boxed 5
        if m:
            best = m[-1]
    return best


def _unwrap(cmd: str, s: str) -> str:
    pat = re.compile(r"\\" + cmd + r"\s*\{")
    while True:
        m = pat.search(s)
        if not m:
            return s
        i = m.end()
        depth, j = 1, i
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        if depth:
            return s
        s = s[: m.start()] + s[i : j - 1] + s[j:]


def normalize_answer(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    for cmd in ("text", "textbf", "mathrm", "mbox", "mathbf", "boxed"):
        s = _unwrap(cmd, s)
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;:! ]", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("$", "")
    s = s.strip()
    s = re.sub(r"\.$", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # 1,234,567 -> 1234567
    m = re.fullmatch(r"[a-zA-Z]=(.+)", s)
    if m:
        s = m.group(1)
    return s


def _to_number(s: str) -> float | None:
    if s is None:
        return None
    t = s
    m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", t)
    if m:
        try:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(t)
    except ValueError:
        return None


def answers_match(pred: str | None, gold: str | None) -> bool:
    a, b = normalize_answer(pred), normalize_answer(gold)
    if a is None or b is None or a == "":
        return False
    if a == b:
        return True
    x, y = _to_number(a), _to_number(b)
    if x is not None and y is not None:
        return abs(x - y) <= 1e-6 * max(1.0, abs(y))
    return False


def gold_from_row(dataset: str, row: dict) -> str:
    if dataset == "gsm8k":
        return row["answer"].split("####")[-1].strip()
    if dataset == "math500":
        return str(row["answer"])
    if dataset == "aime25":
        return str(row.get("answer") if row.get("answer") is not None else row.get("solution"))
    raise ValueError(dataset)


# ---------------------------------------------------------------------------
# ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(dataset: str) -> tuple[list[str], dict]:
    """Gold answers in upstream order + alignment check against the cached prompt JSONL."""
    from datasets import load_dataset

    from bastion import benchmark as ub

    cfg = ub.DATASETS[dataset]
    ds = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    rows = list(ds)
    golds = [gold_from_row(dataset, r) for r in rows]
    check = {"hf_rows": len(rows), "cache_rows": None, "prompt_mismatches": None}
    cache = ub.CACHE_DIR / f"{dataset}.jsonl"
    if cache.exists():
        cached = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        check["cache_rows"] = len(cached)
        check["prompt_mismatches"] = sum(
            1 for r, c in zip(rows, cached) if cfg["format"](r) != c["turns"][0]
        ) + abs(len(rows) - len(cached))
    return golds, check


def score(records: list[dict], golds: dict[str, list[str]], decode) -> tuple[list[dict], list[dict]]:
    per = []
    for rec in records:
        ds = rec["dataset"]
        if ds not in golds:
            continue
        text = decode(rec["model"], rec["output_ids"])
        boxed = last_boxed(text)
        gold = golds[ds][rec["orig_idx"]]
        per.append({
            "model": rec["model"], "dataset": ds, "sample_idx": rec["sample_idx"], "orig_idx": rec["orig_idx"],
            "turn": rec["turn"], "mode": rec["mode"], "fixed_budget": rec.get("fixed_budget"),
            "method": method_label(rec["mode"], rec.get("fixed_budget")),
            "pred": boxed, "gold": gold, "has_box": boxed is not None,
            "correct": answers_match(boxed, gold),
            "hit_max_new_tokens": rec["num_output_tokens"] >= rec.get("max_new_tokens", 10**9),
        })
    groups = defaultdict(list)
    for p in per:
        groups[(p["model"], p["dataset"], p["mode"], p["fixed_budget"])].append(p)
    summary = []
    for (model, ds, mode, budget), rows in groups.items():
        summary.append({
            "model": model, "dataset": ds, "mode": mode,
            "fixed_budget": "" if budget is None else budget,
            "method": method_label(mode, budget),
            "n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "n_no_boxed": sum(not r["has_box"] for r in rows),
            "n_hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rows),
        })
    summary.sort(key=lambda s: (s["model"], dataset_sort_key(s["dataset"]), s["mode"], str(s["fixed_budget"])))
    return summary, per


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args(argv)

    records = [r for r in load_records(args.runs) if r["dataset"] in MATH_DATASETS]
    if not records:
        print("no gsm8k/math500/aime25 records found")
        return
    if str(ROOT / "upstream") not in sys.path:
        sys.path.insert(0, str(ROOT / "upstream"))
    golds, checks = {}, {}
    for ds in sorted({r["dataset"] for r in records}):
        golds[ds], checks[ds] = load_ground_truth(ds)
        print(f"[gt] {ds}: {checks[ds]}")
        if checks[ds]["prompt_mismatches"]:
            print(f"[WARN] {ds}: HF rows do not align with upstream's cached prompts; scores unreliable", file=sys.stderr)

    from transformers import AutoTokenizer

    toks = {}

    def decode(model, ids):
        if model not in toks:
            try:
                toks[model] = AutoTokenizer.from_pretrained(model, local_files_only=True)
            except Exception:
                toks[model] = AutoTokenizer.from_pretrained(model)
        return toks[model].decode(ids, skip_special_tokens=True)

    summary, per = score(records, golds, decode)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.prefix}math_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out / f"{args.prefix}math_scores_per_record.jsonl", "w", encoding="utf-8") as fh:
        for p in per:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / f"{args.prefix}math_alignment_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    lines = ["| Model | Dataset | Method | n | correct | accuracy | no box | hit max_new_tokens |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {model_short(s['model'])} | {s['dataset']} | {s['method']} | {s['n']} | {s['correct']} | "
                     f"{100 * s['accuracy']:.1f}% | {s['n_no_boxed']} | {s['n_hit_max_new_tokens']} |")
    md = "# Boxed-answer accuracy (quality sanity check)\n\n" + "\n".join(lines) + "\n"
    (out / f"{args.prefix}math_scores.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
