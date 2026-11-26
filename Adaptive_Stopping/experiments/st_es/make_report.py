"""Collect the run outputs of one config into small, committed result files.

Reads the calibration sweep, the evaluation replay of the selected configs,
the live benchmark and the tagger validation (all under ``ST_ES_RUN_DIR``), and
writes to ``Adaptive_Stopping/experiments/st_es/results/<evaluation dataset>/``:

* ``calibration_sweep.csv``: every (step type, delta) and step-count config on the calibration split;
* ``test_selected.csv``: standard + selected configs on the held-out split;
* ``live_summary.json``: measured latency / throughput / peak memory;
* ``calibration_curves.png``, ``test_tradeoff.png``;
* ``provenance.json``: trace checksums, configs, commit and environment.

It also replaces this dataset's rows in ``Performance_Metrics/results/st_es.csv``
(the repository's result schema). Replay rows carry no latency; latency and
memory come only from the live run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from Adaptive_Stopping.experiments.st_es.generate import cache_path
from Adaptive_Stopping.experiments.st_es.io_utils import (
    DEFAULT_RUN_DIR, git_commit as _git_commit, read_jsonl, upsert_metric_rows, write_json,
)
from Backbones.registry import get_backbone_entry

REPO = Path(__file__).resolve().parents[3]
METRICS_CSV = REPO / "Performance_Metrics" / "results" / "st_es.csv"

# Chart tokens (reference palette of the dataviz method; first three categorical slots, validated all-pairs).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def git_commit() -> str:
    return _git_commit(REPO)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("accuracy", "mean_tokens", "saved_pct", "stop_rate", "mean_answer_tokens"):
            r[k] = float(r[k])
        for k in ("delta", "k", "n"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else None
        r["tag"] = r["tag"] or None
    return rows


def paired_bootstrap(per_example: Path, key_a: str, key_b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% paired-bootstrap CI of accuracy(key_a) - accuracy(key_b) over the same problems."""
    import random

    with open(per_example, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    a = {r["uid"]: int(r["correct"]) for r in rows if r["key"] == key_a}
    b = {r["uid"]: int(r["correct"]) for r in rows if r["key"] == key_b}
    uids = sorted(set(a) & set(b))
    if not uids:
        raise ValueError(f"{per_example}: no paired rows for {key_a!r} and {key_b!r} (re-run replay_sweep)")
    diffs = [a[u] - b[u] for u in uids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return {"diff": sum(diffs) / len(diffs), "ci_low": boots[int(0.025 * n_boot)], "ci_high": boots[int(0.975 * n_boot) - 1],
            "n": len(uids), "a_only": sum(d > 0 for d in diffs), "b_only": sum(d < 0 for d in diffs)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")


def plot_calibration(rows, selected, title, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    std = next(r for r in rows if r["kind"] == "standard")
    sel_tags = [r["tag"] for r in (selected.get("st_es") or {}).values() if r]
    highlight = []
    for t in ["verification", "self_talk", *sel_tags]:
        if t not in highlight and len(highlight) < 3:
            highlight.append(t)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    tags = sorted({r["tag"] for r in rows if r["kind"] == "st_es"})
    for t in tags:
        pts = sorted((r for r in rows if r["kind"] == "st_es" and r["tag"] == t), key=lambda r: r["delta"])
        xs, ys = [p["mean_tokens"] for p in pts], [p["accuracy"] for p in pts]
        if t in highlight:
            c = SERIES[highlight.index(t)]
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, label=f"ST-ES: {t} (δ=0…20)", zorder=3)
            # label the δ=0 end, where the curves are far apart (their δ=20 ends all meet the standard point)
            ax.annotate(f"{t}, δ=0", (xs[0], ys[0]), xytext=(-6, 0), textcoords="offset points", color=INK2,
                        fontsize=8, va="center", ha="right")
        else:
            ax.plot(xs, ys, color=AXIS, lw=1, zorder=2,
                    label="ST-ES: other step types" if t == next(x for x in tags if x not in highlight) else None)
    sc = sorted((r for r in rows if r["kind"] == "step_count"), key=lambda r: r["k"])
    ax.plot([r["mean_tokens"] for r in sc], [r["accuracy"] for r in sc], color=INK2, lw=2, ls="--", marker="s", ms=3,
            label="step-count baseline (k steps)", zorder=3)
    ax.plot([std["mean_tokens"]], [std["accuracy"]], marker="*", ms=14, color=INK, ls="none", label="standard", zorder=4)
    for frac in (0.85, 0.90, 0.95):
        y = frac * std["accuracy"]
        ax.axhline(y, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.annotate(f"{int(frac * 100)}% of standard", (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                    textcoords="offset points", ha="right", color=MUTED, fontsize=7)
    ax.set_xlabel("mean generated tokens per problem (replay)", color=INK2)
    ax.set_ylabel("accuracy (pass@1, Math-Verify)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_test(rows, title, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=SURFACE)
    _style(ax)
    std = next(r for r in rows if r["kind"] == "standard")
    for i, (kind, label, marker) in enumerate([("st_es", "ST-ES (calibrated)", "o"),
                                               ("step_count", "step-count baseline (calibrated)", "s")]):
        pts = sorted((r for r in rows if r["kind"] == kind and r.get("target")), key=lambda r: r["target"])
        if not pts:
            continue
        ax.plot([p["mean_tokens"] for p in pts], [p["accuracy"] for p in pts], color=SERIES[i], lw=2, marker=marker,
                ms=8, label=label, zorder=3, markeredgecolor=SURFACE, markeredgewidth=2)
        by_key: dict[str, list] = {}
        for p in pts:  # several targets can select the same configuration: one label per point
            by_key.setdefault(p["key"], []).append(p)
        for group in by_key.values():
            p = group[0]
            name = "/".join(str(int(round(100 * q["target"]))) for q in group) + "%"
            desc = f"{p['tag']}, δ={p['delta']}" if kind == "st_es" else f"k={p['k']}"
            ax.annotate(f"{name}  ({desc})", (p["mean_tokens"], p["accuracy"]), xytext=(8, -3),
                        textcoords="offset points", color=INK2, fontsize=8)
    ax.plot([std["mean_tokens"]], [std["accuracy"]], marker="*", ms=14, color=INK, ls="none", label="standard",
            zorder=4)
    ax.set_xlabel("mean generated tokens per problem (replay)", color=INK2)
    ax.set_ylabel("accuracy (pass@1, Math-Verify)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed, cap, prompt = cfg["seed"], cfg["max_new_tokens"], cfg["prompt_id"]
    tr = {s: cache_path(args.run_dir, cfg[s]["dataset_id"], cfg[s]["num_samples"], cfg[s]["offset"], seed, cap, prompt)
          for s in ("calibration", "evaluation")}
    rep = {s: args.run_dir / "replay" / tr[s].parent.name / tr[s].stem for s in tr}
    cal_rows = read_csv(rep["calibration"] / "grid" / "sweep.csv")
    cal_sum = json.loads((rep["calibration"] / "grid" / "summary.json").read_text(encoding="utf-8"))
    sel = json.loads((rep["calibration"] / "grid" / "selected.json").read_text(encoding="utf-8"))
    test_rows = read_csv(rep["evaluation"] / "selected" / "sweep.csv")
    test_sum = json.loads((rep["evaluation"] / "selected" / "summary.json").read_text(encoding="utf-8"))
    live_path = args.run_dir / "live" / cfg["evaluation"]["dataset_id"] / f"n{cfg['live']['num_samples']}_s{seed}" / "summary.json"
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else None
    tv_path = args.run_dir / "tagger_validation.json"
    tagger_val = json.loads(tv_path.read_text(encoding="utf-8")) if tv_path.exists() else None

    # attach calibration targets to the test rows
    for kind, picks in sel["selected"].items():
        for t, r in picks.items():
            if r is None:
                continue
            for tr_row in test_rows:
                if tr_row["key"] == r["key"]:
                    tr_row.setdefault("targets", []).append(float(t))
    expanded = []
    for r in test_rows:
        if r["kind"] == "standard":
            expanded.append({**r, "target": None})
        for t in r.get("targets", []):
            expanded.append({**r, "target": t})
    # paired 95% bootstrap CIs on the held-out split: each config vs standard, and ST-ES vs step-count per target
    pe = rep["evaluation"] / "selected" / "per_example.csv"
    for r in expanded:
        if r["kind"] != "standard":
            ci = paired_bootstrap(pe, r["key"], "standard")
            r.update({"acc_diff_vs_standard": round(ci["diff"], 4), "ci95_low": round(ci["ci_low"], 4),
                      "ci95_high": round(ci["ci_high"], 4)})
    st_vs_sc = {}
    for t in sel["targets"]:
        a = (sel["selected"]["st_es"].get(str(t)) or {}).get("key")
        b = (sel["selected"]["step_count"].get(str(t)) or {}).get("key")
        if a and b:
            st_vs_sc[str(t)] = {"st_es": a, "step_count": b, **paired_bootstrap(pe, a, b)}

    ds = cfg["evaluation"]["dataset_id"]
    # Main configs report under the evaluation dataset id; other runs (e.g. the smoke config) set report_name
    # so they can never overwrite the main results, and write_metrics: false to keep them out of the metrics CSV.
    tag = cfg.get("report_name", ds)
    out = REPO / "Adaptive_Stopping" / "experiments" / "st_es" / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rep["calibration"] / "grid" / "sweep.csv", out / "calibration_sweep.csv")
    fields = ["kind", "key", "target", "tag", "delta", "k", "n", "accuracy", "mean_tokens", "saved_pct", "stop_rate",
              "mean_answer_tokens", "acc_diff_vs_standard", "ci95_low", "ci95_high"]
    write_json(out / "paired_comparisons.json", {
        "method": "paired bootstrap over the same held-out problems, 2000 resamples, seed 0; diff = accuracy(a) - accuracy(b)",
        "st_es_vs_step_count": st_vs_sc,
        "config_vs_standard": {f"{r['key']}@{r['target']}": {k: r[k] for k in ("acc_diff_vs_standard", "ci95_low", "ci95_high")}
                               for r in expanded if r["kind"] != "standard"},
    })
    with open(out / "test_selected.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(expanded)
    if live:
        write_json(out / "live_summary.json", {k: v for k, v in live.items() if k not in ("environment",)})
    plot_calibration(cal_rows, sel["selected"],
                     f"Calibration ({cfg['calibration']['dataset_id']}, n={cal_sum['n']}): accuracy vs. tokens",
                     out / "calibration_curves.png")
    plot_test(expanded, f"Held-out test ({ds}, n={test_sum['n']}): calibrated configurations",
              out / "test_tradeoff.png")

    manifests = {s: json.loads(tr[s].with_suffix(".manifest.json").read_text(encoding="utf-8")) for s in tr}
    commit = git_commit()
    write_json(out / "provenance.json", {
        "config": cfg, "commit_at_report": commit,
        "traces": {s: {"file": tr[s].name, "sha256": manifests[s]["sha256"], "num_records": manifests[s]["num_records"],
                       "records_reused_from_earlier_run": manifests[s].get("records_reused_from_earlier_run", 0),
                       "batching": manifests[s]["batching"], "throughput_stats": manifests[s]["throughput_stats"]}
                   for s in tr},
        "calibration_summary": {k: v for k, v in cal_sum.items() if k != "environment"},
        "test_summary": {k: v for k, v in test_sum.items() if k != "environment"},
        "selected": sel, "tagger_validation": tagger_val and {k: tagger_val[k] for k in
                                                              ("source", "n_steps", "exact_agreement", "per_tag")},
        "environment": manifests["evaluation"]["environment"],
    })

    # ---------------- Performance_Metrics rows ----------------
    bb = get_backbone_entry(cfg["backbone_id"])
    env = manifests["evaluation"]["environment"]
    prompt_lens = [len(r["prompt_ids"]) for r in read_jsonl(tr["evaluation"])]
    common = {"paper_id": cfg["paper_id"], "commit": commit, "backbone_id": bb["id"], "backbone_revision": bb["revision"],
              "dataset_id": ds, "split": "test", "seed": seed, "device": env.get("gpu", "unknown"),
              "framework_version": f"torch {env.get('torch')}; transformers {env.get('transformers')}",
              "dtype": bb["dtype"], "input_length": round(sum(prompt_lens) / len(prompt_lens), 1),
              "quality_metric": "pass@1 accuracy (Math-Verify, 1 sample)", "energy_j": ""}
    rows = []
    for r in expanded:
        if r["kind"] == "standard":
            name, status, method = "standard", "partial", "standard generation (no stopping)"
        elif r["kind"] == "st_es":
            name, status, method = f"st_es@{r['target']:.2f}", "approximate", f"ST-ES {r['tag']} delta={r['delta']}"
        else:
            name, status, method = f"step_count@{r['target']:.2f}", "partial", f"step-count baseline k={r['k']}"
        rows.append({**common, "experiment_id": f"st_es-{tag}-replay-{name}", "method": method, "batch_size": "",
                     "output_length": round(r["mean_tokens"], 1), "quality_value": round(r["accuracy"], 4),
                     "latency_median_ms": "", "latency_p95_ms": "", "throughput": "", "peak_memory_mb": "",
                     "status": status,
                     "notes": f"offline replay over cached traces (n={r['n']}, cap {cap} tokens); no runtime measured; "
                              f"tokens saved {r['saved_pct']:.1f}%; stop rate {r['stop_rate']:.2f}"
                              + ("; rule-based tagger substitute" if r["kind"] == "st_es" else "")})
    if live:
        for name, p in live["policies"].items():
            is_std = p["config"] is None
            rows.append({**common, "experiment_id": f"st_es-{tag}-live-{name.split(':')[0]}",
                         "method": "standard generation (no stopping)" if is_std else f"ST-ES live {name.split(':', 1)[1]}",
                         "batch_size": 1, "output_length": round(p["mean_generated_tokens"], 1),
                         "quality_value": round(p["accuracy"], 4),
                         "latency_median_ms": round(1000 * p["latency_median_s"], 1),
                         "latency_p95_ms": round(1000 * p["latency_p95_s"], 1),
                         "throughput": f"{p['decode_tokens_per_s']:.1f} tok/s",
                         "peak_memory_mb": round(p["peak_allocated_mb_max"], 1),
                         "status": "partial" if is_std else "approximate",
                         "notes": f"live measured run, n={p['n']}, batch 1, 1 warm-up; latency = prompt IDs on GPU to "
                                  f"final answer incl. tagging and forced answer; peak = torch max_memory_allocated; "
                                  f"stop rate {p['stop_rate']:.2f}"})
    # replace only this script's rows for this dataset; rows written by other scripts (e.g. prompt-guided) stay
    if not cfg.get("write_metrics", True):
        print(f"wrote {out}; metrics rows not written (write_metrics: false)")
        return
    upsert_metric_rows(METRICS_CSV, rows, id_prefix=f"st_es-{tag}-")
    print(f"wrote {out} and {len(rows)} rows to {METRICS_CSV}")


if __name__ == "__main__":
    main()
