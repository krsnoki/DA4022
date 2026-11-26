"""Reproduction script: SCPS vs baselines on MMLU-Pro (or the synthetic smoke set).

Usage (from the repository root):

    python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
        --config Adaptive_Stopping/configs/scps_mmlupro_qwen35_9b.yaml \
        --out-dir results/scps_mmlupro

Outputs in ``--out-dir``:

* ``per_question_<method>.jsonl``: portfolio answer classes, gold label,
  realized tokens, rounds, stop reason and wall time for every question;
* ``summary.json``: aggregate metrics, paired bootstrap deltas, environment;
* ``results.csv``: one row per method in the ``Performance_Metrics`` schema.

Timing scope: wall-clock per question around the controller call,
including tokenization, generation, option scoring and canonicalization;
excluding model load and warm-up questions.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import yaml

from Adaptive_Stopping.baselines.self_consistency import SelfConsistency, SelfConsistencyConfig
from Adaptive_Stopping.baselines.tree_prefix import TreePrefix, TreePrefixConfig
from Adaptive_Stopping.common.budget import Budget
from Adaptive_Stopping.methods.budget_policy.scps.controller import SCPSConfig, SCPSController
from Backbones.Generative_Pretrained.factory import build_backbone
from Performance_Metrics.schema import environment_metadata, make_row, peak_memory_mb, percentile, write_csv

METHODS = ("scps", "self_consistency", "tree_prefix", "semantic_pruning")


def _strict(cls, d: Dict):
    known = set(cls.__dataclass_fields__)
    bad = set(d) - known
    if bad:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(bad)}")
    return cls(**d)


def build_method(name: str, cfg: Dict, backbone, budget: Budget, run_seed: int):
    if name == "scps":
        return SCPSController(backbone, budget, SCPSConfig.from_dict(cfg), run_seed)
    if name == "self_consistency":
        return SelfConsistency(backbone, budget, _strict(SelfConsistencyConfig, cfg), run_seed)
    if name in ("tree_prefix", "semantic_pruning"):
        cfg = dict(cfg)
        cfg["semantic_pruning"] = name == "semantic_pruning"
        return TreePrefix(backbone, budget, _strict(TreePrefixConfig, cfg), run_seed)
    raise ValueError(f"unknown method {name!r}; expected one of {METHODS}")


def load_config(path: str) -> Dict:
    cfg = yaml.safe_load(Path(path).read_text())
    required = ["experiment_id", "paper_id", "report_status", "backbone", "dataset", "run_seed", "budget", "methods"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config {path} missing keys: {missing}")
    _strict(Budget, cfg["budget"])
    for name in cfg["methods"]:
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r} in config")
    return cfg


def load_items(ds: Dict) -> List[Dict]:
    if ds["id"].startswith("synthetic"):
        from Dataloaders.synthetic.mock_mcq import make_synthetic_mcq

        return make_synthetic_mcq(ds["num_questions"], seed=ds["seed"], **(ds.get("synthetic") or {}))
    from Dataloaders.evaluation.mmlu_pro import load_mmlu_pro, sample_questions

    records = load_mmlu_pro(ds.get("split", "test"), ds.get("revision"), ds.get("local_path"))
    exclude = []
    if ds.get("exclude_ids_file"):
        exclude = [int(x) for x in Path(ds["exclude_ids_file"]).read_text().split()]
    return sample_questions(records, ds["num_questions"], ds["seed"], exclude)


def paired_bootstrap(a: List[int], b: List[int], iters: int = 10000, seed: int = 0):
    """Mean of a - b and a percentile 95% interval, resampling questions."""
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"delta": sum(d) / n, "ci95": [means[int(0.025 * iters)], means[int(0.975 * iters) - 1]],
            "wins": sum(1 for x in d if x > 0), "losses": sum(1 for x in d if x < 0)}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", default=None, help="comma-separated subset of the config's methods")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N sampled questions")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(cfg["dataset"])
    if args.limit is not None:
        items = items[: args.limit]
    budget = Budget(**cfg["budget"])
    t_load = time.perf_counter()
    backbone = build_backbone(cfg["backbone"], items)
    load_s = time.perf_counter() - t_load
    meta = backbone.get_model_metadata()
    names = args.methods.split(",") if args.methods else list(cfg["methods"])

    per_method, summary = {}, {"experiment_id": cfg["experiment_id"], "num_questions": len(items),
                               "budget": cfg["budget"], "backbone": meta, "model_load_s": load_s,
                               "environment": environment_metadata(), "methods": {}}
    for name in names:
        method = build_method(name, cfg["methods"][name], backbone, budget, cfg["run_seed"])
        for w in items[: cfg.get("warmup_questions", 0)]:
            method.run(w)
        recs = []
        with open(out / f"per_question_{name}.jsonl", "w") as f:
            for it in items:
                res = method.run(it)
                led = res.ledger
                rec = {
                    "question_id": it["question_id"], "gold": it["answer"], "answers": res.answers,
                    "pass": res.pass_at_n(it["answer"]), "unique": res.unique_answers(),
                    "dup_token_frac": res.duplicate_token_fraction(), "majority": res.majority_answer(),
                    "rollouts": led["rollouts"], "decode_tokens": led["decode_tokens"],
                    "aux_tokens": led["aux_tokens"], "total_tokens": led["total_tokens"],
                    "prompt_tokens": led["prompt_tokens"], "rounds": res.rounds, "stop_reason": res.stop_reason,
                    "violations": res.metadata.get("violations", []), "wall_time_s": res.wall_time_s,
                }
                f.write(json.dumps(rec) + "\n")
                recs.append(rec)
        per_method[name] = recs
        n = len(recs) or 1
        lat = [r["wall_time_s"] * 1000 for r in recs]
        dups = [r["dup_token_frac"] for r in recs if r["dup_token_frac"] is not None]
        total_wall = sum(r["wall_time_s"] for r in recs)
        summary["methods"][name] = {
            "pass_at_n": sum(r["pass"] for r in recs) / n,
            "pass_count": sum(r["pass"] for r in recs),
            "majority_accuracy": sum(r["majority"] == r["gold"] for r in recs) / n,
            "mean_total_tokens": sum(r["total_tokens"] for r in recs) / n,
            "mean_decode_tokens": sum(r["decode_tokens"] for r in recs) / n,
            "mean_aux_tokens": sum(r["aux_tokens"] for r in recs) / n,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in recs) / n,
            "mean_rollouts": sum(r["rollouts"] for r in recs) / n,
            "mean_unique_answers": sum(r["unique"] for r in recs) / n,
            "mean_dup_token_frac": (sum(dups) / len(dups)) if dups else None,
            "latency_median_ms": percentile(lat, 50),
            "latency_p95_ms": percentile(lat, 95),
            "decode_tokens_per_s": (sum(r["decode_tokens"] for r in recs) / total_wall) if total_wall > 0 else None,
            "budget_violations": sum(1 for r in recs if r["violations"]),
            "stop_reasons": {s: sum(1 for r in recs if r["stop_reason"] == s) for s in {r["stop_reason"] for r in recs}},
        }

    if "scps" in per_method:
        summary["paired_vs_scps"] = {
            b: paired_bootstrap([r["pass"] for r in per_method["scps"]], [r["pass"] for r in per_method[b]], seed=cfg["run_seed"])
            for b in per_method if b != "scps"
        }
    mem = peak_memory_mb()
    summary["peak_memory"] = mem
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    commit = git_commit()
    rows = []
    device = summary["environment"].get("device_name") or meta.get("device") or summary["environment"]["processor"]
    framework = ";".join(f"{k}={meta[k]}" for k in meta if k.endswith("_version")) or None
    framework = f"python={summary['environment']['python']}" + (f";{framework}" if framework else "")
    r3 = lambda x: None if x is None else round(x, 3)
    mem_val = mem["cuda_peak_allocated_mb"] if mem["cuda_peak_allocated_mb"] is not None else mem["process_peak_rss_mb"]
    mem_label = "cuda_peak_allocated" if mem["cuda_peak_allocated_mb"] is not None else "process_peak_rss"
    for name, s in summary["methods"].items():
        rows.append(make_row(
            experiment_id=cfg["experiment_id"], paper_id=cfg["paper_id"], method=name, commit=commit,
            backbone_id=meta.get("id"), backbone_revision=meta.get("revision"), dataset_id=cfg["dataset"]["id"],
            split=cfg["dataset"].get("split"), seed=f"data={cfg['dataset']['seed']};run={cfg['run_seed']}",
            device=device, framework_version=framework, dtype=meta.get("dtype"), batch_size=1,
            input_length=round(s["mean_prompt_tokens"], 2), output_length=round(s["mean_decode_tokens"], 2),
            quality_metric=f"portfolio_pass@{budget.max_rollouts}", quality_value=round(s["pass_at_n"], 4),
            latency_median_ms=r3(s["latency_median_ms"]), latency_p95_ms=r3(s["latency_p95_ms"]),
            throughput=r3(s["decode_tokens_per_s"]), peak_memory_mb=r3(mem_val), energy_j=None,
            status=cfg["report_status"],
            notes=(f"n={len(items)}; mean_total_tokens={s['mean_total_tokens']:.2f} (decode+aux); "
                   f"unique={s['mean_unique_answers']:.3f}; dup_tok={r3(s['mean_dup_token_frac'])}; "
                   f"majority_acc={s['majority_accuracy']:.4f}; latency=per-question wall clock; "
                   f"throughput=decode tok/s; memory={mem_label} for whole process (shared across methods); "
                   f"energy not_measured"),
        ))
    write_csv(rows, str(out / "results.csv"))
    print(json.dumps({k: {m: round(v["pass_at_n"], 4) for m, v in summary["methods"].items()} for k in ["pass_at_n"]}))
    return summary


if __name__ == "__main__":
    main()
