"""Reproduction script: SCPS vs baselines on MMLU-Pro (or the synthetic smoke set).

Usage (from the repository root):

    python -m Adaptive_Stopping.experiments.scps_mmlupro.run \
        --config Adaptive_Stopping/configs/scps_mmlupro_qwen35_9b.yaml \
        --out-dir results/scps_mmlupro

Methods can be run in separate invocations (``--methods``) into the same
``--out-dir``; the summary is rebuilt from every per-question file present.

Outputs in ``--out-dir``:

* ``per_question_<method>.jsonl``: portfolio answer classes, gold label,
  realized tokens, rounds, stop reason and wall time for every question;
* ``run_meta_<method>.json``: environment, backbone metadata, model load
  time and peak memory of the invocation that produced that method;
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

from Adaptive_Stopping.baselines.option_topk import OptionTopK, OptionTopKConfig
from Adaptive_Stopping.baselines.self_consistency import SelfConsistency, SelfConsistencyConfig
from Adaptive_Stopping.baselines.tree_prefix import TreePrefix, TreePrefixConfig
from Adaptive_Stopping.common.budget import Budget
from Adaptive_Stopping.methods.budget_policy.scps.controller import SCPSConfig, SCPSController
from Backbones.Generative_Pretrained.factory import build_backbone
from Performance_Metrics.schema import environment_metadata, make_row, peak_memory_mb, percentile, write_csv

METHODS = ("scps", "self_consistency", "tree_prefix", "semantic_pruning", "option_topk")


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
    if name == "option_topk":
        return OptionTopK(backbone, budget, _strict(OptionTopKConfig, cfg), run_seed)
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


def _load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_methods(cfg: Dict, items: List[Dict], names: List[str], out: Path, resume: bool) -> None:
    budget = Budget(**cfg["budget"])
    t_load = time.perf_counter()
    backbone = build_backbone(cfg["backbone"], items)
    load_s = time.perf_counter() - t_load
    meta = backbone.get_model_metadata()
    for name in names:
        method = build_method(name, cfg["methods"][name], backbone, budget, cfg["run_seed"])
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        for w in items[: cfg.get("warmup_questions", 0)]:
            method.run(w)
        path = out / f"per_question_{name}.jsonl"
        wanted = {it["question_id"] for it in items}
        done = {r["question_id"]: r for r in (_load_records(path) if resume else []) if r["question_id"] in wanted}
        with open(path, "w", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
            f.flush()
            for i, it in enumerate(items):
                if it["question_id"] in done:
                    continue
                res = method.run(it)
                led = res.ledger
                rec = {
                    "question_id": it["question_id"], "category": it.get("category"), "gold": it["answer"],
                    "answers": res.answers, "pass": res.pass_at_n(it["answer"]), "unique": res.unique_answers(),
                    "dup_token_frac": res.duplicate_token_fraction(), "majority": res.majority_answer(),
                    "rollouts": led["rollouts"], "decode_tokens": led["decode_tokens"],
                    "aux_tokens": led["aux_tokens"], "total_tokens": led["total_tokens"],
                    "prompt_tokens": led["prompt_tokens"], "rounds": res.rounds, "stop_reason": res.stop_reason,
                    "violations": res.metadata.get("violations", []), "wall_time_s": res.wall_time_s,
                }
                rec.update(res.metadata.get("record_extra", {}))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{name}] {i + 1}/{len(items)} pass={rec['pass']} tokens={rec['total_tokens']} "
                      f"t={rec['wall_time_s']:.1f}s", flush=True)
        run_meta = {"method": name, "method_config": cfg["methods"][name], "backbone": meta, "model_load_s": load_s,
                    "environment": environment_metadata(), "peak_memory": peak_memory_mb(), "commit": git_commit(),
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / f"run_meta_{name}.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")


def method_summary(recs: List[Dict]) -> Dict:
    n = len(recs) or 1
    lat = [r["wall_time_s"] * 1000 for r in recs]
    dups = [r["dup_token_frac"] for r in recs if r["dup_token_frac"] is not None]
    total_wall = sum(r["wall_time_s"] for r in recs)
    return {
        "n": len(recs),
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
        "null_completions": sum(a is None for r in recs for a in r["answers"]),
        "latency_median_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "latency_max_ms": max(lat) if lat else None,
        "decode_tokens_per_s": (sum(r["decode_tokens"] for r in recs) / total_wall) if total_wall > 0 else None,
        "budget_violations": sum(1 for r in recs if r["violations"]),
        "stop_reasons": {s: sum(1 for r in recs if r["stop_reason"] == s) for s in sorted({r["stop_reason"] for r in recs})},
    }


def topk_controls(topk: Dict, ref: Dict) -> Dict:
    """Gold-in-top-K rates from stored root option probabilities.

    ``matched``: per question, K = number of distinct answers the reference
    method found on that question (K = 0 counts as a miss).
    """
    common = [q for q in ref if q in topk and topk[q].get("root_probs")]
    curve = {}
    for k in range(1, 11):
        hits = [int(topk[q]["gold"] in sorted(topk[q]["root_probs"], key=lambda a: -topk[q]["root_probs"][a])[:k]) for q in common]
        curve[k] = sum(hits) / len(hits) if hits else None
    matched = []
    for q in common:
        rp = topk[q]["root_probs"]
        k = ref[q]["unique"]
        matched.append(int(k > 0 and topk[q]["gold"] in sorted(rp, key=lambda a: -rp[a])[:k]))
    return {"n": len(common), "gold_in_topk_curve": curve,
            "matched_k_pass": sum(matched) / len(matched) if matched else None,
            "matched_k_pass_list": matched, "common_ids": common}


def summarize(cfg: Dict, items: List[Dict], out: Path) -> Dict:
    budget = Budget(**cfg["budget"])
    order = {it["question_id"]: i for i, it in enumerate(items)}
    cats = {it["question_id"]: it.get("category") for it in items}
    per_method, metas = {}, {}
    for name in cfg["methods"]:
        recs = [r for r in _load_records(out / f"per_question_{name}.jsonl") if r["question_id"] in order]
        if not recs:
            continue
        for r in recs:
            r.setdefault("category", cats.get(r["question_id"]))
        per_method[name] = sorted(recs, key=lambda r: order[r["question_id"]])
        mp = out / f"run_meta_{name}.json"
        metas[name] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
    summary = {"experiment_id": cfg["experiment_id"], "num_questions": len(items), "budget": cfg["budget"],
               "methods": {m: method_summary(r) for m, r in per_method.items()}, "run_meta": metas}
    complete = {m for m, r in per_method.items() if len(r) == len(items)}
    summary["complete_methods"] = sorted(complete)
    by_id = {m: {r["question_id"]: r for r in recs} for m, recs in per_method.items()}
    if "scps" in by_id:
        paired = {}
        for b in by_id:
            if b in ("scps", "option_topk"):
                continue
            common = [q for q in by_id["scps"] if q in by_id[b]]
            paired[b] = {"n": len(common),
                         "pass": paired_bootstrap([by_id["scps"][q]["pass"] for q in common],
                                                  [by_id[b][q]["pass"] for q in common], seed=cfg["run_seed"]),
                         "majority": paired_bootstrap([int(by_id["scps"][q]["majority"] == by_id["scps"][q]["gold"]) for q in common],
                                                      [int(by_id[b][q]["majority"] == by_id[b][q]["gold"]) for q in common],
                                                      seed=cfg["run_seed"])}
        summary["paired_vs_scps"] = paired
        if "option_topk" in by_id:
            ctl = topk_controls(by_id["option_topk"], by_id["scps"])
            ids = ctl.pop("common_ids")
            m = ctl.pop("matched_k_pass_list")
            ctl["scps_vs_matched_k"] = paired_bootstrap([by_id["scps"][q]["pass"] for q in ids], m, seed=cfg["run_seed"])
            summary["option_topk_control"] = ctl
        per_cat = {}
        for q, r in by_id["scps"].items():
            c = r.get("category") or "unknown"
            d = per_cat.setdefault(c, {"n": 0, "scps_pass": 0})
            d["n"] += 1
            d["scps_pass"] += r["pass"]
            for b in by_id:
                if b != "scps" and q in by_id[b]:
                    d[f"{b}_pass"] = d.get(f"{b}_pass", 0) + by_id[b][q]["pass"]
        summary["per_category_pass_counts"] = dict(sorted(per_cat.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    rows = []
    r3 = lambda x: None if x is None else round(x, 3)
    for name, s in summary["methods"].items():
        meta = metas.get(name) or {}
        bb, env, mem = meta.get("backbone", {}), meta.get("environment", {}), meta.get("peak_memory", {})
        framework = ";".join(f"{k}={bb[k]}" for k in bb if k.endswith("_version"))
        framework = (f"python={env['python']}" + (f";{framework}" if framework else "")) if env else None
        if mem.get("cuda_peak_allocated_mb") is not None:
            mem_val, mem_label = mem["cuda_peak_allocated_mb"], "cuda_peak_allocated (this method)"
        else:
            mem_val, mem_label = mem.get("process_peak_rss_mb"), "process_peak_rss" if mem else "not recorded"
        rows.append(make_row(
            experiment_id=cfg["experiment_id"], paper_id=cfg["paper_id"], method=name,
            commit=meta.get("commit"), backbone_id=bb.get("id") or cfg["backbone"]["id"],
            backbone_revision=bb.get("revision"), dataset_id=cfg["dataset"]["id"], split=cfg["dataset"].get("split"),
            seed=f"data={cfg['dataset']['seed']};run={cfg['run_seed']}",
            device=env.get("device_name") or bb.get("device") or env.get("processor"), framework_version=framework,
            dtype=(f"{bb.get('dtype')}" + (f"+{bb['quantization']}" if bb.get("quantization") else "")) if bb else None,
            batch_size=1,
            input_length=None if name == "option_topk" else round(s["mean_prompt_tokens"], 2),
            output_length=round(s["mean_decode_tokens"], 2),
            quality_metric=(f"gold_in_top{cfg['methods'][name].get('k', 8)}_options" if name == "option_topk"
                            else f"portfolio_pass@{budget.max_rollouts}"),
            quality_value=round(s["pass_at_n"], 4),
            latency_median_ms=r3(s["latency_median_ms"]), latency_p95_ms=r3(s["latency_p95_ms"]),
            throughput=r3(s["decode_tokens_per_s"]), peak_memory_mb=r3(mem_val), energy_j=None,
            status=cfg["report_status"] if name in complete else "not reproduced",
            notes=(f"n={s['n']}/{len(items)}; mean_total_tokens={s['mean_total_tokens']:.2f} (decode+aux); "
                   f"unique={s['mean_unique_answers']:.3f}; dup_tok={r3(s['mean_dup_token_frac'])}; "
                   f"majority_acc={s['majority_accuracy']:.4f}; latency=per-question wall clock; "
                   f"throughput=decode tok/s; memory={mem_label}; energy not_measured"),
        ))
    write_csv(rows, str(out / "results.csv"))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", default=None, help="comma-separated subset of the config's methods")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N sampled questions")
    ap.add_argument("--resume", action="store_true", help="keep finished questions in existing per-question files")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary.json and results.csv, run nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(cfg["dataset"])
    if args.limit is not None:
        items = items[: args.limit]
    names = args.methods.split(",") if args.methods else list(cfg["methods"])
    unknown = [n for n in names if n not in cfg["methods"]]
    if unknown:
        raise ValueError(f"methods not in config: {unknown}")
    if not args.summarize_only:
        run_methods(cfg, items, names, out, args.resume)
    summary = summarize(cfg, items, out)
    print(json.dumps({m: {"n": v["n"], "pass": round(v["pass_at_n"], 4), "majority": round(v["majority_accuracy"], 4)}
                      for m, v in summary["methods"].items()}))
    return summary


if __name__ == "__main__":
    main()
