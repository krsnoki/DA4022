"""Prompt-guided baselines on the evaluation split: generate, score, compare with standard and ST-ES.

Uses the same backbone, problems, seed, sampling, token cap and batching as the
standard traces, and changes only the prompt (``baselines/prompt_guided.py``).
Traces are cached and resumable like ``generate.py``. Outputs:

* ``results/<dataset>/prompt_guided.csv``: accuracy, mean generated tokens,
  tokens saved and a paired-bootstrap Δ accuracy vs standard per variant;
* ``results/<dataset>/prompt_guided.meta.json``: trace checksums and settings;
* metrics rows ``st_es_pguided-<dataset>-*`` in ``Performance_Metrics/results/st_es_prompt_guided.csv``.

    python -m Adaptive_Stopping.experiments.st_es.run_prompt_guided --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import yaml

from Adaptive_Stopping.baselines.prompt_guided import VARIANTS, build_guided_messages
from Adaptive_Stopping.common.cached_generation import GenJob, generate_full_traces
from Adaptive_Stopping.experiments.st_es.generate import cache_path
from Adaptive_Stopping.experiments.st_es.io_utils import (
    DEFAULT_RUN_DIR, append_jsonl, environment_metadata, git_commit, limit_gpu_memory, read_jsonl, sha256_file,
    upsert_metric_rows, write_json,
)
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Dataloaders.evaluation.gsm8k_math import load_math_reasoning
from Performance_Metrics.quality.math_answer import SCORER_ID, is_correct

REPO = Path(__file__).resolve().parents[3]
METRICS_CSV = REPO / "Performance_Metrics" / "results" / "st_es_prompt_guided.csv"


def generate_variant(bb, examples, variant, out: Path, cfg: dict, kv_budget: int) -> dict:
    tok = bb.tokenizer
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    by_uid = {ex.uid: ex for ex in examples}
    jobs = [GenJob(uid=ex.uid, prompt_ids=bb.preprocess(build_guided_messages(ex, variant, cfg["prompt_id"])))
            for ex in examples if ex.uid not in done]
    print(f"{variant}: {len(examples)} problems, {len(done)} cached")
    if not jobs:
        return {}
    s = cfg["sampling"]
    sampling = {"do_sample": True, "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"]}

    def on_finish(job):
        ex = by_uid[job.uid]
        append_jsonl(out, {"uid": ex.uid, "dataset_id": ex.dataset_id, "index": ex.index, "question": ex.question,
                           "gold": ex.gold, "gold_format": ex.gold_format, "meta": ex.meta, "prompt_id": variant,
                           "prompt_ids": job.prompt_ids, "gen_ids": job.gen_ids,
                           "gen_text": tok.decode(job.gen_ids, skip_special_tokens=False),
                           "n_gen_tokens": len(job.gen_ids), "finish_reason": job.finish_reason, "rounds": job.rounds})

    return generate_full_traces(bb.model, jobs, eos_ids=bb.eos_token_ids, pad_id=tok.pad_token_id,
                                max_new_tokens=cfg["max_new_tokens"], seed=cfg["seed"], sampling=sampling,
                                kv_token_budget=kv_budget, on_finish=on_finish)


def paired_ci(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--kv-token-budget", type=int, default=52_000)
    ap.add_argument("--score-only", action="store_true", help="do not generate; score the cached traces")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    ds, n, off, seed, cap = ev["dataset_id"], ev["num_samples"], ev["offset"], cfg["seed"], cfg["max_new_tokens"]
    examples = load_math_reasoning(ds, n, seed, off)
    paths = {v: cache_path(args.run_dir, ds, n, off, seed, cap, v) for v in args.variants}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        bb = HFCausalLMBackbone.from_registry(cfg["backbone_id"], device="cuda")
        limit_gpu_memory(0.85)
        for v in args.variants:
            t0 = time.time()
            stats = generate_variant(bb, examples, v, paths[v], cfg, args.kv_token_budget)
            write_json(paths[v].with_suffix(".manifest.json"), {
                "cache_file": paths[v].name, "sha256": sha256_file(paths[v]),
                "num_records": sum(1 for _ in read_jsonl(paths[v])), "variant": v, "paper_name": VARIANTS[v][0],
                "dataset_id": ds, "num_samples": n, "offset": off, "seed": seed, "max_new_tokens": cap,
                "sampling": cfg["sampling"], "backbone": bb.get_model_metadata(), "environment": environment_metadata(),
                "throughput_stats": stats, "wall_s": time.time() - t0})

    # ---------------- score ----------------
    std_path = cache_path(args.run_dir, ds, n, off, seed, cap, cfg["prompt_id"])
    std = {r["uid"]: r for r in read_jsonl(std_path)}
    std_ok = {u: int(is_correct(r["gen_text"], r["gold"], r["gold_format"])) for u, r in std.items()}
    std_tok = sum(r["n_gen_tokens"] for r in std.values()) / len(std)
    rows = [{"variant": "standard", "paper_name": "Standard", "n": len(std), "accuracy": sum(std_ok.values()) / len(std),
             "mean_tokens": std_tok, "saved_pct": 0.0, "acc_diff_vs_standard": "", "ci95_low": "", "ci95_high": "",
             "hit_token_cap": sum(r["finish_reason"] == "length" for r in std.values())}]
    for v in args.variants:
        recs = {r["uid"]: r for r in read_jsonl(paths[v])}
        uids = sorted(set(recs) & set(std))
        if len(uids) < len(std):
            print(f"{v}: only {len(uids)}/{len(std)} problems cached; scoring those")
        ok = {u: int(is_correct(recs[u]["gen_text"], recs[u]["gold"], recs[u]["gold_format"])) for u in uids}
        tok_mean = sum(recs[u]["n_gen_tokens"] for u in uids) / len(uids)
        d, lo, hi = paired_ci([ok[u] for u in uids], [std_ok[u] for u in uids])
        rows.append({"variant": v, "paper_name": VARIANTS[v][0], "n": len(uids), "accuracy": sum(ok.values()) / len(uids),
                     "mean_tokens": tok_mean, "saved_pct": 100 * (1 - tok_mean / std_tok),
                     "acc_diff_vs_standard": d, "ci95_low": lo, "ci95_high": hi,
                     "hit_token_cap": sum(recs[u]["finish_reason"] == "length" for u in uids)})
    out = REPO / "Adaptive_Stopping" / "experiments" / "st_es" / "results" / ds
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompt_guided.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    write_json(out / "prompt_guided.meta.json", {
        "variants": {v: VARIANTS[v] for v in args.variants}, "scorer": SCORER_ID,
        "traces": {v: {"file": paths[v].name, "sha256": sha256_file(paths[v])} for v in args.variants if paths[v].exists()},
        "standard_traces": std_path.name, "ci": "paired bootstrap, 2000 resamples, seed 0"})

    env = environment_metadata()
    metric_rows = []
    for r in rows[1:]:
        metric_rows.append({
            "experiment_id": f"st_es_pguided-{ds}-{r['variant'].split('.')[0]}", "paper_id": cfg["paper_id"],
            "method": f"prompt-guided baseline {r['paper_name']} (no early stopping)", "commit": git_commit(REPO),
            "backbone_id": cfg["backbone_id"], "backbone_revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            "dataset_id": ds, "split": "test", "seed": seed, "device": env.get("gpu", "unknown"),
            "framework_version": f"torch {env.get('torch')}; transformers {env.get('transformers')}", "dtype": "bfloat16",
            "batch_size": "", "input_length": "", "output_length": round(r["mean_tokens"], 1),
            "quality_metric": "pass@1 accuracy (Math-Verify, 1 sample)", "quality_value": round(r["accuracy"], 4),
            "latency_median_ms": "", "latency_p95_ms": "", "throughput": "", "peak_memory_mb": "", "energy_j": "",
            "status": "partial",
            "notes": f"cached generation n={r['n']}, cap {cap}; no runtime measured; tokens saved vs standard "
                     f"{r['saved_pct']:.1f}%; acc diff {r['acc_diff_vs_standard']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"})
    upsert_metric_rows(METRICS_CSV, metric_rows, id_prefix=f"st_es_pguided-{ds}-")
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
