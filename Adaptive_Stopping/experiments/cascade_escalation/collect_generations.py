"""Stage 1 (GPU): generate answers + confidence scores for one model on all datasets.

For every (model, dataset) it writes ``<out>/generations/<dataset>/<slug>.jsonl.gz``
with one row per query::

    uid, response, n_in, n_out, finish, correct, gpu_s, scores{5 log-prob scores}

and ``<out>/generations/meta/<slug>.json`` (model revision, parameter count,
backend, timings, peak memory, environment). Existing outputs are skipped, so
the command can be re-run after an interruption.

Example::

    python Adaptive_Stopping/experiments/cascade_escalation/collect_generations.py \
        --config Adaptive_Stopping/experiments/cascade_escalation/configs/qwen25_phi35_pool.yaml \
        --model Qwen/Qwen2.5-0.5B-Instruct --out results/cascade --backend auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, environment_metadata, generation_path, load_config, write_jsonl  # noqa: E402

from Adaptive_Stopping.methods.cascade_escalation.scores import token_scores_from_topk  # noqa: E402
from Dataloaders.evaluation import cascade_benchmarks as bench  # noqa: E402


def hub_metadata(model_id: str) -> dict:
    """Resolved commit sha and safetensors parameter count from the Hub (best effort)."""
    out = {}
    try:
        from huggingface_hub import HfApi, get_safetensors_metadata

        out["revision"] = HfApi().model_info(model_id).sha
        out["hub_parameter_count"] = int(sum(get_safetensors_metadata(model_id).parameter_count.values()))
    except Exception as e:  # offline / tiny test models
        out["hub_error"] = repr(e)[:200]
    return out


def make_generator(model_id: str, backend: str, cfg: dict, revision=None):
    if backend in ("auto", "vllm"):
        try:
            from generation import VLLMGenerator

            return VLLMGenerator(model_id, dtype=cfg.get("dtype", "bfloat16"), top_k=cfg.get("top_k_logprobs", 15),
                                 max_model_len=cfg.get("max_model_len", 4096), revision=revision)
        except Exception as e:
            if backend == "vllm":
                raise
            print(f"[collect] vLLM unavailable ({e!r}); falling back to transformers", flush=True)
    from generation import HFGenerator

    return HFGenerator(model_id, dtype=cfg.get("dtype", "bfloat16"), top_k=cfg.get("top_k_logprobs", 15),
                       revision=revision)


def default_batch_size(n_params: int, max_new_tokens: int) -> int:
    if n_params < 2e9:
        bs = 64
    elif n_params < 5e9:
        bs = 32
    else:
        bs = 16
    return bs if max_new_tokens <= 256 else max(4, bs // 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="model id from the config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--num-examples", type=int, default=None, help="override (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for output file names (smoke runs)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    spec = next((m for m in cfg["models"] if m["id"] == args.model), None)
    if spec is None:
        raise SystemExit(f"model {args.model} not in config")
    slug = spec["slug"] + args.tag
    out = Path(args.out)
    datasets = args.datasets or list(cfg["datasets"])

    todo = [d for d in datasets if not generation_path(out, d, slug).exists()]
    if not todo:
        print(f"[collect] {slug}: all outputs exist, skipping", flush=True)
        return 0

    meta = {"model_id": args.model, "slug": slug, "config_num_parameters": spec.get("num_parameters"),
            "license": spec.get("license"), **hub_metadata(args.model), "environment": environment_metadata()}
    t_load = time.perf_counter()
    gen = make_generator(args.model, args.backend, cfg, revision=meta.get("revision"))
    meta["load_s"] = time.perf_counter() - t_load
    meta["backend"] = gen.backend
    if gen.backend == "hf":
        meta["loaded_num_parameters"] = gen.num_parameters()
    n_params = int(spec.get("num_parameters") or meta.get("hub_parameter_count") or meta.get("loaded_num_parameters"))
    meta["num_parameters_used_for_flops"] = n_params

    meta_path = out / "generations" / "meta" / f"{slug}.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        meta["datasets"] = old.get("datasets", {})
    else:
        meta["datasets"] = {}

    import torch

    for ds in todo:
        dcfg = cfg["datasets"][ds]
        n = args.num_examples if args.num_examples is not None else dcfg.get("num_examples")
        records = bench.load_records(ds, num_examples=n, seed=cfg.get("seed", 2026))
        mnt = int(dcfg.get("max_new_tokens", bench.MAX_NEW_TOKENS[ds]))
        bs = args.batch_size or default_batch_size(n_params, mnt)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[collect] {slug} {ds}: {len(records)} prompts, max_new_tokens={mnt}, backend={gen.backend}", flush=True)
        t0 = time.perf_counter()
        res = gen.generate([r["prompt"] for r in records], max_new_tokens=mnt, batch_size=bs)
        wall = time.perf_counter() - t0
        rows = []
        for rec, r in zip(records, res):
            sc = token_scores_from_topk(r["chosen_logprobs"], r["topk_logprobs"], k=cfg.get("top_k_logprobs", 15))
            rows.append({
                "uid": rec["uid"], "response": r["response"], "n_in": r["n_in"], "n_out": r["n_out"],
                "finish": r["finish"], "correct": int(bench.grade(r["response"], rec)),
                "gpu_s": r["gpu_s"], "scores": sc,
            })
        acc = sum(x["correct"] for x in rows) / max(len(rows), 1)
        info = {"n": len(rows), "accuracy": acc, "wall_s": wall, "max_new_tokens": mnt, "batch_size": bs,
                "truncated_frac": sum(x["finish"] == "length" for x in rows) / max(len(rows), 1),
                "mean_n_in": sum(x["n_in"] for x in rows) / max(len(rows), 1),
                "mean_n_out": sum(x["n_out"] for x in rows) / max(len(rows), 1)}
        if torch.cuda.is_available():
            info["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            free, total = torch.cuda.mem_get_info()
            info["device_memory_used_mb"] = (total - free) / 2**20
            if gen.backend == "vllm":
                info["memory_note"] = "vLLM pre-allocates gpu_memory_utilization of device memory; use live run for peak memory"
        meta["datasets"][ds] = info
        write_jsonl(generation_path(out, ds, slug), rows)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[collect] {slug} {ds}: acc={acc:.3f} wall={wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
