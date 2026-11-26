"""Generate and cache full reasoning traces (standard inference, no stopping).

These traces are (1) the *standard* baseline and (2) the input of the offline
ST-ES / step-count replay. Output: one JSONL record per example plus a
manifest with the config, environment, throughput and the JSONL sha256.
Caches are written outside the repository (``ST_ES_RUN_DIR``) and never committed.

Example (from the repository root)::

    python -m Adaptive_Stopping.experiments.st_es.generate --dataset-id text.gsm8k.train \\
        --num-samples 150 --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Reclaim cached allocator blocks early: the growing KV cache fragments memory, and on Windows the
# overflow silently spills into (slow) shared system memory instead of raising OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")

import torch  # noqa: E402  (after the allocator setting)

from Adaptive_Stopping.common.cached_generation import GenJob, generate_full_traces
from Adaptive_Stopping.experiments.st_es.io_utils import (
    DEFAULT_RUN_DIR, append_jsonl, environment_metadata, limit_gpu_memory, read_jsonl, sha256_file, write_json,
)
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Dataloaders.evaluation.gsm8k_math import build_messages, load_math_reasoning


def cache_path(run_dir: Path, dataset_id: str, num_samples: int, offset: int, seed: int, cap: int,
               prompt_id: str) -> Path:
    return Path(run_dir) / "traces" / dataset_id / f"{prompt_id}_n{num_samples}_o{offset}_s{seed}_cap{cap}.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone-id", default="text.deepseek-r1-distill-qwen-1.5b")
    ap.add_argument("--prompt-id", default="math_cot_boxed.v1")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None, help="default: the model's generation_config value")
    # Defaults sized for an 8 GB GPU holding the 3.55 GB bf16 1.5B model. On Windows (WDDM), going over the
    # dedicated memory does not raise OOM: it spills into shared system memory and generation slows down
    # several-fold, so the budget leaves headroom. KV cost of this model: ~28 KB per token.
    # Each round decodes into a preallocated StaticCache (no per-step concatenation, so no allocator
    # fragmentation); the budget bounds that cache: batch * (context + chunk) <= kv_token_budget.
    # `expandable_segments` is not supported on Windows.
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--kv-token-budget", type=int, default=52_000)  # measured peak ~4.8 GB at 36k (cap 6.8 GB)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.85,
                    help="cap for PyTorch's caching allocator (fraction of GPU memory); 0 = no cap")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    examples = load_math_reasoning(args.dataset_id, args.num_samples, args.seed, args.offset)
    out = cache_path(args.run_dir, args.dataset_id, args.num_samples, args.offset, args.seed,
                     args.max_new_tokens, args.prompt_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out)} if out.exists() else set()
    print(f"{len(examples)} examples, {len(done)} already cached in {out}")
    if {ex.uid for ex in examples} <= done and out.with_suffix(".manifest.json").exists():
        print("all traces cached and manifest present; nothing to do")  # keep the original manifest/provenance
        return

    bb = HFCausalLMBackbone.from_registry(args.backbone_id, device=args.device)
    if args.device.startswith("cuda"):
        limit_gpu_memory(args.gpu_memory_fraction)
    tok = bb.tokenizer
    gen_cfg = bb.model.generation_config
    top_k = args.top_k if args.top_k is not None else gen_cfg.top_k
    sampling = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": top_k}
    by_uid = {ex.uid: ex for ex in examples}
    jobs = [GenJob(uid=ex.uid, prompt_ids=bb.preprocess(build_messages(ex, args.prompt_id)))
            for ex in examples if ex.uid not in done]

    def on_finish(job: GenJob):
        ex = by_uid[job.uid]
        append_jsonl(out, {
            "uid": ex.uid, "dataset_id": ex.dataset_id, "index": ex.index, "question": ex.question,
            "gold": ex.gold, "gold_format": ex.gold_format, "meta": ex.meta, "prompt_id": args.prompt_id,
            "prompt_ids": job.prompt_ids, "gen_ids": job.gen_ids,
            "gen_text": tok.decode(job.gen_ids, skip_special_tokens=False),
            "n_gen_tokens": len(job.gen_ids), "finish_reason": job.finish_reason, "rounds": job.rounds,
        })

    t0 = time.time()
    stats = generate_full_traces(
        bb.model, jobs, eos_ids=bb.eos_token_ids, pad_id=tok.pad_token_id, max_new_tokens=args.max_new_tokens,
        seed=args.seed, sampling=sampling, chunk_tokens=args.chunk_tokens, max_batch=args.max_batch,
        kv_token_budget=args.kv_token_budget, on_finish=on_finish,
    ) if jobs else {}

    manifest = {
        "cache_file": out.name, "sha256": sha256_file(out), "num_records": sum(1 for _ in read_jsonl(out)),
        "dataset_id": args.dataset_id, "num_samples": args.num_samples, "offset": args.offset, "seed": args.seed,
        "prompt_id": args.prompt_id, "max_new_tokens": args.max_new_tokens, "sampling": sampling,
        "batching": {"chunk_tokens": args.chunk_tokens, "max_batch": args.max_batch,
                     "kv_token_budget": args.kv_token_budget,
                     "prefill": "per-row, assembled into a left-padded StaticCache per round",
                     "gpu_memory_fraction": args.gpu_memory_fraction,
                     "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")},
        "backbone": bb.get_model_metadata(), "environment": environment_metadata(),
        "throughput_stats": {**stats, "note": "batched cached generation; throughput only, not request latency"},
        "records_reused_from_earlier_run": len(done),  # >0 means this run resumed an interrupted one
        "started_unix": t0,
    }
    write_json(out.with_suffix(".manifest.json"), manifest)
    print({k: manifest[k] for k in ("cache_file", "sha256", "num_records")}, stats)


if __name__ == "__main__":
    main()
