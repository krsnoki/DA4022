"""Draw n samples per problem with a given remasking policy (Sec. 4.1, 4.2, Fig. 2-9).

Writes one JSON line per problem (resumable): completions, extracted answers,
correctness, per-sample NFEs, wall-clock time and peak memory.

Example (Fig. 2, TLC on GSM8K with LLaDA):

    python Adaptive_Stopping/experiments/tempered_remasking/run_sampling.py \
        --model llada --dataset gsm8k --preset tlc --n-samples 64 \
        --out results/tempered_remasking/llada_gsm8k_tlc.jsonl

Post-RL evaluation (Sec. 4.3): ``--preset ct --token-temperature 0 --threshold 0.8 --n-samples 1``.
"""

from __future__ import annotations

import argparse
import time

import torch

from common import (
    append_jsonl,
    environment_metadata,
    grade,
    load_config,
    load_problems,
    policy_from_preset,
    read_jsonl,
    sampler_config,
)
from tempered_remasking import generate
from tempered_remasking.models import load_hf_dllm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["llada", "dream"], required=True)
    ap.add_argument("--model-path", default=None, help="override checkpoint (e.g. an RL-trained policy)")
    ap.add_argument("--dataset", choices=["gsm8k", "math500", "humaneval", "mbpp"], required=True)
    ap.add_argument("--preset", choices=["ar", "lc", "random", "tlc", "ct", "tct"], required=True)
    ap.add_argument("--tpos", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--token-temperature", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    policy = policy_from_preset(cfg, args.preset, args.tpos, args.threshold)
    scfg = sampler_config(cfg, token_temperature=args.token_temperature, steps=args.steps)
    mcfg = cfg["models"][args.model]
    handle = load_hf_dllm(args.model, args.model_path or mcfg["model_id"], mcfg.get("revision"), args.device,
                          getattr(torch, cfg["sampling"].get("dtype", "bfloat16")))
    problems = load_problems(cfg, args.dataset, args.limit)
    done = {r["id"] for r in read_jsonl(args.out)}
    run_meta = {"env": environment_metadata(), "policy": policy.config(), "sampler": scfg.__dict__,
                "model_id": handle.model_id, "revision": handle.revision, "dataset": args.dataset,
                "preset": args.preset, "n_samples": args.n_samples, "seed": args.seed,
                "batch_size": args.batch_size, "dtype": cfg["sampling"].get("dtype", "bfloat16")}
    print(run_meta)

    for idx, prob in enumerate(problems):
        if prob["id"] in done:
            continue
        gen = torch.Generator(device=args.device).manual_seed(args.seed * 1_000_003 + idx)
        prompt = handle.encode_chat(prob["prompt"], torch.device(args.device))
        samples = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        remaining = args.n_samples
        while remaining > 0:
            b = min(args.batch_size, remaining)
            res = generate(handle.model_fn, prompt.unsqueeze(0).expand(b, -1), policy, scfg,
                           handle.mask_token_id, handle.eos_token_ids, gen)
            for j in range(b):
                text = handle.decode(res.generated[j].tolist())
                ans, ok = grade(args.dataset, text, prob)
                samples.append({"text": text, "answer": ans, "correct": bool(ok), "nfe": int(res.nfe[j]),
                                "nfe_paper": int(res.nfe_paper_convention[j]), "stop_reason": res.stop_reasons[j]})
            remaining -= b
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rec = {"id": prob["id"], "question": prob["question"], "reference": prob["reference"], "samples": samples,
               "wall_time_s": wall, "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
               "run": run_meta}
        append_jsonl(args.out, rec)
        acc = sum(s["correct"] for s in samples) / len(samples)
        print(f"[{idx + 1}/{len(problems)}] {prob['id']} acc={acc:.3f} mean_nfe={sum(s['nfe'] for s in samples) / len(samples):.1f} t={wall:.1f}s")


if __name__ == "__main__":
    main()
