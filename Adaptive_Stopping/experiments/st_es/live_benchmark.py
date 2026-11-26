"""Live (measured) latency and peak memory: standard generation vs. ST-ES online.

Runs each request at batch size 1 on the first ``--num-samples`` examples of the
evaluation subset (same seeded order as the cached traces). Policies are
interleaved per example (standard, then each ST-ES/step-count config) so
thermal or clock drift affects them alike. Each request gets ``seed`` reset, so
with deterministic kernels an ST-ES run follows the standard run's tokens up
to its stop point.

Measured per request (wall clock, CUDA-synchronised): end-to-end latency
(prompt in -> final answer out, including tagging and the forced answer), time
to first token, reasoning and answer phases, policy (tagger) time, generated
tokens, and peak allocated/reserved CUDA memory (``torch.cuda.max_memory_*``,
reset before each request). Model loading and one warm-up request are excluded.

Example::

    python -m Adaptive_Stopping.experiments.st_es.live_benchmark --dataset-id text.gsm8k.test \\
        --num-samples 30 --max-new-tokens 4096 --selected <calibration>/grid/selected.json --targets 0.9
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from Adaptive_Stopping.baselines.step_count import StepCountStopping
from Adaptive_Stopping.common.forced_answer import answer_text
from Adaptive_Stopping.common.live_generation import generate_with_policy
from Adaptive_Stopping.common.replay import replay_policy
from Adaptive_Stopping.common.steps import segment_token_ids
from Adaptive_Stopping.experiments.st_es.io_utils import (
    DEFAULT_RUN_DIR, append_jsonl, environment_metadata, limit_gpu_memory, run_relative, write_json,
)
from Adaptive_Stopping.methods.step_tagging.policy import StepTagEarlyStopping
from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone
from Dataloaders.evaluation.gsm8k_math import build_messages, load_math_reasoning
from Performance_Metrics.quality.math_answer import SCORER_ID, is_correct


def make_policy(cfg: dict | None):
    if cfg is None:
        return None
    if cfg["kind"] == "st_es":
        return StepTagEarlyStopping(cfg["tag"], cfg["delta"])
    if cfg["kind"] == "step_count":
        return StepCountStopping(cfg["k"])
    raise ValueError(cfg)


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-id", default="math_cot_boxed.v1")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--selected", type=Path, required=True, help="selected.json from the calibration replay")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9])
    ap.add_argument("--kinds", nargs="+", default=["st_es"], choices=["st_es", "step_count"])
    ap.add_argument("--backbone-id", default="text.deepseek-r1-distill-qwen-1.5b")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--only-uids", nargs="*", default=None, help="diagnostics: run only these example uids")
    args = ap.parse_args(argv)

    sel = json.loads(args.selected.read_text(encoding="utf-8"))
    policies = [("standard", None)]
    for kind in args.kinds:
        for t in args.targets:
            row = sel["selected"][kind].get(str(t))
            if row is None:
                print(f"no {kind} config reached target {t}; skipped")
                continue
            cfg = {"kind": row["kind"], "tag": row["tag"], "delta": row["delta"], "k": row["k"]}
            policies.append((f"{kind}@{t:.2f}:{row['key']}", cfg))

    out_dir = args.out_dir or (args.run_dir / "live" / args.dataset_id / f"n{args.num_samples}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = out_dir / "requests.jsonl"
    if req_path.exists():
        req_path.unlink()

    examples = load_math_reasoning(args.dataset_id, args.num_samples, args.seed, 0)
    if args.only_uids:
        examples = [ex for ex in examples if ex.uid in set(args.only_uids)]
    bb = HFCausalLMBackbone.from_registry(args.backbone_id, device="cuda")
    limit_gpu_memory(0.85)
    tok, model = bb.tokenizer, bb.model
    gc = model.generation_config
    sampling = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": gc.top_k}
    eos, pad = bb.eos_token_ids, tok.pad_token_id

    # warm-up (excluded): short generation on the first prompt
    warm = bb.preprocess(build_messages(examples[0], args.prompt_id))
    generate_with_policy(model, tok, warm, None, max_new_tokens=32, sampling=sampling, seed=0, eos_ids=eos, pad_id=pad)

    t_all = time.time()
    for i, ex in enumerate(examples):
        prompt = bb.preprocess(build_messages(ex, args.prompt_id))
        for name, cfg in policies:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            res = generate_with_policy(model, tok, prompt, make_policy(cfg), max_new_tokens=args.max_new_tokens,
                                       sampling=sampling, seed=args.seed, eos_ids=eos, pad_id=pad)
            res["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            res["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
            text = tok.decode(res["gen_ids"], skip_special_tokens=False)
            if res["stopped"]:
                text = answer_text(tok, res["answer_ids"])
            res["correct"] = is_correct(text, ex.gold, ex.gold_format)
            if cfg is not None:
                # audit: replaying the same policy offline on the live tokens must stop at the same token
                rep = replay_policy(make_policy(cfg), segment_token_ids(res["gen_ids"], tok), len(res["gen_ids"]))
                res["replay_cut"] = rep.cut_tokens if rep.stopped else None
                res["replay_agrees"] = rep.stopped == res["stopped"] and (not rep.stopped or rep.cut_tokens == res["cut"])
            rec = {"uid": ex.uid, "policy": name, "config": cfg, "prompt_tokens": len(prompt), **res,
                   "answer_tokens": len(res["answer_ids"]), "scored_text_tail": text[-300:]}
            append_jsonl(req_path, rec)
            print(f"[{i + 1}/{len(examples)}] {name:40s} tok={res['n_generated']:5d} lat={res['latency_s']:.1f}s "
                  f"ok={int(res['correct'])} stop={res['stopped']} peak={res['peak_allocated_mb']:.0f}MB")

    # summary
    rows = [json.loads(l) for l in req_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = {"dataset_id": args.dataset_id, "num_samples": len(examples), "seed": args.seed,
               "max_new_tokens": args.max_new_tokens, "sampling": sampling, "batch_size": 1, "warmup_requests": 1,
               "timing_scope": "prompt token IDs on GPU -> final answer tokens; excludes tokenisation and model load; "
                               "includes tagging and the forced answer; CUDA-synchronised wall clock",
               "memory_scope": "torch.cuda.max_memory_allocated / max_memory_reserved, reset per request",
               "scorer": SCORER_ID, "backbone": bb.get_model_metadata(), "environment": environment_metadata(),
               "selected_from": run_relative(args.selected, args.run_dir), "wall_s": time.time() - t_all, "policies": {}}
    for name, cfg in policies:
        rs = [r for r in rows if r["policy"] == name]
        lat = [r["latency_s"] for r in rs]
        toks = [r["n_generated"] for r in rs]
        summary["policies"][name] = {
            "config": cfg, "n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "mean_generated_tokens": statistics.mean(toks),
            "latency_mean_s": statistics.mean(lat), "latency_median_s": statistics.median(lat),
            "latency_p95_s": percentile(lat, 0.95),
            "ttft_median_s": statistics.median([r["ttft_s"] for r in rs if r["ttft_s"] is not None]),
            "decode_tokens_per_s": sum(toks) / sum(lat),
            "policy_time_total_s": sum(r["policy_time_s"] for r in rs),
            "policy_time_per_step_ms": 1000 * sum(r["policy_time_s"] for r in rs) / max(1, sum(r["steps_seen"] or 0 for r in rs)),
            "stop_rate": sum(r["stopped"] for r in rs) / len(rs),
            "replay_agreement": (sum(r.get("replay_agrees", False) for r in rs) / len(rs)) if cfg else None,
            "peak_allocated_mb_max": max(r["peak_allocated_mb"] for r in rs),
            "peak_reserved_mb_max": max(r["peak_reserved_mb"] for r in rs),
        }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary["policies"], indent=1))


if __name__ == "__main__":
    main()
