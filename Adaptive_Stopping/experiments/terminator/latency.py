"""Online latency benchmark (step E): batch size 1, token-by-token HF decoding
with a KV cache, run once without stopping and once with TERMINATOR.

The probe layer keeps its own KV cache, so each thinking token costs one base
step plus one probe-layer step on the newest hidden state. Timings use
torch.cuda.synchronize at the start and end of each problem; probe time is
measured with extra synchronisation around each probe call.

Example:
    python Adaptive_Stopping/experiments/terminator/latency.py --method terminator \
        --probe probe.pt --n 50 --out latency_terminator.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from Dataloaders.evaluation.math_reasoning import extract_boxed, is_equiv, load_math500  # noqa: E402
from Adaptive_Stopping.methods.learned_exit.terminator.generation import (  # noqa: E402
    build_prompt_ids,
    generate_with_exit,
)
from Adaptive_Stopping.methods.learned_exit.terminator.policy import TerminatorPolicy  # noqa: E402
from Adaptive_Stopping.methods.learned_exit.terminator.probe import TerminatorProbe  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["no_stop", "terminator"], required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-think", type=int, default=6000)
    ap.add_argument("--max-answer", type=int, default=2048)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    probe = policy = None
    if args.method == "terminator":
        probe = TerminatorProbe(model)
        probe.load_state_dict(torch.load(args.probe, map_location="cpu"))
        probe = probe.float().to(device).eval()
        policy = TerminatorPolicy(args.threshold, args.window)
    base_mem = torch.cuda.memory_allocated(device) / 2**20

    examples = load_math500(n=args.n, seed=args.data_seed)
    # warm-up so CUDA kernels and allocator are ready before timing
    generate_with_exit(model, tok, build_prompt_ids(tok, "What is 1+1?"), probe=probe,
                       policy=None, max_think=32, max_answer=8)

    t_start = time.time()
    rows = []
    with open(args.out, "w") as f:
        for e in examples:
            if args.deadline_s and time.time() - t_start > args.deadline_s:
                print("deadline reached", flush=True)
                break
            seed = (args.seed * 1000003 + zlib.crc32(e.uid.encode())) % (2**31 - 1)
            torch.cuda.reset_peak_memory_stats(device)
            res = generate_with_exit(
                model, tok, build_prompt_ids(tok, e.problem), probe=probe, policy=policy,
                max_think=args.max_think, max_answer=args.max_answer, seed=seed,
                time_probe=probe is not None,
            )
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            n_think, n_ans = len(res["think_ids"]), len(res["answer_ids"])
            row = {
                "uid": e.uid,
                "method": args.method,
                "threshold": args.threshold if probe is not None else None,
                "window": args.window if probe is not None else None,
                "budget_tokens": None,
                "seed": args.seed,
                "correct": bool(is_equiv(extract_boxed(res["answer_text"]), e.answer)),
                "thinking_tokens": n_think,
                "answer_tokens": n_ans,
                "stopped_early": res["stopped_early"],
                "latency_s": res["total_s"],
                "peak_mem_mb": peak,
                "label_answer_pos": None,
                "exit_pos": res["exit_pos"],
                "think_s": res["think_s"],
                "probe_s": res["probe_s"],
                "think_end": res["think_end"],
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{len(rows)} {e.uid} think {n_think} ans {n_ans} {res['total_s']:.1f}s "
                  f"{row['think_end']} correct {row['correct']}", flush=True)

    lat = [r["latency_s"] for r in rows]
    toks = sum(r["thinking_tokens"] + r["answer_tokens"] for r in rows)
    think_tok = sum(r["thinking_tokens"] for r in rows)
    summary = {
        "method": args.method,
        "threshold": args.threshold if probe is not None else None,
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "mean_thinking_tokens": think_tok / len(rows),
        "stop_rate": sum(r["stopped_early"] for r in rows) / len(rows),
        "latency_mean_s": statistics.mean(lat),
        "latency_median_s": statistics.median(lat),
        "latency_p95_s": pct(lat, 0.95),
        "tokens_per_s": toks / sum(lat),
        "probe_ms_per_token": (1000 * sum(r["probe_s"] for r in rows) / think_tok
                               if probe is not None and think_tok else None),
        "peak_mem_mb_max": max(r["peak_mem_mb"] for r in rows),
        "weights_mem_mb": base_mem,
        "probe_params": sum(p.numel() for p in probe.parameters()) if probe is not None else 0,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(summary, indent=2))
    with open(Path(args.out).with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
