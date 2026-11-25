"""Generate one thinking trace per problem with Qwen3 (step A).

Each trace is produced in two phases:
  1. thinking: prompt + <think> -> sample until </think> or --max-think tokens
  2. answer:   prompt + <think> + thinking [+ exit message] + </think> -> answer
When the thinking cap is hit, the exit message is inserted before </think>,
the same way TERMINATOR exits.

Output: one JSON line per problem (see `row` below). The MATH-500 traces are
also the no-stopping baseline.

Example (from the repo root):
    python Adaptive_Stopping/experiments/terminator/generate_traces.py \
        --split math500 --out traces_math500.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Dataloaders.evaluation.math_reasoning import (  # noqa: E402
    extract_boxed,
    is_equiv,
    load_math500,
    load_math_train,
)
from Adaptive_Stopping.methods.learned_exit.terminator.generation import (  # noqa: E402
    EXIT_MESSAGE,
    build_prompt_ids,
    exit_ids,
    special_ids,
)

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def request_seed(base_seed: int, uid: str) -> int:
    return (base_seed * 1000003 + zlib.crc32(uid.encode())) % (2**31 - 1)


def load_examples(split: str, n, seed: int):
    if split == "math500":
        return load_math500(n=n, seed=seed)
    if split == "train":
        return load_math_train(n=n, seed=seed)
    raise ValueError(split)


class VLLMRunner:
    def __init__(self, model, revision, seed, max_model_len, gpu_mem):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            revision=revision,
            dtype="half",
            seed=seed,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts, max_tokens, seeds, stop_ids, temperature, top_p, top_k):
        """Sample continuations for lists of token-id prompts. Returns lists of
        (token_ids, finished_by_stop) in input order."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=s,
                stop_token_ids=stop_ids,
                skip_special_tokens=False,
            )
            for s in seeds
        ]
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
        )
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append((list(c.token_ids), c.finish_reason == "stop"))
        return res


class HFRunner:
    """Fallback when vLLM is not available: batched HF generate with KV cache."""

    def __init__(self, model, revision, seed, batch_size=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.float16, device_map={"": 0}
        ).eval()
        self.batch_size = batch_size
        self.seed = seed

    def generate(self, prompts, max_tokens, seeds, stop_ids, temperature, top_p, top_k):
        torch = self.torch
        pad = self.tokenizer.pad_token_id
        res = []
        torch.manual_seed(seeds[0] if seeds else self.seed)
        for b in range(0, len(prompts), self.batch_size):
            chunk = prompts[b : b + self.batch_size]
            L = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, p in enumerate(chunk):
                ids[i, L - len(p) :] = torch.tensor(p)
                mask[i, L - len(p) :] = 1
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids.cuda(),
                    attention_mask=mask.cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_tokens,
                    eos_token_id=stop_ids,
                    pad_token_id=pad,
                )
            for i in range(len(chunk)):
                toks = out[i, L:].tolist()
                stopped = False
                for j, t in enumerate(toks):
                    if t in stop_ids:
                        toks = toks[: j + 1]
                        stopped = True
                        break
                res.append((toks, stopped))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["math500", "train"], required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-think", type=int, default=6000)
    ap.add_argument("--max-answer", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="stop starting new chunks after this many seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    examples = load_examples(args.split, args.n, args.data_seed)
    examples = examples[args.shard :: args.num_shards]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["uid"] for line in f if line.strip()}
    examples = [e for e in examples if e.uid not in done]
    print(f"[{args.split} shard {args.shard}] {len(examples)} to generate, {len(done)} done",
          flush=True)
    if not examples:
        return

    if args.backend == "vllm":
        runner = VLLMRunner(args.model, args.revision, args.seed,
                            max_model_len=args.max_think + args.max_answer + 1024,
                            gpu_mem=args.gpu_mem)
    else:
        runner = HFRunner(args.model, args.revision, args.seed)
    tok = runner.tokenizer
    think_id, think_end_id = special_ids(tok)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    exit_seq = exit_ids(tok, EXIT_MESSAGE)

    for c0 in range(0, len(examples), args.chunk):
        if args.deadline_s is not None and time.time() - t_start > args.deadline_s:
            print("deadline reached, stopping", flush=True)
            break
        chunk = examples[c0 : c0 + args.chunk]
        prompts = [build_prompt_ids(tok, e.problem) for e in chunk]
        seeds = [request_seed(args.seed, e.uid) for e in chunk]

        t0 = time.time()
        think_out = runner.generate(prompts, args.max_think, seeds, [think_end_id, eos_id],
                                    args.temperature, args.top_p, args.top_k)
        t_think = time.time() - t0

        thinks, ends, prefixes = [], [], []
        for p, (toks, stopped) in zip(prompts, think_out):
            end = "cap"
            if toks and toks[-1] == think_end_id:
                toks, end = toks[:-1], "natural"
            elif toks and toks[-1] == eos_id:
                toks, end = toks[:-1], "eos"
            thinks.append(toks)
            ends.append(end)
            tail = [think_end_id] if end != "cap" else exit_seq
            prefixes.append(p + toks + tail)

        t0 = time.time()
        ans_out = runner.generate(prefixes, args.max_answer, [s + 1 for s in seeds], [eos_id],
                                  args.temperature, args.top_p, args.top_k)
        t_ans = time.time() - t0

        with open(args.out, "a") as f:
            for e, p, th, end, (ans, stopped) in zip(chunk, prompts, thinks, ends, ans_out):
                if ans and ans[-1] == eos_id:
                    ans = ans[:-1]
                ans_text = tok.decode(ans, skip_special_tokens=True)
                pred = extract_boxed(ans_text)
                row = {
                    "uid": e.uid,
                    "split": args.split,
                    "problem": e.problem,
                    "subject": e.subject,
                    "level": e.level,
                    "gold": e.answer,
                    "seed": args.seed,
                    "request_seed": request_seed(args.seed, e.uid),
                    "prompt_ids": p,
                    "think_ids": th,
                    "answer_ids": ans,
                    "think_end": end,
                    "answer_truncated": not stopped,
                    "think_text": tok.decode(th, skip_special_tokens=False),
                    "answer_text": ans_text,
                    "pred": pred,
                    "correct": is_equiv(pred, e.answer),
                    "thinking_tokens": len(th),
                    "answer_tokens": len(ans),
                    "chunk_think_s": t_think,
                    "chunk_answer_s": t_ans,
                    "chunk_size": len(chunk),
                }
                f.write(json.dumps(row) + "\n")
        n_tok = sum(len(t) for t in thinks)
        print(f"[{args.split} shard {args.shard}] chunk {c0 // args.chunk}: {len(chunk)} traces, "
              f"{n_tok} thinking tokens in {t_think:.0f}s ({n_tok / max(t_think, 1e-6):.0f} tok/s), "
              f"answers {t_ans:.0f}s, elapsed {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
