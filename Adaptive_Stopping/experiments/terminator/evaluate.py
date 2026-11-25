"""Offline evaluation of TERMINATOR and baselines on MATH-500 (step D).

The probe is causal, so its prediction at thinking token t depends only on
tokens <= t. The exit point TERMINATOR would choose during generation can
therefore be read off the probe probabilities of the full no-stop trace.
Only the answer has to be generated again:

    prompt + thinking[:exit] + exit message + </think>  ->  answer

Methods (same traces, same sampling settings):
- no_stop:       the original trace (thinking capped at 6000 tokens)
- terminator:    exit from the sliding-window rule, for several thresholds
- fixed_budget:  every trace cut at the same token count, equal to the mean
                 thinking length of TERMINATOR at the reference threshold
- oracle:        cut right after the labelled first appearance of the answer
If a method does not cut a trace, the no-stop result is reused, which is what
online generation with the same sample would produce.

Outputs in --out-dir:
  per_problem.jsonl      test rows, one per (uid, method, threshold/budget)
  per_problem_val.jsonl  the same for the validation traces
  summary.csv            accuracy, thinking tokens, compression, stop rate
  threshold_selection.json

With two GPUs, run one process per GPU with --num-shards 2 --shard 0/1, then
once more with --merge. Generated answers are appended to
OUT/cache.shardK.jsonl after every chunk; an interrupted shard (for example
after --deadline-s) can be continued with --resume-cache.

Example:
    python Adaptive_Stopping/experiments/terminator/evaluate.py \
        --traces traces_math500.jsonl traces_train.jsonl --labels labels.jsonl \
        --probs probs_test.jsonl probs_val.jsonl --out-dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

T0 = time.time()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from Dataloaders.evaluation.math_reasoning import extract_boxed, is_equiv  # noqa: E402
from Adaptive_Stopping.methods.learned_exit.terminator.policy import exit_index  # noqa: E402

FIELDS = ["uid", "method", "threshold", "window", "budget_tokens", "seed", "correct",
          "thinking_tokens", "answer_tokens", "stopped_early", "latency_s", "peak_mem_mb",
          "label_answer_pos", "exit_pos"]


def load_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def make_row(tr, method, cut, label_pos, threshold=None, window=None, budget=None):
    """Row for a trace. cut is the number of thinking tokens kept, or None
    when the method does not stop this trace (no-stop result is reused)."""
    return {
        "uid": tr["uid"],
        "method": method,
        "threshold": threshold,
        "window": window,
        "budget_tokens": budget,
        "seed": tr["seed"],
        "correct": bool(tr["correct"]),
        "thinking_tokens": len(tr["think_ids"]) if cut is None else cut,
        "answer_tokens": len(tr["answer_ids"]),
        "stopped_early": cut is not None,
        "latency_s": None,
        "peak_mem_mb": None,
        "label_answer_pos": label_pos,
        "exit_pos": cut,
    }


def plan_rows(traces, labels, probs, thresholds, window, budget):
    """Rows for all methods; rows with stopped_early need a new answer."""
    rows = []
    for tr in traces:
        lab = labels.get(tr["uid"], {}).get("label_pos")
        n = len(tr["think_ids"])
        rows.append(make_row(tr, "no_stop", None, lab))
        p = probs.get(tr["uid"])
        for t in thresholds:
            cut = exit_index(p, t, window) if p is not None else None
            rows.append(make_row(tr, "terminator", cut, lab, threshold=t, window=window))
        if budget is not None:
            cut = budget if n > budget else None
            rows.append(make_row(tr, "fixed_budget", cut, lab, budget=budget))
        if lab is not None:
            cut = lab + 1 if lab + 1 < n else None
            rows.append(make_row(tr, "oracle", cut, lab))
        else:
            rows.append(make_row(tr, "oracle", None, None))
    return rows


class DeadlineReached(Exception):
    pass


class Cache:
    """Generated answers keyed by (kind, uid, cut), appended to a JSONL file
    after every chunk so an interrupted run can be resumed."""

    def __init__(self, path, resume_paths=()):
        self.items = {}
        for p in list(resume_paths) + ([path] if path else []):
            if p and Path(p).exists():
                for e in load_jsonl([p]):
                    self.items[(e["kind"], e["uid"], e["cut"])] = e
        self.f = open(path, "a") if path else None
        if self.items:
            print(f"cache: {len(self.items)} entries loaded", flush=True)

    def get(self, kind, uid, cut):
        return self.items.get((kind, uid, cut))

    def add(self, e):
        self.items[(e["kind"], e["uid"], e["cut"])] = e
        if self.f:
            self.f.write(json.dumps(e) + "\n")

    def flush(self):
        if self.f:
            self.f.flush()


def run_chunks(keys, make_prompt, max_tokens, runner, args, on_result):
    """Generate one answer per key, args.chunk prompts at a time. Keys are
    sorted so prompts of the same trace are adjacent (prefix caching)."""
    eos_id = runner.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for i in range(0, len(keys), args.chunk):
        if args.deadline_s and time.time() - T0 > args.deadline_s:
            raise DeadlineReached(f"{len(keys) - i} of {len(keys)} prompts left")
        part = keys[i : i + args.chunk]
        prompts, seeds = zip(*[make_prompt(k) for k in part])
        t = time.time()
        outs = runner.generate(list(prompts), max_tokens, list(seeds), [eos_id],
                               args.temperature, args.top_p, args.top_k)
        for k, (ans, stopped) in zip(part, outs):
            if ans and ans[-1] == eos_id:
                ans = ans[:-1]
            on_result(k, ans, stopped)
        print(f"  {i + len(part)}/{len(keys)} done, chunk {time.time() - t:.0f}s, "
              f"elapsed {time.time() - T0:.0f}s", flush=True)


def extend_truncated(traces, runner, exit_seq, args, cache):
    """Continue no-stop answers that hit the trace-time answer cap, so every
    method gets the same answer budget (args.max_answer)."""
    from generate_traces import request_seed

    tok = runner.tokenizer
    think_end_id = tok.convert_tokens_to_ids("</think>")
    todo = [t for t in traces if t.get("answer_truncated") and len(t["answer_ids"]) < args.max_answer]
    by_uid = {t["uid"]: t for t in todo}

    def make_prompt(uid):
        t = by_uid[uid]
        tail = [think_end_id] if t["think_end"] != "cap" else exit_seq
        return (t["prompt_ids"] + t["think_ids"] + tail + t["answer_ids"],
                request_seed(t["seed"], t["uid"]) + 2)

    def on_result(uid, more, stopped):
        cache.add({"kind": "extend", "uid": uid, "cut": None, "ids": more, "finished": stopped})

    keys = sorted(u for u in by_uid if cache.get("extend", u, None) is None)
    print(f"extending {len(todo)} truncated no-stop answers ({len(keys)} not cached)", flush=True)
    if keys:
        run_chunks(keys, make_prompt, args.max_answer - min(len(t["answer_ids"]) for t in todo),
                   runner, args, on_result)
        cache.flush()
    for t in todo:
        e = cache.get("extend", t["uid"], None)
        t["answer_ids"] = (t["answer_ids"] + e["ids"])[: args.max_answer]
        t["answer_text"] = tok.decode(t["answer_ids"], skip_special_tokens=True)
        t["pred"] = extract_boxed(t["answer_text"])
        t["correct"] = bool(is_equiv(t["pred"], t["gold"]))
        t["answer_truncated"] = not e["finished"]
    return len(todo)


def regenerate(rows, traces_by_uid, runner, exit_seq, args, cache):
    """New answers after early exits. Rows of the same trace with the same
    exit position share one generation."""
    from generate_traces import request_seed

    def make_prompt(key):
        uid, cut = key
        tr = traces_by_uid[uid]
        return (tr["prompt_ids"] + tr["think_ids"][:cut] + exit_seq,
                request_seed(tr["seed"], uid) + 1)

    def on_result(key, ans, stopped):
        text = runner.tokenizer.decode(ans, skip_special_tokens=True)
        gold = traces_by_uid[key[0]]["gold"]
        cache.add({"kind": "answer", "uid": key[0], "cut": key[1], "answer_tokens": len(ans),
                   "correct": bool(is_equiv(extract_boxed(text), gold)), "finished": stopped})

    need = {(r["uid"], r["exit_pos"]) for r in rows if r["stopped_early"]}
    keys = sorted(k for k in need if cache.get("answer", *k) is None)
    print(f"generating {len(need)} distinct answers for {sum(r['stopped_early'] for r in rows)} "
          f"rows ({len(keys)} not cached)", flush=True)
    if keys:
        run_chunks(keys, make_prompt, args.max_answer, runner, args, on_result)
        cache.flush()
    for r in rows:
        if r["stopped_early"]:
            e = cache.get("answer", r["uid"], r["exit_pos"])
            r["correct"] = e["correct"]
            r["answer_tokens"] = e["answer_tokens"]


def group_key(r):
    return (r["method"], r["threshold"], r["window"], r["budget_tokens"])


def summarize(rows):
    base = {r["uid"]: r["thinking_tokens"] for r in rows if r["method"] == "no_stop"}
    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    out = []
    for (method, thr, win, bud), rs in groups.items():
        n = len(rs)
        mean_tok = sum(r["thinking_tokens"] for r in rs) / n
        base_tok = sum(base[r["uid"]] for r in rs) / n
        out.append({
            "method": method, "threshold": thr, "window": win, "budget_tokens": bud, "n": n,
            "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
            "mean_thinking_tokens": round(mean_tok, 1),
            "compression": round(mean_tok / base_tok, 4) if base_tok else None,
            "compression_per_sample": round(
                sum(r["thinking_tokens"] / max(1, base[r["uid"]]) for r in rs) / n, 4),
            "stop_rate": round(sum(r["stopped_early"] for r in rs) / n, 4),
            "mean_answer_tokens": round(sum(r["answer_tokens"] for r in rs) / n, 1),
        })
    order = {"no_stop": 0, "terminator": 1, "fixed_budget": 2, "oracle": 3}
    out.sort(key=lambda s: (order.get(s["method"], 9), s["threshold"] or 0))
    return out


def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in FIELDS}) + "\n")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_shard(args, out):
    from generate_traces import MODEL, REVISION, HFRunner, VLLMRunner

    traces = {r["uid"]: r for r in load_jsonl(args.traces)}
    labels = {r["uid"]: r for r in load_jsonl([args.labels])}
    probs = {r["uid"]: r["probs"] for r in load_jsonl(args.probs)}
    test = [traces[u] for u, lab in labels.items() if lab["fold"] == "test" and u in traces]
    val = [traces[u] for u, lab in labels.items() if lab["fold"] == "val" and u in traces]
    print(f"test {len(test)}, val {len(val)}", flush=True)

    # fixed budget = mean TERMINATOR thinking length at the reference threshold,
    # over all test traces (only needs the probe outputs)
    ref = plan_rows(test, labels, probs, [args.reference_threshold], args.window, None)
    ref_tok = [r["thinking_tokens"] for r in ref if r["method"] == "terminator"]
    budget = int(round(sum(ref_tok) / len(ref_tok)))
    print(f"fixed budget = {budget} tokens", flush=True)

    k, n = args.shard, args.num_shards
    test = sorted(test, key=lambda t: t["uid"])[k::n]
    val = sorted(val, key=lambda t: t["uid"])[k::n]
    print(f"shard {k}/{n}: test {len(test)}, val {len(val)}", flush=True)

    if args.backend == "vllm":
        runner = VLLMRunner(MODEL, REVISION, 0, max_model_len=9216, gpu_mem=0.9)
    else:
        runner = HFRunner(MODEL, REVISION, 0)
    from Adaptive_Stopping.methods.learned_exit.terminator.generation import EXIT_MESSAGE, exit_ids

    exit_seq = exit_ids(runner.tokenizer, EXIT_MESSAGE)
    cache = Cache(args.cache or out / f"cache.shard{k}.jsonl", args.resume_cache)
    test_rows = plan_rows(test, labels, probs, args.thresholds, args.window, budget)
    val_rows = [r for r in plan_rows(val, labels, probs, args.thresholds, args.window, None)
                if r["method"] in ("no_stop", "terminator")]
    try:
        n_ext = extend_truncated(test + val, runner, exit_seq, args, cache)
        # no-stop rows copy correctness from the (possibly extended) traces
        for r in test_rows + val_rows:
            if not r["stopped_early"]:
                r["correct"] = bool(traces[r["uid"]]["correct"])
                r["answer_tokens"] = len(traces[r["uid"]]["answer_ids"])
        regenerate(test_rows + val_rows, traces, runner, exit_seq, args, cache)
    except DeadlineReached as e:
        print(f"deadline reached, {e}; rerun with --resume-cache to continue", flush=True)
        sys.exit(3)

    write_rows(out / f"per_problem.shard{k}.jsonl", test_rows)
    write_rows(out / f"per_problem_val.shard{k}.jsonl", val_rows)
    with open(out / f"meta.shard{k}.json", "w") as f:
        json.dump({"fixed_budget_tokens": budget, "extended_no_stop_answers": n_ext,
                   "n_test": len(test), "n_val": len(val)}, f)


def merge(args, out):
    metas = [json.loads(p.read_text()) for p in sorted(out.glob("meta.shard*.json"))]
    if len(metas) != args.num_shards:
        sys.exit(f"found {len(metas)} finished shards, expected {args.num_shards}")
    test_rows = load_jsonl(sorted(out.glob("per_problem.shard*.jsonl")))
    val_rows = load_jsonl(sorted(out.glob("per_problem_val.shard*.jsonl")))
    write_rows(out / "per_problem.jsonl", test_rows)
    write_rows(out / "per_problem_val.jsonl", val_rows)
    test_sum = summarize(test_rows)
    val_sum = summarize(val_rows)
    write_csv(out / "summary.csv", test_sum)
    write_csv(out / "summary_val.csv", val_sum)

    # threshold selection on val: fewest thinking tokens with accuracy within tolerance
    val_base = next(s for s in val_sum if s["method"] == "no_stop")
    ok = [s for s in val_sum if s["method"] == "terminator"
          and s["accuracy"] >= val_base["accuracy"] - args.val_tolerance]
    chosen = min(ok, key=lambda s: s["mean_thinking_tokens"]) if ok else None
    sel = {"val_no_stop_accuracy": val_base["accuracy"], "tolerance": args.val_tolerance,
           "selected_threshold": chosen["threshold"] if chosen else None,
           "fixed_budget_tokens": metas[0]["fixed_budget_tokens"],
           "reference_threshold": args.reference_threshold,
           "max_answer_tokens": args.max_answer,
           "extended_no_stop_answers": sum(m["extended_no_stop_answers"] for m in metas)}
    with open(out / "threshold_selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    for s in test_sum:
        print(json.dumps(s))
    print(json.dumps(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--probs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--reference-threshold", type=float, default=0.7)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-answer", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--val-tolerance", type=float, default=0.01,
                    help="accuracy drop allowed on val when selecting a threshold")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the traces over this many processes (one per GPU)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true",
                    help="only combine finished shards in --out-dir into the summaries")
    ap.add_argument("--chunk", type=int, default=256, help="prompts per generate call")
    ap.add_argument("--cache", default=None, help="default: OUT/cache.shardK.jsonl")
    ap.add_argument("--resume-cache", nargs="*", default=[],
                    help="cache files of an earlier, interrupted run")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="stop before a new chunk after this many seconds (exit code 3)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.merge:
        if not (args.traces and args.labels and args.probs):
            ap.error("--traces, --labels and --probs are required")
        run_shard(args, out)
    if args.merge or args.num_shards == 1:
        merge(args, out)


if __name__ == "__main__":
    main()


