#!/usr/bin/env python
"""Precision diagnostic: is the T=0 AR/DFlash/BASTION output mismatch a bf16 numerics effect?

For each dtype (default bf16 and fp32) the target and the DFlash drafter are
loaded in that dtype (attention ``sdpa`` for everything) and every selected
gsm8k prompt is decoded greedily with
  * AR       = ``dflash_generate(block_size=1)``,
  * DFlash   = ``dflash_generate(block_size=<drafter block size>)``,
  * BASTION  = ``bastion_generate`` adaptive, with the arguments the harness
               uses (cached calibration + warm profile constants, read only).
Each DFlash/BASTION output is compared with the AR output of the same dtype.
At every first divergence the AR target is run once, teacher-forced, on
prompt + common prefix (same dtype) and the top-1 / top-2 logit margin at that
position is recorded. Small margins mean the two decoders disagree on a near
tie, i.e. a numerical effect rather than a method bug. AR outputs of different
dtypes are also compared with each other (margin from the later dtype's model).

Nothing is written under upstream/cache (the gsm8k cache file must exist).
Example (GPU):
  .venv/bin/python analysis/precision_check.py --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 --out-dir results/precision_check_qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import repro_lib as rl  # noqa: E402  (puts ./upstream on sys.path)
from check_lossless import first_divergence  # noqa: E402

METHODS = ("ar", "dflash", "bastion")
DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def prepare_prompts(args) -> tuple[list[dict], int]:
    """First ``n`` prompts of the upstream gsm8k selection (cap ``selection_cap``, as in E1)."""
    from transformers import AutoTokenizer

    from bastion import benchmark as ub

    jsonl = ub.CACHE_DIR / "gsm8k.jsonl"
    if not jsonl.exists():  # upstream's loader would download and write into upstream/cache
        raise FileNotFoundError(f"{jsonl} missing; this script only reads upstream/cache")
    rows = ub.load_and_process_dataset("gsm8k")
    idx = rl.upstream_selection_indices(len(rows), args.selection_cap)[: args.num_samples]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for i in idx:
        text = ub._apply_chat_template(tok, [{"role": "user", "content": rows[i]["turns"][0]}], False)
        prompts.append({"orig_idx": i, "input_ids": tok.encode(text)})
    return prompts, tok.eos_token_id


def load_models(args, dtype_name: str, device):
    import torch
    from transformers import AutoModelForCausalLM

    from dflash.model import DFlashDraftModel

    dtype = getattr(torch, DTYPES[dtype_name])
    target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=dtype)
    draft = DFlashDraftModel.from_pretrained(args.draft_model, attn_implementation="sdpa", dtype=dtype)
    return target.to(device).eval(), draft.to(device).eval()


def bastion_settings(args, device) -> dict:
    """Cost-model keys and profile constants exactly as run_benchmark.py passes them (files read only)."""
    from bastion import benchmark as ub

    rl.register_local_gpu_specs()
    model_key = args.model_key or ub.infer_cost_model_name_from_target(args.model)
    gpu_key = args.gpu_key or (ub.infer_cost_gpu_type_from_device(device) if device.type == "cuda" else "b4500")
    calib = Path(args.calibration_json or ub.CACHE_DIR / f"calibration_{gpu_key}_{model_key}.json")
    prof_path = Path(args.profile_json or ub.CACHE_DIR / f"calibration_{gpu_key}_{model_key}.profile.warm.json")
    if not calib.is_file() or not prof_path.is_file():
        raise FileNotFoundError(f"need existing calibration {calib} and warm profile {prof_path}")
    prof = json.loads(prof_path.read_text(encoding="utf-8"))
    os.environ["CALIBRATION_JSON_PATH"] = str(calib)
    return {"cost_model_name": model_key, "cost_gpu_type": gpu_key,
            "draft_latency_s": float(prof["draft_latency_s"]),
            "non_verify_latency_s": float(prof["non_verify_latency_s"]),
            "calibration_json": str(calib), "profile_json": str(prof_path)}


# ---------------------------------------------------------------------------
# generation and margins
# ---------------------------------------------------------------------------

def generate(method: str, target, draft, input_ids, *, max_new_tokens, eos, block_size, bastion_kw) -> list[int]:
    from bastion.tree_draft import bastion_generate
    from dflash.model import dflash_generate

    common = dict(target=target, input_ids=input_ids, max_new_tokens=max_new_tokens,
                  stop_token_ids=None if eos is None else [eos], temperature=0.0, return_stats=False)
    if method == "bastion":
        kw = {k: bastion_kw[k] for k in ("cost_model_name", "cost_gpu_type", "draft_latency_s", "non_verify_latency_s")}
        out = bastion_generate(draft, block_size=block_size, **common, **kw)
    else:
        out = dflash_generate(draft, block_size=1 if method == "ar" else block_size, **common)
    return out[0, input_ids.shape[1]:].tolist()


def margin_at(target, prompt_ids: list[int], prefix: list[int], ar_token, other_token) -> dict:
    """Teacher-forced logits of the next token after prompt + prefix (one forward, model's dtype)."""
    import torch

    ids = torch.tensor([prompt_ids + prefix], device=target.device)
    with torch.inference_mode():
        logits = target(ids, logits_to_keep=1).logits[0, -1].float()
    top = torch.topk(logits, 2)
    out = {"top1_id": int(top.indices[0]), "top2_id": int(top.indices[1]),
           "top1_top2_margin": float(top.values[0] - top.values[1])}
    out["teacher_forced_argmax_equals_ar"] = ar_token is not None and out["top1_id"] == ar_token
    if ar_token is not None and other_token is not None:
        out["ar_minus_other_logit"] = float(logits[ar_token] - logits[other_token])
    return out


def divergence_row(target, prompt: dict, ref: list[int], other: list[int]) -> dict:
    d = first_divergence(ref, other)
    row = {"orig_idx": prompt["orig_idx"], "len_ref": len(ref), "len_other": len(other), "first_divergence": d}
    if d is not None:
        ref_tok = ref[d] if d < len(ref) else None
        other_tok = other[d] if d < len(other) else None
        row.update(ref_token=ref_tok, other_token=other_tok,
                   **margin_at(target, prompt["input_ids"], ref[:d], ref_tok, other_tok))
    return row


def run_dtype(target, draft, prompts, *, eos, max_new_tokens, bastion_kw, methods=METHODS, log=print) -> dict:
    import torch

    block_size = int(draft.block_size)
    outputs: dict[str, list[list[int]]] = {m: [] for m in methods}
    for p in prompts:
        ids = torch.tensor([p["input_ids"]], device=target.device)
        for m in methods:
            outputs[m].append(generate(m, target, draft, ids, max_new_tokens=max_new_tokens, eos=eos,
                                       block_size=block_size, bastion_kw=bastion_kw))
        log(f"  #{p['orig_idx']}: " + " ".join(f"{m}={len(outputs[m][-1])}" for m in methods))
    comparisons = {m: [divergence_row(target, p, outputs["ar"][i], outputs[m][i]) for i, p in enumerate(prompts)]
                   for m in methods if m != "ar"}
    return {"block_size": block_size, "outputs": outputs, "comparisons": comparisons}


def summarize(rows: list[dict]) -> dict:
    div = [r for r in rows if r["first_divergence"] is not None]
    margins = [r["top1_top2_margin"] for r in div]
    return {
        "n": len(rows), "exact_match": len(rows) - len(div),
        "exact_match_rate": (len(rows) - len(div)) / len(rows) if rows else None,
        "first_divergences": [r["first_divergence"] for r in div],
        "median_first_divergence": float(np.median([r["first_divergence"] for r in div])) if div else None,
        "margins": margins,
        "median_margin": float(np.median(margins)) if margins else None,
        "max_margin": float(np.max(margins)) if margins else None,
        "teacher_forced_agrees_with_ref": sum(bool(r["teacher_forced_argmax_equals_ar"]) for r in div),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_markdown(path: Path, result: dict) -> None:
    a = result["args"]
    lines = [f"# Precision check: {a['model']} + {a['draft_model']}", "",
             f"{len(result['prompts'])} gsm8k prompts (orig_idx {[p['orig_idx'] for p in result['prompts']]}), "
             f"max_new_tokens={a['max_new_tokens']}, T=0, attention sdpa, device {a['device']}.",
             "In `X vs Y`, Y is the reference. Margin = top-1 minus top-2 logit of Y's target model, teacher-forced "
             "on prompt + common prefix, at the first divergence (same dtype as Y). TF argmax = ref: the "
             "teacher-forced argmax equals Y's token there.", "",
             "| comparison | exact match | first divergences | median margin | max margin | TF argmax = ref |",
             "|---|---|---|---|---|---|"]
    for name, s in result["summary"].items():
        fmt = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
        lines.append(f"| {name} | {s['exact_match']}/{s['n']} ({s['exact_match_rate']:.0%}) | "
                     f"{s['first_divergences'] or '-'} | {fmt(s['median_margin'])} | {fmt(s['max_margin'])} | "
                     f"{s['teacher_forced_agrees_with_ref']}/{s['n'] - s['exact_match']} |")
    lines += ["", "Per-divergence margins (logits):", ""]
    for name, rows in result["comparisons"].items():
        for r in rows:
            if r["first_divergence"] is not None:
                lines.append(f"- {name} #{r['orig_idx']}: pos {r['first_divergence']}, ref tok {r['ref_token']} vs "
                             f"{r['other_token']}, top1-top2 {r['top1_top2_margin']:.4g}, "
                             f"ref-other {r.get('ar_minus_other_logit', float('nan')):.4g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> dict:
    import torch

    args = build_parser().parse_args(argv)
    rl.check_method_env(False)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 must really be fp32 on GPU
    torch.backends.cudnn.allow_tf32 = False
    out_dir = Path(args.out_dir)
    if (ROOT / "upstream").resolve() in [out_dir.resolve(), *out_dir.resolve().parents]:
        raise ValueError("--out-dir must not be inside upstream/")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, eos = prepare_prompts(args)
    bastion_kw = bastion_settings(args, device)
    result = {"args": vars(args), "bastion": bastion_kw, "prompts": prompts, "eos": eos,
              "torch": torch.__version__, "comparisons": {}, "summary": {}, "outputs": {},
              "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for dt in args.dtypes:
        print(f"=== {dt}", flush=True)
        target, draft = load_models(args, dt, device)
        res = run_dtype(target, draft, prompts, eos=eos, max_new_tokens=args.max_new_tokens, bastion_kw=bastion_kw)
        result["outputs"][dt] = res["outputs"]
        for m, rows in res["comparisons"].items():
            result["comparisons"][f"{m}-{dt} vs ar-{dt}"] = rows
        for prev in result["outputs"]:  # AR across dtypes; margin from the current (later) dtype
            if prev != dt:
                result["comparisons"][f"ar-{prev} vs ar-{dt}"] = [
                    divergence_row(target, p, result["outputs"][dt]["ar"][i], result["outputs"][prev]["ar"][i])
                    for i, p in enumerate(prompts)]
        del target, draft
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["summary"] = {k: summarize(v) for k, v in result["comparisons"].items()}
    (out_dir / "precision_check.json").write_text(json.dumps(rl.to_jsonable(result), indent=1), encoding="utf-8")
    write_markdown(out_dir / "precision_check.md", result)
    print((out_dir / "precision_check.md").read_text(encoding="utf-8"))
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--dtypes", nargs="+", choices=list(DTYPES), default=["bf16", "fp32"])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--selection-cap", type=int, default=64, help="gsm8k cap of the E1 selection (first N are used)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-key", default=None, help="cost-model key (default: inferred from --model)")
    p.add_argument("--gpu-key", default=None, help="cost-model GPU key (default: inferred on cuda, b4500 on cpu)")
    p.add_argument("--calibration-json", default=None, help="default upstream/cache/calibration_<gpu>_<model>.json")
    p.add_argument("--profile-json", default=None,
                   help="default upstream/cache/calibration_<gpu>_<model>.profile.warm.json")
    return p


if __name__ == "__main__":
    main()
