"""Quantize a (fine-tuned) checkpoint at several bit-widths and evaluate it.

    python Quantization_and_Pruning/experiments/jacobian_noise/eval_quant.py \
        --config Quantization_and_Pruning/configs/jacobian_noise/eval_qwen2_5_0_5b.yaml \
        --set model.path=outputs/jn/qwen05__jacobian_a0.5__s0/checkpoint

For each tag in ``quant.tags`` (e.g. FP16, W8A8, W6A6, W4A8, W4A4, W4A4QK4) the
model is re-loaded, quantized (RTN or GPTQ weights + per-token activations,
simulated), evaluated, and timed.  Results are appended to
``<output_dir>/results.csv`` in the schema of Performance_Metrics/README_metrics.md
and, with more detail, to ``results_detail.jsonl``.

Noise is never active here: models are in ``eval()`` mode.  The
``jacobian_noise`` attention implementation is used for every tag (it equals
eager attention when noise is off) so that all rows share one attention code path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    PAPER_ID, append_jsonl, append_result_rows, env_metadata, load_config, measure_latency, peak_memory_mb,
    pick_device, reset_peak_memory, set_seed,
)

import torch  # noqa: E402

from Backbones.registry import get_model_metadata, load_backbone  # noqa: E402
from Quantization_and_Pruning.quantization.activation.fake_quant import (  # noqa: E402
    FakeQuantConfig, apply_fake_quant, model_size_bytes,
)
from Quantization_and_Pruning.quantization.qat.jacobian_noise import (  # noqa: E402
    JacobianNoiseConfig, attach_jacobian_noise, detach_jacobian_noise, use_jacobian_noise_attention,
)
from Quantization_and_Pruning.quantization.weight_only.gptq import gptq_quantize_model  # noqa: E402


# ------------------------------------------------------------------ loading
def read_run_info(path: str) -> dict:
    """run.json written by train_*.py next to the checkpoint (if any)."""
    if not path:
        return {}
    cand = os.path.join(os.path.dirname(os.path.abspath(path.rstrip("/"))), "run.json")
    if os.path.exists(cand):
        with open(cand) as f:
            return json.load(f)
    return {}


def train_method_label(run: dict) -> str:
    if not run:
        return "pretrained(no-finetune)"
    n = run["config"]["noise"]
    if n["mode"] == "none":
        return "finetune(no-noise)"
    if n["mode"] == "uniform_matched":
        return "finetune+uniform_matched_noise"
    return f"finetune+jacobian_noise(mode={n['mode']},alpha={n.get('alpha', 0.5)})"


def eval_dtype(name: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model(mcfg: dict, task: str, dtype, device):
    path, bid = mcfg.get("path"), mcfg["backbone_id"]
    if path:
        if task == "lm":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager")
            pre = AutoTokenizer.from_pretrained(path) if os.path.exists(os.path.join(path, "tokenizer_config.json")) else None
        else:
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(path, dtype=dtype, attn_implementation="eager")
            pre = AutoProcessor.from_pretrained(path) if os.path.exists(os.path.join(path, "preprocessor_config.json")) else None
    else:
        if bid.startswith("test."):
            torch.manual_seed(int(mcfg.get("init_seed", 0)))  # same random init for every tag
        model, pre, _ = load_backbone(bid, dtype=dtype, attn_implementation="eager")
    return model.to(device).eval(), pre


# ------------------------------------------------------------------- LM task
@torch.no_grad()
def lm_perplexity(model, windows, device) -> float:
    nll, n = 0.0, 0
    for w in windows:
        w = w.to(device)
        logits = model(input_ids=w, use_cache=False).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], w[0, 1:], reduction="sum")
        nll += float(loss)
        n += w.shape[1] - 1
    return math.exp(nll / n)


def lm_data(cfg: dict, tok, vocab_size: int):
    ecfg, qcfg = cfg["eval"]["lm"], cfg["quant"]
    if ecfg["dataset"] == "synthetic.lm":
        g = torch.Generator().manual_seed(0)
        L = int(ecfg.get("seq_len", 32))
        ev = [torch.randint(0, vocab_size, (1, L), generator=g) for _ in range(int(ecfg.get("max_windows") or 4))]
        cal = [torch.randint(0, vocab_size, (1, L), generator=g) for _ in range(int(qcfg.get("calib_samples", 4)))]
        return ev, cal
    from Dataloaders.evaluation.wikitext2 import calibration_windows, eval_windows, load_wikitext2_tokens

    test_tok = load_wikitext2_tokens(tok, "test")
    ev = eval_windows(test_tok, int(ecfg.get("seq_len", 2048)), ecfg.get("max_windows"))
    cal = []
    if qcfg.get("weight_method") == "gptq":
        train_tok = load_wikitext2_tokens(tok, "train")
        cal = calibration_windows(train_tok, int(qcfg.get("calib_samples", 128)), int(qcfg.get("calib_seq_len", 2048)), int(qcfg.get("calib_seed", 0)))
    return ev, cal


def run_lm_eval(model, tok, ecfg: dict) -> dict:
    tasks = ecfg.get("lm_eval_tasks") or []
    if not tasks:
        return {}
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=int(ecfg.get("lm_eval_batch_size", 8)))
    res = simple_evaluate(model=lm, tasks=list(tasks), limit=ecfg.get("lm_eval_limit"), num_fewshot=0)
    out = {}
    for task, metrics in res["results"].items():
        key = "acc_norm,none" if task in ("hellaswag", "piqa", "arc_challenge") else "acc,none"
        if key not in metrics:
            key = next((k for k in metrics if k.startswith("acc")), None)
        if key:
            out[f"{task}/{key.split(',')[0]}"] = metrics[key]
    return out


# --------------------------------------------------------------- SigLIP task
def siglip_eval(model, processor, class_emb, cfg, device, dtype):
    from siglip_utils import image_batches, zero_shot_logits

    ecfg, dcfg = cfg["eval"]["siglip"], cfg["data"]
    scale = model.logit_scale.exp().detach()
    correct = total = 0
    with torch.no_grad():
        for x, y in image_batches(processor, dcfg, "validation", 0, int(ecfg.get("num_eval", 10000)), int(ecfg.get("batch_size", 64)), seed=dcfg.get("seed", 0)):
            pred = zero_shot_logits(model, x.to(device, dtype), class_emb, scale).argmax(-1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(1, total), total


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    ecfg, mcfg, qcfg = cfg["experiment"], cfg["model"], cfg["quant"]
    task = cfg["task"]
    set_seed(int(ecfg.get("seed", 0)))
    device = pick_device(ecfg.get("device", "auto"))
    dtype = eval_dtype(cfg["eval"].get("dtype", "auto"), device)
    out_dir = ecfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    run = read_run_info(mcfg.get("path"))
    bmeta = get_model_metadata(mcfg["backbone_id"])
    train_label = mcfg.get("label") or train_method_label(run)
    env = env_metadata(device)
    seed_label = run.get("config", {}).get("experiment", {}).get("seed", ecfg.get("seed", 0))

    base, pre = load_model(mcfg, task, dtype, device)
    class_emb = None
    if task == "lm":
        eval_windows_, calib = lm_data(cfg, pre, base.config.vocab_size)
    else:
        from siglip_utils import build_class_embeddings, image_batches

        class_emb = build_class_embeddings(base, pre, cfg["data"], device)
        if qcfg.get("weight_method") == "gptq":
            n_cal = int(qcfg.get("calib_samples", 256))
            calib = [x for x, _ in image_batches(pre, cfg["data"], "train", 0, n_cal, 32, seed=cfg["data"].get("seed", 0))]
        else:
            calib = []

    diag = None
    if cfg["eval"].get("jacobian_diagnostics"):
        diag = jacobian_diagnostics(base, task, eval_windows_ if task == "lm" else None, pre, cfg, device, dtype)
    del base

    for tag in qcfg["tags"]:
        model, _ = load_model(mcfg, task, dtype, device)
        use_jacobian_noise_attention(model)
        fq = FakeQuantConfig.from_tag(tag, **{k: v for k, v in qcfg.get("fake_quant", {}).items()})
        info = {"quant_tag": fq.tag, "weight_method": qcfg.get("weight_method", "rtn") if fq.w_bits else "none"}
        if fq.w_bits and qcfg.get("weight_method") == "gptq":
            fwd = (lambda m, b: m(input_ids=b.to(device), use_cache=False)) if task == "lm" else (
                lambda m, b: m.get_image_features(pixel_values=b.to(device, dtype)))
            info.update(gptq_quantize_model(model, calib, fwd, fq, percdamp=float(qcfg.get("percdamp", 0.01)), log=lambda s: None))
            info.update(apply_fake_quant(model, fq, weights_already_quantized=True))
        elif fq.tag != "FP16":
            info.update(apply_fake_quant(model, fq))
        info.update(model_size_bytes(model, cfg=fq))

        reset_peak_memory(device)
        metrics = {}
        if task == "lm":
            metrics["wikitext2_ppl"] = lm_perplexity(model, eval_windows_, device)
            metrics.update(run_lm_eval(model, pre, cfg["eval"]["lm"]))
            n_eval, inp_len = len(eval_windows_), eval_windows_[0].shape[1]
            lat_in = eval_windows_[0][:, : int(cfg.get("latency", {}).get("input_length", inp_len))].to(device)
            lat_fn = lambda: model(input_ids=lat_in, use_cache=False)  # noqa: E731
            dataset_id, split = cfg["eval"]["lm"].get("dataset_id", "text.wikitext2_raw.test"), "test"
        else:
            top1, n_eval = siglip_eval(model, pre, class_emb, cfg, device, dtype)
            metrics["imagenet_zeroshot_top1"] = top1
            inp_len = int(bmeta.get("input_spec", {}).get("image_size", [0])[0] or cfg["data"].get("image_size", 0))
            lb = int(cfg.get("latency", {}).get("batch_size", 32))
            size = cfg["data"].get("image_size") or inp_len
            lat_x = torch.randn(lb, 3, size, size, device=device, dtype=dtype)
            lat_fn = lambda: model.get_image_features(pixel_values=lat_x)  # noqa: E731
            dataset_id, split = "vision.imagenet1k_256.validation", "validation"
        peak = peak_memory_mb(device)
        lat = measure_latency(lat_fn, device, int(cfg.get("latency", {}).get("warmup", 3)), int(cfg.get("latency", {}).get("iters", 10))) if cfg.get("latency", {}).get("enabled", True) else {}

        method = f"{train_label} | {info['weight_method']}-{fq.tag}" if fq.w_bits or fq.a_bits else f"{train_label} | {fq.tag}"
        exp_id = f"{ecfg['name']}|{train_label}|{info['weight_method']}|{fq.tag}|s{seed_label}"
        notes = (
            f"simulated (fake) quantization, no low-bit kernels; latency is FP compute with quant-dequant overhead; "
            f"eval examples={n_eval}; {json.dumps(info)}"
        )
        rows = []
        for name, val in metrics.items():
            rows.append({
                "experiment_id": exp_id, "paper_id": PAPER_ID, "method": method, "commit": env["commit"],
                "backbone_id": mcfg["backbone_id"], "backbone_revision": bmeta.get("revision"),
                "dataset_id": dataset_id if not name.startswith(("arc", "hellaswag", "piqa", "winogrande", "boolq", "truthfulqa")) else f"lm_eval:{name.split('/')[0]}",
                "split": split, "seed": seed_label, "device": env["device"],
                "framework_version": f"torch {env['torch']}; transformers {env['transformers']}",
                "dtype": str(dtype).replace("torch.", ""), "batch_size": 1 if task == "lm" else cfg["eval"]["siglip"].get("batch_size", 64),
                "input_length": inp_len, "output_length": None, "quality_metric": name, "quality_value": round(val, 6),
                "latency_median_ms": round(lat["latency_median_ms"], 3) if lat else None,
                "latency_p95_ms": round(lat["latency_p95_ms"], 3) if lat else None,
                "throughput": None, "peak_memory_mb": round(peak, 1) if peak else None, "energy_j": None,
                "status": ecfg.get("status", "approximate"), "notes": notes,
            })
        append_result_rows(os.path.join(out_dir, "results.csv"), rows)
        append_jsonl(os.path.join(out_dir, "results_detail.jsonl"), {
            "experiment_id": exp_id, "train_method": train_label, "quant": fq.to_dict(), "info": info,
            "metrics": metrics, "latency": lat, "peak_memory_mb": peak, "env": env, "model": mcfg, "seed": seed_label,
        })
        print(f"[eval_quant] {fq.tag:10s} {info['weight_method']:5s} {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if diag is not None:
        append_jsonl(os.path.join(out_dir, "jacobian_diagnostics.jsonl"), {"train_method": train_label, "seed": seed_label, **diag})


@torch.no_grad()
def jacobian_diagnostics(model, task, windows, processor, cfg, device, dtype) -> dict:
    """Mean ||J_softmax||_F^2 per layer on evaluation data (FP model, no noise)."""
    ctrl = attach_jacobian_noise(model, JacobianNoiseConfig(mode="none"), include=None if task == "lm" else "vision_model")
    with ctrl.measure():
        if task == "lm":
            for w in windows[: int(cfg["eval"].get("diagnostic_batches", 8))]:
                model(input_ids=w.to(device), use_cache=False)
        else:
            from siglip_utils import image_batches

            for i, (x, _) in enumerate(image_batches(processor, cfg["data"], "validation", 0, 256, 32, seed=cfg["data"].get("seed", 0))):
                model.get_image_features(pixel_values=x.to(device, dtype))
        stats = ctrl.mean_jacobian_sq()
    detach_jacobian_noise(model, "eager")
    per_layer = {str(i): float(torch.nanmean(v)) for i, v in stats.items()}
    return {"mean_jacobian_fro_sq_per_layer": per_layer, "mean_jacobian_fro_sq": sum(per_layer.values()) / max(1, len(per_layer))}


if __name__ == "__main__":
    main()
