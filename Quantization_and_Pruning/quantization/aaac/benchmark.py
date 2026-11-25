from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from Backbones.Generative_Pretrained.qwen2_5 import MODEL_ID, MODEL_REVISION, load_qwen25_05b
from Dataloaders.calibration.activation_energy import eligible_linear_modules
from Dataloaders.calibration.layer_targets import iter_quantization_targets
from Dataloaders.evaluation.wikitext2 import EvalWindows, load_wikitext2_eval_windows
from Quantization_and_Pruning.quantization.aaac.config import AAACConfig
from Quantization_and_Pruning.quantization.aaac.fit import fit_aaac_layer
from Quantization_and_Pruning.quantization.aaac.linear import (
    measure_model_storage_bytes,
    replace_linear_layers_with_aaac,
)
from Quantization_and_Pruning.quantization.aaac.rtn import quantize_model_rtn


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _default_eval_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def _read_peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        # Current MPS driver allocation + process peak RSS (macOS ru_maxrss is in bytes)
        mps_mem = int(torch.mps.driver_allocated_memory())
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return max(mps_mem, rss)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if platform.system() == "Darwin" else rss * 1024)


def _hardware_metadata(device: torch.device, fit_device: torch.device) -> dict[str, Any]:
    return {
        "eval_device": str(device),
        "fit_device": str(fit_device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else ("Apple Silicon MPS" if device.type == "mps" else None)
        ),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
    }


@torch.no_grad()
def evaluate_wikitext2_perplexity(
    model: nn.Module,
    eval_windows: EvalWindows,
    device: torch.device,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Compute aggregate next-token perplexity over non-overlapping 2,048-token windows."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    num_windows = eval_windows.input_ids.shape[0]

    for start in range(0, num_windows, batch_size):
        batch = eval_windows.input_ids[start : start + batch_size].to(device)
        outputs = model(input_ids=batch, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += float(loss_sum.item())
        total_tokens += int(shift_labels.numel())

    mean_nll = total_nll / total_tokens
    return {
        "perplexity": float(math.exp(mean_nll)),
        "mean_nll": mean_nll,
        "evaluated_tokens": total_tokens,
        "evaluated_windows": num_windows,
        "total_available_windows": eval_windows.total_available_windows,
        "run_status": "partial" if eval_windows.is_partial else "full",
    }


@torch.no_grad()
def benchmark_forward_latency(
    model: nn.Module,
    eval_windows: EvalWindows,
    device: torch.device,
    batch_size: int = 1,
    warmup_steps: int = 2,
    repetitions: int = 10,
) -> dict[str, float]:
    """Measure median/p95 forward latency and tokens/s on identical evaluation tokens."""
    model.eval()
    sample = eval_windows.input_ids[:batch_size].to(device)
    tokens_per_step = int(sample.numel())

    for _ in range(warmup_steps):
        _ = model(input_ids=sample, use_cache=False)
    _sync_device(device)

    latencies_s: list[float] = []
    for _ in range(repetitions):
        _sync_device(device)
        t0 = time.perf_counter()
        _ = model(input_ids=sample, use_cache=False)
        _sync_device(device)
        latencies_s.append(time.perf_counter() - t0)

    arr_ms = np.array(latencies_s, dtype=np.float64) * 1000.0
    median_s = float(np.median(latencies_s))
    return {
        "median_latency_ms": float(np.median(arr_ms)),
        "p95_latency_ms": float(np.percentile(arr_ms, 95)),
        "tokens_per_sec": float(tokens_per_step / median_s),
    }


def run_variant(
    variant_name: str,
    cfg: dict[str, Any],
    eval_windows: EvalWindows,
    device: torch.device,
    fit_device: torch.device,
    model_dtype: torch.dtype,
    activation_energies: Optional[dict[str, torch.Tensor]] = None,
    selection_size: Optional[int] = None,
) -> dict[str, Any]:
    _reset_peak_memory(device)
    model, _ = load_qwen25_05b(device=str(device), dtype=model_dtype)
    target_layers = list(eligible_linear_modules(model).keys())
    scale_dtype = _resolve_dtype(cfg["scale_dtype"])
    codebook_dtype = _resolve_dtype(cfg["codebook_dtype"])

    quant_time_s = 0.0
    if variant_name == "rtn":
        t0 = time.perf_counter()
        quantize_model_rtn(
            model=model,
            target_layer_names=target_layers,
            group_size=cfg["group_size"],
            scale_dtype=scale_dtype,
        )
        _sync_device(device)
        quant_time_s = time.perf_counter() - t0
    elif variant_name.startswith("aaac"):
        assert activation_energies is not None and selection_size is not None
        aaac_cfg = AAACConfig(
            group_size=cfg["group_size"],
            selection_size=selection_size,
            outer_iters=cfg["outer_iters"],
            inner_iters=cfg["inner_iters"],
            scale_dtype=scale_dtype,
            codebook_dtype=codebook_dtype,
        )
        t0 = time.perf_counter()
        fits = {}
        for target in iter_quantization_targets(model, activation_energies, device=fit_device):
            fits[target.name] = fit_aaac_layer(target.weight, target.importance, config=aaac_cfg).to(
                device
            )
        replace_linear_layers_with_aaac(model, fits)
        _sync_device(device)
        quant_time_s = time.perf_counter() - t0

    storage = measure_model_storage_bytes(model)

    _reset_peak_memory(device)
    ppl_metrics = evaluate_wikitext2_perplexity(
        model=model,
        eval_windows=eval_windows,
        device=device,
        batch_size=cfg["evaluation"]["batch_size"],
    )
    timing_metrics = benchmark_forward_latency(
        model=model,
        eval_windows=eval_windows,
        device=device,
        batch_size=cfg["evaluation"]["batch_size"],
        warmup_steps=cfg["timing"]["warmup_steps"],
        repetitions=cfg["timing"]["repetitions"],
    )
    peak_runtime_memory_bytes = _read_peak_memory_bytes(device)

    del model
    _reset_peak_memory(device)

    return {
        "variant": variant_name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(model_dtype),
        "group_size": None if variant_name == "unmodified" else cfg["group_size"],
        "selection_size": selection_size,
        "run_status": ppl_metrics["run_status"],
        "wikitext2_ppl": round(ppl_metrics["perplexity"], 4),
        "evaluated_windows": ppl_metrics["evaluated_windows"],
        "evaluated_tokens": ppl_metrics["evaluated_tokens"],
        "quantization_time_s": round(quant_time_s, 3),
        "median_latency_ms": round(timing_metrics["median_latency_ms"], 3),
        "p95_latency_ms": round(timing_metrics["p95_latency_ms"], 3),
        "tokens_per_sec": round(timing_metrics["tokens_per_sec"], 2),
        "peak_dense_runtime_memory_mb": round(peak_runtime_memory_bytes / (1024**2), 2),
        "packed_quantized_mb": round(storage["total_quantized_bytes"] / (1024**2), 3),
        "total_checkpoint_mb": round(storage["total_checkpoint_bytes"] / (1024**2), 3),
        "storage_breakdown_bytes": storage,
        "energy_joules": "",
        "energy_status": "unmeasured",
        "runtime_note": (
            "Dense dequantization reference runtime; packed bytes are accounted separately "
            "and throughput/memory do not establish native INT4 kernel acceleration."
        ),
    }


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Variant | Status | g | S | WikiText-2 PPL | Quant Time (s) | Median Lat (ms) | "
        "p95 Lat (ms) | Tokens/s | Peak Dense Mem (MB) | Packed Quant (MB) | Total Ckpt (MB) | Energy |"
    )
    sep = "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['run_status']} | {r['group_size'] or '-'} | "
            f"{r['selection_size'] or '-'} | {r['wikitext2_ppl']:.4f} | "
            f"{r['quantization_time_s']:.3f} | {r['median_latency_ms']:.3f} | "
            f"{r['p95_latency_ms']:.3f} | {r['tokens_per_sec']:.2f} | "
            f"{r['peak_dense_runtime_memory_mb']:.2f} | {r['packed_quantized_mb']:.3f} | "
            f"{r['total_checkpoint_mb']:.3f} | {r['energy_status']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="DA4022 AAAC vs RTN vs Unmodified Benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Quantization_and_Pruning/quantization/aaac/benchmark_config.json"),
    )
    parser.add_argument("--device", type=str, default=_default_eval_device())
    parser.add_argument(
        "--fit-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for running fit_aaac_layer (CPU or CUDA recommended).",
    )
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-windows", type=int, default=None, help="Override for partial smoke runs")
    parser.add_argument("--variants", nargs="+", default=["unmodified", "rtn", "aaac_s128"])
    parser.add_argument("--output-json", type=Path, default=Path("aaac_benchmark_results.json"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.max_windows is not None:
        cfg["evaluation"]["max_windows"] = args.max_windows

    device = torch.device(args.device)
    fit_device = torch.device(args.fit_device)
    model_dtype = _resolve_dtype(args.dtype) if args.dtype else _default_model_dtype(device)

    _, tokenizer = load_qwen25_05b(device="cpu", dtype=torch.float32)
    eval_windows = load_wikitext2_eval_windows(
        tokenizer=tokenizer,
        seq_len=cfg["evaluation"]["seq_len"],
        max_windows=cfg["evaluation"]["max_windows"],
        dataset_name=cfg["evaluation"]["dataset_name"],
        dataset_config=cfg["evaluation"]["dataset_config"],
        split=cfg["evaluation"]["split"],
    )

    activation_energies = None
    if any(v.startswith("aaac") for v in args.variants):
        energy_path = Path(cfg["calibration"]["activation_energy_path"])
        activation_energies = torch.load(energy_path, map_location="cpu", weights_only=False)

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        s_val = 128 if variant == "aaac_s128" else (16 if variant == "aaac_s16" else None)
        print(f"\n>>> Running variant: {variant} (eval={device}, fit={fit_device}, dtype={model_dtype})")
        rows.append(
            run_variant(
                variant_name=variant,
                cfg=cfg,
                eval_windows=eval_windows,
                device=device,
                fit_device=fit_device,
                model_dtype=model_dtype,
                activation_energies=activation_energies,
                selection_size=s_val,
            )
        )

    payload = {
        "hardware": _hardware_metadata(device, fit_device),
        "config": cfg,
        "results": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    print("\n" + format_markdown_table(rows))
    print(f"\nSaved full telemetry to {args.output_json}")


if __name__ == "__main__":
    main()