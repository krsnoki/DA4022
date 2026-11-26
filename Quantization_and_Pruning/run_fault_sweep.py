"""Run the reduced floating-point fault injection sweep on CIFAR-10."""

import argparse
import csv
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .bit_flip_injector import BitFlipInjector
from .format_codec import (
    decode_e4m11,
    decode_e5m10,
    decode_e8m7,
    encode_e4m11,
    encode_e5m10,
    encode_e8m7,
)
from .train_cifar10 import build_resnet18


CODECS = {
    "e4m11": (encode_e4m11, decode_e4m11, 4),
    "e5m10": (encode_e5m10, decode_e5m10, 5),
    "e8m7": (encode_e8m7, decode_e8m7, 8),
}

METRIC_FIELDS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id",
    "backbone_revision", "dataset_id", "split", "seed", "device",
    "framework_version", "dtype", "batch_size", "input_length",
    "output_length", "quality_metric", "quality_value",
    "latency_median_ms", "latency_p95_ms", "throughput", "peak_memory_mb",
    "energy_j", "status", "notes",
]


def load_resnet18_checkpoint(checkpoint_path: str) -> nn.Module:
    """Load a training-script checkpoint into the experiment ResNet-18."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must contain a non-empty model state_dict")

    model = build_resnet18()
    model.load_state_dict(checkpoint, strict=True)
    if model.fc.out_features != 10:
        raise ValueError("CIFAR-10 checkpoint must have exactly 10 output classes")
    model.eval()
    return model


def load_resnet18_cifar10(checkpoint_path: str) -> Tuple[nn.Module, Iterable]:
    """Load a trained 10-class ResNet-18 and the CIFAR-10 test loader."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    model = load_resnet18_checkpoint(checkpoint_path)

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=0
    )
    return model, testloader


def _collect_inputs(loader: Iterable, n_images: int, device: torch.device):
    if n_images <= 0:
        raise ValueError("n_images must be greater than zero")

    inputs = []
    targets = []
    collected = 0
    for batch_inputs, batch_targets in loader:
        take = min(n_images - collected, len(batch_inputs))
        inputs.append(batch_inputs[:take])
        targets.append(batch_targets[:take])
        collected += take
        if collected == n_images:
            break
    if collected != n_images:
        raise ValueError(
            f"Requested {n_images} images but the loader provided {collected}"
        )
    return torch.cat(inputs).to(device), torch.cat(targets).to(device)


def _evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> Tuple[float, list]:
    correct = 0
    timings = []
    model.eval()
    with torch.inference_mode():
        warmup_inputs = inputs[:min(batch_size, len(inputs))]
        model(warmup_inputs)
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]
            started = time.perf_counter()
            logits = model(batch_inputs)
            timings.append((time.perf_counter() - started, len(batch_inputs)))
            if logits.ndim != 2 or logits.shape[0] != len(batch_targets):
                raise ValueError("Model output must have shape (batch, classes)")
            correct += (logits.argmax(dim=1) == batch_targets).sum().item()
    return correct / len(inputs), timings


def run_fault_sweep(
    model: nn.Module,
    testloader: Iterable,
    fmt: str = "e4m11",
    k: int = 8,
    bit_group: str = "exponent",
    n_trials: int = 20,
    n_images: int = 2000,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = "cpu",
    clamp_exponent: bool = False,
) -> dict:
    """Measure clean, quantized, and faulted quality for a fixed image subset."""
    if fmt not in CODECS:
        raise ValueError(f"Unsupported floating-point format: {fmt}")
    if k < 0:
        raise ValueError("k must be non-negative")
    if n_trials <= 0:
        raise ValueError("n_trials must be greater than zero")

    if isinstance(device, str):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
    elif device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    inputs, targets = _collect_inputs(testloader, n_images, device)
    batch_size = getattr(testloader, "batch_size", None) or min(256, n_images)

    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    }
    if not parameters:
        raise ValueError("Model has no weight parameters to encode")

    encode, decode_fn, exponent_bits = CODECS[fmt]
    if clamp_exponent and fmt in ("e5m10", "e8m7"):
        decode = lambda packed, scale=1.0: decode_fn(packed, scale, clamp_exponent=True)
    else:
        decode = decode_fn

    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }
    encoded_weights = {}
    quantized_weights = {}
    slices = {}
    packed_parts = []
    packed_offset = 0

    for name, parameter in parameters.items():
        packed, scale = encode(parameter.detach().cpu().numpy())
        decoded = decode(packed, scale)
        encoded_weights[name] = (packed, scale)
        quantized_weights[name] = torch.from_numpy(decoded).to(
            dtype=parameter.dtype, device=device
        )
        flattened = packed.reshape(-1)
        slices[name] = (packed_offset, packed_offset + flattened.size)
        packed_parts.append(flattened)
        packed_offset += flattened.size

    packed_model = np.concatenate(packed_parts)

    def apply_weights(weights: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, values in weights.items():
                parameters[name].copy_(values.to(device))

    injector = BitFlipInjector(seed=seed)
    trial_results = []
    faulted_timings = []
    try:
        clean_accuracy, clean_timings = _evaluate(
            model, inputs, targets, batch_size
        )
        apply_weights(quantized_weights)
        quantized_accuracy, quantized_timings = _evaluate(
            model, inputs, targets, batch_size
        )

        for trial in range(n_trials):
            faulted_packed, metadata = injector.inject(
                packed_model,
                k=k,
                bit_group=bit_group,
                exponent_bits=exponent_bits,
            )
            # Only the flipped weights change, so decode just those and
            # patch them into a copy of the clean quantized weights.
            faulted_weights = {}
            for flat_index in sorted(set(metadata["indices"])):
                name = next(
                    n for n, (start, end) in slices.items()
                    if start <= flat_index < end
                )
                if name not in faulted_weights:
                    faulted_weights[name] = quantized_weights[name].clone()
                start = slices[name][0]
                value = decode(
                    faulted_packed[flat_index:flat_index + 1],
                    encoded_weights[name][1],
                )[0]
                faulted_weights[name].view(-1)[flat_index - start] = float(value)

            apply_weights(faulted_weights)
            faulted_accuracy, timings = _evaluate(
                model, inputs, targets, batch_size
            )
            faulted_timings.extend(timings)
            trial_results.append({
                "trial": trial,
                "k": k,
                "bit_group": bit_group,
                "faulted_accuracy": faulted_accuracy,
                "quality_drop": quantized_accuracy - faulted_accuracy,
                "quality_drop_vs_fp32": clean_accuracy - faulted_accuracy,
                "metadata": metadata,
            })
            apply_weights(quantized_weights)
    finally:
        apply_weights(original_weights)

    faulted_accuracies = [result["faulted_accuracy"] for result in trial_results]
    drops = [result["quality_drop"] for result in trial_results]
    drops_vs_fp32 = [result["quality_drop_vs_fp32"] for result in trial_results]
    processor = platform.processor() or platform.machine()

    def get_commit_hash() -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            return val if val else "unknown"
        except Exception:
            return "unknown"

    commit_hash = get_commit_hash()

    def make_metrics_row(
        method: str,
        accuracy: float,
        timings: list,
        notes: str,
    ) -> dict:
        elapsed = sum(duration for duration, _ in timings)
        latency_ms = [
            duration * 1000 / batch_count
            for duration, batch_count in timings
        ]
        return {
            "experiment_id": "fault_sweep",
            "paper_id": "Haripriya_Tudu_2026_submission_131",
            "method": method,
            "commit": commit_hash,
            "backbone_id": "vision.resnet18.cifar10.finetuned",
            "backbone_revision": "checkpoint",
            "dataset_id": "cifar10",
            "split": "test",
            "seed": seed,
            "device": str(device),
            "framework_version": torch.__version__,
            "dtype": "float32",
            "batch_size": batch_size,
            "input_length": "32x32",
            "output_length": 10,
            "quality_metric": "top1_accuracy",
            "quality_value": float(accuracy),
            "latency_median_ms": float(np.median(latency_ms)),
            "latency_p95_ms": float(np.percentile(latency_ms, 95)),
            "throughput": n_images / elapsed if elapsed > 0 else "",
            "peak_memory_mb": "",
            "energy_j": "",
            "status": "partial-reproduction",
            "notes": (
                f"images={n_images}; timing_scope=model_forward_only; warmup_batches=1; "
                f"peak_memory_and_energy=not_measured; processor={processor}; "
                f"{notes}"
            ),
        }

    metrics_rows = [
        make_metrics_row(
            "FP32_baseline",
            clean_accuracy,
            clean_timings,
            "format=FP32; faults=0",
        ),
        make_metrics_row(
            f"{fmt}_no_faults",
            quantized_accuracy,
            quantized_timings,
            f"format={fmt}; faults=0; "
            f"accuracy_delta_vs_FP32={quantized_accuracy - clean_accuracy:.6f}",
        ),
        make_metrics_row(
            f"{fmt}_fault_injection",
            float(np.mean(faulted_accuracies)),
            faulted_timings,
            f"format={fmt}; k={k}; bit_group={bit_group}; trials={n_trials}; "
            f"mean_accuracy_drop_vs_{fmt}={np.mean(drops):.6f}; "
            f"mean_accuracy_drop_vs_FP32={np.mean(drops_vs_fp32):.6f}; "
            f"clamp_exponent={clamp_exponent}",
        ),
    ]
    return {
        "clean_accuracy": clean_accuracy,
        "quantized_accuracy": quantized_accuracy,
        "format": fmt,
        "k": k,
        "bit_group": bit_group,
        "avg_faulted_accuracy": float(np.mean(faulted_accuracies)),
        "avg_quality_drop": float(np.mean(drops)),
        "avg_quality_drop_vs_fp32": float(np.mean(drops_vs_fp32)),
        "min_quality_drop": float(np.min(drops)),
        "max_quality_drop": float(np.max(drops)),
        "n_trials": n_trials,
        "n_images": n_images,
        "trial_results": trial_results,
        "metrics_row": metrics_rows[-1],
        "metrics_rows": metrics_rows,
        "clamp_exponent": clamp_exponent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 fault robustness sweep")
    parser.add_argument("--checkpoint", required=True, help="10-class ResNet-18 state_dict")
    parser.add_argument("--format", choices=CODECS, default="e4m11")
    parser.add_argument("--k", type=int, default=8, help="Fault attempts per trial")
    parser.add_argument(
        "--bit-group",
        choices=["exponent", "sign", "mantissa_MSB", "mantissa_LSB"],
        default="exponent",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to run inference on (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--clamp-exponent",
        action="store_true",
        help="Decode all-ones exponent as largest finite exponent for e5m10 and e8m7",
    )
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    model, testloader = load_resnet18_cifar10(args.checkpoint)
    results = run_fault_sweep(
        model,
        testloader,
        fmt=args.format,
        k=args.k,
        bit_group=args.bit_group,
        n_trials=args.trials,
        n_images=args.images,
        seed=args.seed,
        device=args.device,
        clamp_exponent=args.clamp_exponent,
    )
    for row in results["metrics_rows"]:
        row["backbone_revision"] = Path(args.checkpoint).name
    print(
        f"clean FP32 accuracy: {results['clean_accuracy']:.4f}\n"
        f"clean {args.format} accuracy: {results['quantized_accuracy']:.4f}\n"
        f"mean faulted accuracy: {results['avg_faulted_accuracy']:.4f}\n"
        f"mean accuracy drop vs {args.format}: {results['avg_quality_drop']:.4f}\n"
        f"mean accuracy drop vs FP32: {results['avg_quality_drop_vs_fp32']:.4f}",
        file=sys.stderr,
    )

    if args.csv_output:
        with args.csv_output.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(results["metrics_rows"])
        print(f"metrics written to {args.csv_output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(results["metrics_rows"])


if __name__ == "__main__":
    main()
