"""Run the complete hardware-feasible synthetic reproduction."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from copy import deepcopy
from pathlib import Path

import torch

from ..calibration import collect_calibration
from ..configuration import validate_config
from ..conversion import convert_experts
from ..distillation import DistillationConfig, forward_kl, train_distillation
from ..evaluation import benchmark_inference, parameter_bytes, parameter_count, perplexity
from ..synthetic import TinyDenseLM, TinyMoELM, patterned_tokens


def batches(tensor: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [tensor[index : index + batch_size] for index in range(0, len(tensor), batch_size)]


def write_metrics_csv(results: dict, path: Path) -> None:
    """Export the same measured values as results.json on every run."""
    fieldnames = ['experiment_id', 'paper_id', 'method', 'backbone_id', 'dataset_id', 'split', 'seed', 'device', 'framework_version', 'dtype', 'batch_size', 'input_length', 'quality_metric', 'quality_value', 'latency_median_ms', 'throughput_tokens_per_second', 'peak_memory_mib', 'energy_j', 'status', 'notes']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in [('routed_teacher', results['teacher']), *results['students'].items()]:
            teacher = method == 'routed_teacher'
            writer.writerow(dict(experiment_id='synthetic_teacher' if teacher else f'synthetic_{method}_post_kd', paper_id='arxiv_2605_28207', method=method, backbone_id='tiny_moe_lm' if teacher else 'tiny_dense_lm', dataset_id='synthetic_pattern', split='evaluation', seed=results['seed'], device=results['hardware']['device'], framework_version=results['software']['torch'], dtype='float32', batch_size=results['data']['batch_size'], input_length=results['data']['sequence_length'], quality_metric='perplexity', quality_value=metrics['perplexity'] if teacher else metrics['post_distillation_perplexity'], latency_median_ms=metrics['latency_ms_median'], throughput_tokens_per_second=metrics['throughput_tokens_per_second'], peak_memory_mib=metrics['peak_memory_mib'], energy_j='', status='partial', notes=f"{metrics['peak_memory_scope']}; {metrics['parameters']} parameters"))


def train_teacher(model: TinyMoELM, data: list[torch.Tensor], steps: int, learning_rate: float) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for step in range(steps):
        batch = data[step % len(data)]
        loss = model(batch, labels=batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run(config: dict, output_dir: Path) -> dict:
    validate_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    model_cfg = config["model"]
    teacher = TinyMoELM(**model_cfg)
    train = patterned_tokens(config["data"]["train_samples"], config["data"]["sequence_length"], model_cfg["vocab_size"], seed)
    calibration = patterned_tokens(config["data"]["calibration_samples"], config["data"]["sequence_length"], model_cfg["vocab_size"], seed + 1)
    evaluation = patterned_tokens(config["data"]["evaluation_samples"], config["data"]["sequence_length"], model_cfg["vocab_size"], seed + 2)
    batch_size = config["data"]["batch_size"]
    train_batches = batches(train, batch_size)
    calibration_batches = batches(calibration, batch_size)
    evaluation_batches = batches(evaluation, batch_size)

    teacher_losses = train_teacher(teacher, train_batches, config["teacher_training"]["steps"], config["teacher_training"]["learning_rate"])
    stats = collect_calibration(teacher, calibration_batches)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.save(output_dir / "calibration_stats.pt")
    torch.save(teacher.state_dict(), output_dir / "teacher_state.pt")
    (output_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    teacher_ppl = perplexity(teacher, evaluation_batches)
    teacher_result = {
        "perplexity": teacher_ppl,
        "parameters": parameter_count(teacher),
        "parameter_bytes": parameter_bytes(teacher),
        **benchmark_inference(teacher, evaluation_batches[0], repeats=config["benchmark_repeats"]),
    }
    methods = {}
    experts = [expert.export_weights() for expert in teacher.ffn.experts]
    for method in config["methods"]:
        # Recreate the same non-MoE weights for every controlled comparison.
        converted = convert_experts(
            experts,
            stats,
            scoring=method,
            selected_count=config["conversion"]["selected_count"],
            group_count=config["conversion"]["group_count"],
            scaling=config["conversion"]["scaling"],
            lambda_reg=config["conversion"]["lambda_reg"],
        )
        student = TinyDenseLM(teacher, converted.dense_ffn)
        pre_ppl = perplexity(student, evaluation_batches)
        with torch.no_grad():
            pre_kl = float(forward_kl(teacher(evaluation_batches[0]).logits, student(evaluation_batches[0]).logits))
        checkpoint = output_dir / f"{method}_checkpoint.pt"
        kd_cfg = DistillationConfig(**config["distillation"])
        kd_losses = train_distillation(teacher, student, train_batches, kd_cfg, checkpoint_path=checkpoint)
        post_ppl = perplexity(student, evaluation_batches)
        with torch.no_grad():
            post_kl = float(forward_kl(teacher(evaluation_batches[0]).logits, student(evaluation_batches[0]).logits))
        torch.save(student.state_dict(), output_dir / f"{method}_student.pt")
        methods[method] = {
            "selected_experts": converted.selected_experts,
            "groups": converted.groups,
            "pre_distillation_perplexity": pre_ppl,
            "post_distillation_perplexity": post_ppl,
            "pre_distillation_kl": pre_kl,
            "post_distillation_kl": post_kl,
            "distillation_loss_first": kd_losses[0],
            "distillation_loss_last": kd_losses[-1],
            "parameters": parameter_count(student),
            "parameter_bytes": parameter_bytes(student),
            "compression_ratio": parameter_count(student) / parameter_count(teacher),
            **benchmark_inference(student, evaluation_batches[0], repeats=config["benchmark_repeats"]),
        }

    results = {
        "status": "small-scale methodological reproduction; not a paper-result reproduction",
        "seed": seed,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "hardware": {"device": "CPU", "torch_threads": torch.get_num_threads(), "gpu_used": False},
        "data": config["data"],
        "teacher_training": {
            **config["teacher_training"],
            "loss_first": teacher_losses[0],
            "loss_last": teacher_losses[-1],
        },
        "teacher": teacher_result,
        "students": methods,
        "limitations": [
            "The teacher and dataset are synthetic; results do not measure natural-language ability.",
            "No pretrained multi-billion-parameter checkpoint was downloaded or evaluated.",
            "Peak CPU working set includes Python/PyTorch startup and model loading; small model differences may be obscured by runtime overhead.",
            "Latency and throughput are local CPU medians and are not comparable to the paper's GPU results.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_metrics_csv(results, output_dir / 'metrics.csv')
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.config.parent / "results" / "synthetic"
    results = run(config, output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
