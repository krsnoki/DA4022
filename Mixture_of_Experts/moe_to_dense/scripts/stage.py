"""Run calibration, conversion, distillation, or evaluation as separate stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..calibration import CalibrationStats, collect_calibration
from ..configuration import validate_config
from ..conversion import convert_experts
from ..distillation import DistillationConfig, forward_kl, train_distillation
from ..evaluation import benchmark_inference, parameter_count, perplexity
from ..synthetic import TinyDenseLM, TinyMoELM, patterned_tokens
from .reproduce import batches, train_teacher


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def make_data(config: dict, offset: int) -> list[torch.Tensor]:
    model = config["model"]
    data = config["data"]
    names = {0: "train_samples", 1: "calibration_samples", 2: "evaluation_samples"}
    tensor = patterned_tokens(data[names[offset]], data["sequence_length"], model["vocab_size"], config["seed"] + offset)
    return batches(tensor, data["batch_size"])


def load_teacher(config: dict, output: Path) -> TinyMoELM:
    teacher = TinyMoELM(**config["model"])
    teacher.load_state_dict(torch.load(output / "teacher_state.pt", map_location="cpu", weights_only=True))
    return teacher


def build_student(config: dict, output: Path, method: str) -> tuple[TinyMoELM, TinyDenseLM, object]:
    teacher = load_teacher(config, output)
    stats = CalibrationStats.load(output / "calibration_stats.pt")
    converted = convert_experts(
        [expert.export_weights() for expert in teacher.ffn.experts],
        stats,
        scoring=method,
        selected_count=config["conversion"]["selected_count"],
        group_count=config["conversion"]["group_count"],
        scaling=config["conversion"]["scaling"],
        lambda_reg=config["conversion"]["lambda_reg"],
    )
    return teacher, TinyDenseLM(teacher, converted.dense_ffn), converted


def calibration_stage(config: dict, output: Path) -> dict:
    torch.manual_seed(config["seed"])
    teacher = TinyMoELM(**config["model"])
    losses = train_teacher(teacher, make_data(config, 0), config["teacher_training"]["steps"], config["teacher_training"]["learning_rate"])
    stats = collect_calibration(teacher, make_data(config, 1))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), output / "teacher_state.pt")
    stats.save(output / "calibration_stats.pt")
    return {"tokens": stats.token_count, "experts": stats.num_experts, "teacher_loss_first": losses[0], "teacher_loss_last": losses[-1]}


def conversion_stage(config: dict, output: Path, method: str) -> dict:
    _, student, converted = build_student(config, output, method)
    torch.save(student.state_dict(), output / f"{method}_initial.pt")
    metadata = {"method": method, "selected_experts": converted.selected_experts, "groups": converted.groups, "parameters": parameter_count(student)}
    (output / f"{method}_conversion.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def distillation_stage(config: dict, output: Path, method: str) -> dict:
    teacher, student, _ = build_student(config, output, method)
    initial = output / f"{method}_initial.pt"
    if initial.exists():
        student.load_state_dict(torch.load(initial, map_location="cpu", weights_only=True))
    losses = train_distillation(
        teacher,
        student,
        make_data(config, 0),
        DistillationConfig(**config["distillation"]),
        checkpoint_path=output / f"{method}_checkpoint.pt",
    )
    torch.save(student.state_dict(), output / f"{method}_student.pt")
    return {"steps": len(losses), "loss_first": losses[0], "loss_last": losses[-1]}


def evaluation_stage(config: dict, output: Path, method: str) -> dict:
    teacher, student, _ = build_student(config, output, method)
    student.load_state_dict(torch.load(output / f"{method}_student.pt", map_location="cpu", weights_only=True))
    data = make_data(config, 2)
    with torch.no_grad():
        kl = float(forward_kl(teacher(data[0]).logits, student(data[0]).logits))
    return {
        "teacher_perplexity": perplexity(teacher, data),
        "student_perplexity": perplexity(student, data),
        "student_kl": kl,
        **benchmark_inference(student, data[0], repeats=config["benchmark_repeats"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration", "conversion", "distillation", "evaluation"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("sf", "cp", "acp", "do_cp", "do_acp"), default="do_acp")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config.get("torch_threads", 1))
    functions = {
        "calibration": lambda: calibration_stage(config, args.output_dir),
        "conversion": lambda: conversion_stage(config, args.output_dir, args.method),
        "distillation": lambda: distillation_stage(config, args.output_dir, args.method),
        "evaluation": lambda: evaluation_stage(config, args.output_dir, args.method),
    }
    print(json.dumps(functions[args.stage](), indent=2))


if __name__ == "__main__":
    main()
