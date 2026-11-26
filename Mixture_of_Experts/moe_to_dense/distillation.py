"""Forward-KL distillation with masking, accumulation, and resumable checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DistillationConfig:
    learning_rate: float = 1e-3
    steps: int = 25
    gradient_accumulation_steps: int = 1
    temperature: float = 1.0
    checkpoint_every: int = 0


def forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_log_prob = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    token_loss = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1) * temperature**2
    if mask is None:
        return token_loss.mean()
    if mask.shape != token_loss.shape:
        raise ValueError("mask must match logits without vocabulary dimension")
    mask = mask.to(token_loss)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


def save_checkpoint(path: str | Path, student: nn.Module, optimizer: torch.optim.Optimizer, step: int, losses: list[float], config: DistillationConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "losses": losses, "config": asdict(config)}, path)


def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    batches: Iterable[torch.Tensor],
    config: DistillationConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> list[float]:
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    start_step, losses = 0, []
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, losses = int(state["step"]), list(state["losses"])
        if state["config"]["gradient_accumulation_steps"] != config.gradient_accumulation_steps:
            raise ValueError("cannot resume with a different gradient accumulation interval")
        if start_step < config.steps and start_step % config.gradient_accumulation_steps:
            raise ValueError("cannot extend a run that ended with a partial accumulation window")

    materialized = list(batches)
    if not materialized:
        raise ValueError("distillation received no batches")
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint_step = start_step
    for step in range(start_step, config.steps):
        batch = materialized[step % len(materialized)]
        with torch.no_grad():
            teacher_logits = teacher(batch).logits
        student_logits = student(batch).logits
        loss = forward_kl(teacher_logits, student_logits, temperature=config.temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite distillation loss")
        (loss / config.gradient_accumulation_steps).backward()
        update_boundary = (step + 1) % config.gradient_accumulation_steps == 0 or step + 1 == config.steps
        if update_boundary:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        # Save only after an optimizer update, never with pending gradients.
        if checkpoint_path and config.checkpoint_every and update_boundary and step + 1 - last_checkpoint_step >= config.checkpoint_every:
            save_checkpoint(checkpoint_path, student, optimizer, step + 1, losses, config)
            last_checkpoint_step = step + 1
    if checkpoint_path:
        save_checkpoint(checkpoint_path, student, optimizer, config.steps, losses, config)
    return losses
