from copy import deepcopy

import pytest

import torch

from Mixture_of_Experts.moe_to_dense.calibration import collect_calibration
from Mixture_of_Experts.moe_to_dense.conversion import convert_experts
from Mixture_of_Experts.moe_to_dense.distillation import DistillationConfig, forward_kl, train_distillation
from Mixture_of_Experts.moe_to_dense.synthetic import TinyDenseLM, TinyMoELM, patterned_tokens


def make_models():
    torch.manual_seed(9)
    teacher = TinyMoELM(vocab_size=16, hidden_size=8, intermediate_size=4, num_experts=4, top_k=2)
    data = patterned_tokens(8, 6, 16, 2)
    stats = collect_calibration(teacher, [data])
    converted = convert_experts([e.export_weights() for e in teacher.ffn.experts], stats, scoring="sf", selected_count=2, group_count=2)
    return teacher, TinyDenseLM(teacher, converted.dense_ffn), data


def test_forward_kl_mask_and_identity():
    logits = torch.randn(2, 3, 5)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert abs(float(forward_kl(logits, logits, mask))) < 1e-6
    assert forward_kl(logits, logits + 0.2 * torch.randn_like(logits), mask) >= -1e-6


def test_distillation_freezes_teacher_updates_student_and_resumes(tmp_path):
    teacher, student, data = make_models()
    teacher_before = deepcopy(teacher.state_dict())
    student_before = deepcopy(student.state_dict())
    checkpoint = tmp_path / "checkpoint.pt"
    first_losses = train_distillation(teacher, student, [data], DistillationConfig(steps=2, checkpoint_every=1), checkpoint_path=checkpoint)
    assert len(first_losses) == 2
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[name]) for name, value in teacher_before.items())
    assert any(not torch.equal(value, student.state_dict()[name]) for name, value in student_before.items())
    resumed = train_distillation(teacher, student, [data], DistillationConfig(steps=4), checkpoint_path=checkpoint, resume=True)
    assert len(resumed) == 4
    assert all(torch.isfinite(torch.tensor(resumed)))


def test_student_created_from_frozen_teacher_is_trainable():
    teacher, student, data = make_models()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    stats = collect_calibration(teacher, [data])
    converted = convert_experts([e.export_weights() for e in teacher.ffn.experts], stats, scoring="sf", selected_count=2, group_count=2)
    second_student = TinyDenseLM(teacher, converted.dense_ffn)
    assert all(parameter.requires_grad for parameter in second_student.parameters())


def test_resume_across_accumulation_boundary_matches_full_run(tmp_path):
    teacher, student, data = make_models()
    full = deepcopy(student)
    train_distillation(teacher, full, [data], DistillationConfig(steps=6, gradient_accumulation_steps=2))
    path = tmp_path / 'resume.pt'
    train_distillation(teacher, student, [data], DistillationConfig(steps=4, gradient_accumulation_steps=2, checkpoint_every=3), checkpoint_path=path)
    saved = torch.load(path, weights_only=True)
    assert saved['step'] == 4
    train_distillation(teacher, student, [data], DistillationConfig(steps=6, gradient_accumulation_steps=2), checkpoint_path=path, resume=True)
    for name, parameter in full.state_dict().items():
        torch.testing.assert_close(parameter, student.state_dict()[name])


def test_resume_rejects_partial_final_window(tmp_path):
    teacher, student, data = make_models()
    path = tmp_path / 'partial.pt'
    train_distillation(teacher, student, [data], DistillationConfig(steps=3, gradient_accumulation_steps=2), checkpoint_path=path)
    with pytest.raises(ValueError, match='partial accumulation'):
        train_distillation(teacher, student, [data], DistillationConfig(steps=4, gradient_accumulation_steps=2), checkpoint_path=path, resume=True)
