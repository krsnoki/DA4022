import json
from pathlib import Path

import pytest
import torch

from Mixture_of_Experts.moe_to_dense.configuration import validate_config
from Mixture_of_Experts.moe_to_dense.synthetic import TinyMoELM, TinyDenseLM, patterned_tokens
from Mixture_of_Experts.moe_to_dense.calibration import collect_calibration
from Mixture_of_Experts.moe_to_dense.conversion import convert_experts


def config():
    return json.loads((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())


def test_valid_config():
    assert validate_config(config())['seed'] == 4022


@pytest.mark.parametrize('section,key,value', [
    ('model', 'top_k', 99), ('data', 'batch_size', 0),
    ('data', 'sequence_length', 1), ('conversion', 'selected_count', 99),
    ('conversion', 'group_count', 0), ('conversion', 'scaling', 'bad'),
    ('distillation', 'temperature', 0), ('distillation', 'learning_rate', float('nan')),
    ('distillation', 'gradient_accumulation_steps', 0),
])
def test_invalid_config(section, key, value):
    cfg = config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_conversion_shape_dtype_and_deterministic_selection(dtype):
    torch.manual_seed(42)
    teacher = TinyMoELM(vocab_size=16, hidden_size=8, intermediate_size=4, num_experts=4, top_k=2).to(dtype=dtype)
    tokens = patterned_tokens(4, 6, 16, 42)
    assert tokens.dtype == torch.int64
    stats = collect_calibration(teacher, [tokens])
    experts = [e.export_weights() for e in teacher.ffn.experts]
    first = convert_experts(experts, stats, scoring='do_acp', selected_count=2, group_count=2)
    second = convert_experts(experts, stats, scoring='do_acp', selected_count=2, group_count=2)
    assert first.selected_experts == second.selected_experts
    student = TinyDenseLM(teacher, first.dense_ffn)
    logits = student(tokens).logits
    assert logits.shape == (4, 6, 16)
    assert logits.dtype == dtype
    assert all(p.dtype == dtype for p in student.parameters())
