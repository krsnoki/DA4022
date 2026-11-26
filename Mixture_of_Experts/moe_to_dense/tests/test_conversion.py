import torch

from Mixture_of_Experts.moe_to_dense.calibration import collect_calibration
from Mixture_of_Experts.moe_to_dense.conversion import convert_experts
from Mixture_of_Experts.moe_to_dense.synthetic import TinyDenseLM, TinyMoELM, TinyMoELayer, patterned_tokens


def test_end_to_end_conversion_has_no_router_and_reloads(tmp_path):
    torch.manual_seed(7)
    teacher = TinyMoELM(vocab_size=16, hidden_size=8, intermediate_size=4, num_experts=4, top_k=2)
    data = patterned_tokens(8, 6, 16, 3)
    stats = collect_calibration(teacher, [data])
    result = convert_experts(
        [expert.export_weights() for expert in teacher.ffn.experts],
        stats,
        scoring="do_acp",
        selected_count=2,
        group_count=2,
    )
    student = TinyDenseLM(teacher, result.dense_ffn)
    assert not any(isinstance(module, TinyMoELayer) for module in student.modules())
    assert student(data).logits.shape == (8, 6, 16)
    path = tmp_path / "student.pt"
    torch.save(student.state_dict(), path)
    restored = TinyDenseLM(teacher, result.dense_ffn)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.allclose(student(data).logits, restored(data).logits)


def test_calibration_roundtrip(tmp_path):
    teacher = TinyMoELM(vocab_size=16, hidden_size=8, intermediate_size=4, num_experts=4, top_k=2)
    stats = collect_calibration(teacher, [patterned_tokens(3, 5, 16, 1)])
    path = tmp_path / "stats.pt"
    stats.save(path)
    loaded = type(stats).load(path)
    assert loaded.token_count == stats.token_count
    assert torch.equal(loaded.output_gram, stats.output_gram)
