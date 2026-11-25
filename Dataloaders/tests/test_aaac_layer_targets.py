import pytest

torch = pytest.importorskip("torch")
from torch import nn

from Dataloaders.calibration import (
    collect_activation_column_energy,
    iter_quantization_targets,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(5, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 6, bias=True), nn.Linear(6, 4, bias=False)])
        self.lm_head = nn.Linear(4, 5, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        h = self.embed_tokens(input_ids)
        return self.lm_head(self.layers[1](self.layers[0](h)))


def _batch():
    return [{"input_ids": torch.tensor([[0, 1, 2, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}]


def test_targets_have_expected_layout_and_are_copies():
    model = Tiny()
    energy = collect_activation_column_energy(model, _batch())
    targets = list(iter_quantization_targets(model, energy))

    assert [t.name for t in targets] == ["layers.0", "layers.1"]
    assert targets[0].weight.shape == (6, 4) and targets[0].importance.shape == (4,)
    assert targets[1].weight.shape == (4, 6) and targets[1].importance.shape == (6,)
    assert all(t.weight.dtype == torch.float32 and t.importance.dtype == torch.float32 for t in targets)

    before = model.layers[0].weight.clone()
    targets[0].weight.zero_()  # a consumer mutating its copy must not touch the model
    assert torch.equal(model.layers[0].weight, before)
    assert targets[0].module is model.layers[0]


def test_mismatched_energy_raises():
    model = Tiny()
    energy = collect_activation_column_energy(model, _batch())
    energy.pop("layers.1")
    with pytest.raises(ValueError, match="missing"):
        list(iter_quantization_targets(model, energy))


def test_non_finite_importance_raises():
    model = Tiny()
    energy = collect_activation_column_energy(model, _batch())
    energy["layers.0"][0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        list(iter_quantization_targets(model, energy))
