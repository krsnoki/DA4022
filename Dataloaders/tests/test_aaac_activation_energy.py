import pytest

torch = pytest.importorskip("torch")
from torch import nn

from Dataloaders.calibration.activation_energy import (
    collect_activation_column_energy,
    eligible_linear_modules,
)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(3, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.lm_head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
            )

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not self.training
        hidden = self.embed_tokens(input_ids)
        hidden = self.layers[0](hidden)
        return self.lm_head(hidden)


def test_hooks_sum_valid_decoder_inputs_and_are_removed():
    model = TinyCausalLM()
    assert set(eligible_linear_modules(model)) == {"layers.0"}

    energy = collect_activation_column_energy(
        model,
        [
            {
                "input_ids": torch.tensor([[0, 1, 2]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }
        ],
    )

    assert model.training is False
    assert torch.equal(energy["layers.0"], torch.tensor([10.0, 20.0]))
    assert len(model.layers[0]._forward_pre_hooks) == 0
