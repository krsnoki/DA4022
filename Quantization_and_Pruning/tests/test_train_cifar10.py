import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Quantization_and_Pruning.train_cifar10 import train_model


def test_train_model_updates_weights_and_returns_finite_loss():
    model = nn.Linear(2, 2)
    original_weights = model.weight.detach().clone()
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    losses = train_model(
        model,
        loader,
        epochs=1,
        learning_rate=0.1,
        device=torch.device("cpu"),
    )

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
    assert not torch.equal(model.weight.detach(), original_weights)


def test_train_model_rejects_empty_loader():
    model = nn.Linear(2, 2)

    try:
        train_model(
            model,
            [],
            epochs=1,
            learning_rate=0.1,
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "no examples" in str(error)
    else:
        raise AssertionError("empty training loader should be rejected")
