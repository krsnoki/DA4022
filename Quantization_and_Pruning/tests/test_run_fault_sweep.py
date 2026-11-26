import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Quantization_and_Pruning.run_fault_sweep import (
    load_resnet18_checkpoint,
    run_fault_sweep,
)
from Quantization_and_Pruning.train_cifar10 import build_resnet18


def test_training_checkpoint_loads_into_benchmark_model(tmp_path):
    checkpoint_path = tmp_path / "cifar10.pt"
    model = build_resnet18()
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    loaded_model = load_resnet18_checkpoint(str(checkpoint_path))

    assert loaded_model.fc.out_features == 10
    assert loaded_model.conv1.kernel_size == (3, 3)
    assert isinstance(loaded_model.maxpool, nn.Identity)


def test_sweep_uses_requested_images_and_injects_k_faults_across_model():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    original_weights = model.weight.detach().clone()
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    targets = torch.tensor([0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    results = run_fault_sweep(
        model,
        loader,
        fmt="e4m11",
        k=1,
        bit_group="sign",
        n_trials=2,
        n_images=5,
        seed=42,
    )

    assert results["n_images"] == 5
    assert results["n_trials"] == 2
    assert [row["method"] for row in results["metrics_rows"]] == [
        "FP32_baseline",
        "e4m11_no_faults",
        "e4m11_fault_injection",
    ]
    assert all(row["quality_value"] >= 0 for row in results["metrics_rows"])
    assert all(len(trial["metadata"]["indices"]) == 1 for trial in results["trial_results"])
    assert all(
        trial["metadata"]["n_weights"] == model.weight.numel()
        for trial in results["trial_results"]
    )
    assert np.isfinite(results["clean_accuracy"])
    assert np.array_equal(model.weight.detach(), original_weights)
