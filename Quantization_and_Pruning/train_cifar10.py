"""Train a ResNet-18 CIFAR-10 checkpoint for the fault-sweep experiment."""

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def build_resnet18() -> nn.Module:
    """Build the 10-class ResNet-18 used by the experiment."""
    import torchvision

    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def load_cifar10(
    data_dir: str,
    batch_size: int,
    max_train_samples: int = 0,
) -> Tuple[Iterable, int]:
    """Download and load the CIFAR-10 training split."""
    import torchvision
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    if max_train_samples:
        if max_train_samples <= 0 or max_train_samples > len(dataset):
            raise ValueError("max_train_samples must be within the training set")
        dataset = Subset(dataset, range(max_train_samples))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )
    return loader, len(dataset)


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list:
    """Train the model and return the mean training loss for each epoch."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.to(device)

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            total_examples += len(inputs)
        if total_examples == 0:
            raise ValueError("training loader returned no examples")
        losses.append(total_loss / total_examples)
        scheduler.step()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet-18")
    parser.add_argument("--output", type=Path, default=Path("artifacts/resnet18_cifar10.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, sample_count = load_cifar10(
        str(args.data_dir), args.batch_size, args.max_train_samples
    )
    model = build_resnet18()
    losses = train_model(
        model, train_loader, args.epochs, args.learning_rate, device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "metadata": {
                "architecture": "torchvision.resnet18_cifar10",
                "dataset": "CIFAR-10 train",
                "training_examples": sample_count,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "training_loss_by_epoch": losses,
            },
        },
        args.output,
    )
    print(f"checkpoint saved: {args.output}")
    print(f"training examples: {sample_count}; final training loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
