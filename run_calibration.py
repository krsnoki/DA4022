"""Run the Qwen2.5-0.5B C4 calibration pass and save activation energies."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Backbones.Generative_Pretrained import load_qwen25_05b
from Dataloaders.calibration import (
    collect_activation_column_energy,
    iter_c4_calibration_batches,
    iter_quantization_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect C4 activation-column energies for Qwen2.5-0.5B and "
            "save them for a downstream quantization step."
        )
    )
    parser.add_argument(
        "--num-sequences", type=int, default=4,
        help="number of fixed-length C4 windows to use (default: 4)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=2048,
        help="tokens per C4 window (default: 2048)",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help="inference device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("qwen25_05b_c4_activation_energy.pt"),
        help="output file for the energy dictionary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep calibration dtype fixed so runs are comparable across devices.
    model, tokenizer = load_qwen25_05b(device=args.device, dtype=torch.float32)
    batches = iter_c4_calibration_batches(
        tokenizer,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    energy = collect_activation_column_energy(model, batches)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(energy, output_path)

    # Validate the energy map against all eligible model layers while keeping
    # target weight copies one layer at a time.
    target_count = sum(1 for _ in iter_quantization_targets(model, energy))
    print(f"Saved activation energy for {target_count} layers to {output_path}")
    print(
        f"Settings: sequences={args.num_sequences}, "
        f"sequence_length={args.sequence_length}, seed={args.seed}, "
        f"device={next(model.parameters()).device}, dtype=float32"
    )


if __name__ == "__main__":
    main()
