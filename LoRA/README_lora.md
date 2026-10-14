# LoRA and Parameter-Efficient Adaptation

## Purpose

LoRA is a useful **shared baseline** for testing whether resource-aware methods can preserve adaptation quality while reducing trainable parameters or adaptation storage. Keep the adapter mechanism separate from the base model registry and dataset code.

## Paper classification

The official AdaptFM accepted-paper titles do not identify a clear primary LoRA paper. Therefore, this module is currently a baseline/infrastructure module rather than an assigned group of AdaptFM paper reproductions. Do not attach a paper here unless the paper's method actually uses or studies LoRA; when it does, link it in [`PAPER_INDEX.md`](../PAPER_INDEX.md) and cross-reference its primary module.

## Suggested structure

```text
LoRA/
├── README_lora.md
├── configs/
├── adapters/
├── merge_unmerge/
├── tests/
└── experiments/
```

## Required configuration

Record base backbone ID/revision, target modules, rank `r`, scaling `alpha`, dropout, bias behavior, task objective, training dataset/split, seed, trainable parameter count, adapter size, and merge/unmerge mode. Use verified defaults from the chosen implementation library rather than assuming all architectures expose identical target-module names.

## Baselines and tests

- Compare frozen-base inference, LoRA-adapted inference, and (when affordable) full fine-tuning.
- Confirm that the adapter is actually trainable and that base weights are frozen when intended.
- Verify save/load round trips and merged vs. unmerged output equivalence within an explicit tolerance.
- Report adaptation time, peak training memory, trainable parameters, adapter storage size, task quality, and inference latency after merge/unmerge.
- Keep model/dataset loading delegated to `Backbones/` and `Dataloaders/`.

## Suggested references

- [Original LoRA paper (arXiv)](https://arxiv.org/abs/2106.09685)
- [PEFT documentation](https://huggingface.co/docs/peft/index)

These are general method references, not claims that they are AdaptFM workshop papers.
