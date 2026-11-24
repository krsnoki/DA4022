# Dataloader — C4 Calibration Split (Paper #12)

> Registered for: Paper #12 — Rethinking Layer Redundancy
> Registered by: Jash Vaidya

## Dataset Card

| Field | Value |
|---|---|
| **Dataset ID** | `allenai/c4` |
| **Subset / config** | `en` |
| **Split used** | `train` (calibration only — never evaluation) |
| **HuggingFace link** | https://huggingface.co/datasets/allenai/c4 |
| **Licence** | ODC-BY |
| **Task type** | Language modelling (perplexity calibration) |
| **Samples used** | 128 (see `config.yaml`) |
| **Max token length** | 512 |
| **Seed** | 42 |

## Purpose

Used as the calibration dataset for the **perplexity objective** in Paper #12.
The model is NOT evaluated on C4 — it is only used to measure which layers
to prune under a language-modelling signal.

Evaluation perplexity is measured on **WikiText-2** (separate dataset, see `dataloader_wikitext2_paper12.md`).

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
calibration_samples = dataset.take(128)
```

## Notes

- Keep calibration and evaluation splits strictly separate.
- Do not commit any downloaded dataset files to this repository.
