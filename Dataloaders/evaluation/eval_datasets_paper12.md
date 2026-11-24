# Dataloader — Evaluation Datasets (Paper #12)

> Registered for: Paper #12 — Rethinking Layer Redundancy
> Registered by: <!-- YOUR NAME HERE e.g. John Doe (@johndoe) -->

## Datasets Used for Evaluation

### 1. HellaSwag

| Field | Value |
|---|---|
| **Dataset ID** | `hellaswag` |
| **Split** | `validation` |
| **HuggingFace link** | https://huggingface.co/datasets/Rowan/hellaswag |
| **Task type** | Multiple-choice commonsense reasoning |
| **Metric** | Zero-shot accuracy (%) |
| **Examples** | 10,042 (full validation set) |

```python
from datasets import load_dataset
dataset = load_dataset("hellaswag", split="validation")
```

### 2. BoolQ

| Field | Value |
|---|---|
| **Dataset ID** | `boolq` |
| **Split** | `validation` |
| **HuggingFace link** | https://huggingface.co/datasets/google/boolq |
| **Task type** | Yes/No reading comprehension |
| **Metric** | Zero-shot accuracy (%) |
| **Examples** | 3,270 (full validation set) |

```python
from datasets import load_dataset
dataset = load_dataset("boolq", split="validation")
```

### 3. WikiText-2 (Perplexity Evaluation)

| Field | Value |
|---|---|
| **Dataset ID** | `wikitext` |
| **Config** | `wikitext-2-raw-v1` |
| **Split** | `test` |
| **HuggingFace link** | https://huggingface.co/datasets/wikitext |
| **Task type** | Language modelling |
| **Metric** | Perplexity |

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
```

## Notes

- These datasets are evaluation-only. Do not use them for calibration.
- All datasets are publicly available and require no special access.
- Do not commit any downloaded dataset files to this repository.
