# Dataloaders and Dataset Protocol

## Purpose

Provide reusable, reproducible dataset loading and preprocessing. A dataloader should not know which adaptation paper is consuming it; a method receives batches through a stable dataset/batch contract. Dataset splits and transforms must be shared by baseline and method runs.

## Important distinction

Organize the data pipeline by **data role** (`train`, `calibration`, `validation`, `test`, `benchmark`, or `synthetic`), not only by whether the backbone is supervised/self-supervised. Both supervised and self-supervised backbones may need calibration or evaluation loaders.

## Suggested structure

```text
Dataloaders/
├── README_dataloader.md
├── registry.yaml
├── supervised/
├── self_supervised/
├── calibration/
├── evaluation/
├── synthetic/
├── transforms/
├── collators/
└── tests/
```

`supervised/` and `self_supervised/` are optional convenience groupings. The registry must still state the role and exact split. Do not copy the same underlying dataset implementation into several folders just to match a backbone category.

## Loader contract

Every dataset adapter should state:

- dataset ID, source URL, licence/access terms and dataset version;
- task, modality, fields/labels, split names, and split sizes if known;
- preprocessing/tokenization and transform version;
- deterministic seed and sampling strategy;
- batch shape, dtype, padding/masking behavior, and collator;
- whether data is used for training, calibration, validation, or final test;
- download/cache location and any required acceptance steps;
- a tiny smoke-test command that does not download the full dataset where possible.

## Calibration data is not evaluation data

Quantization, pruning, and some cache policies may require calibration examples. Calibration data must not silently draw from the held-out test split. Record the exact calibration sample count and seed. When a method is calibration-free, set `calibration_dataset: null` explicitly in its experiment config.

## Suggested dataset registry entry

```yaml
id: vision.imagenet1k.validation
name: ImageNet-1K validation
modality: vision
task: image_classification
source_url: <official dataset page>
license: <verified terms>
version: <exact version>
role: evaluation
split: validation
preprocessing_id: torchvision.resnet50.imagenet1k.v1
seed: 42
num_examples: <verified count>
labels_available: true
notes: Access and usage terms must be reviewed before downloading.
```

Do not invent dataset licence, version, or split counts; use `unknown` until verified.

## Supported loader patterns

1. **Supervised**: return `(inputs, target)` or a documented dictionary. Record label encoding and class mapping.
2. **Self-supervised**: return raw samples or multiple views; document augmentations and their random seeds.
3. **Calibration**: use fixed, deterministic samples; return inputs only if labels are not needed.
4. **Language modelling**: document tokenizer revision, truncation, context length, padding, attention mask, and whether perplexity uses shifted labels correctly.
5. **Vision-language/multimodal**: use the official processor; document image resize/crop, text tokenization, paired sample identity, and missing-modality behavior.
6. **Synthetic**: use only for smoke/performance microbenchmarks unless the experiment explicitly defines synthetic evaluation.

## Tests

- Verify sample keys and tensor shapes/dtypes.
- Verify deterministic sampling with a fixed seed.
- Verify train/validation/test separation.
- Verify transforms do not accidentally run stochastic augmentation at test time.
- Verify batching, padding, and truncation at boundary sizes.
- Keep large datasets and cache files out of Git; add the relevant local paths to `.gitignore`.

## Reproducibility note

An experiment config should point to `dataset_id`, `split`, `preprocessing_id`, `seed`, and `num_samples`—not contain a second implementation of the loader. See [`templates/DATASET_CARD_TEMPLATE.md`](../templates/DATASET_CARD_TEMPLATE.md).
