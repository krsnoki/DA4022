# Task-Finetuned Backbones

This folder contains **variants** produced by fine-tuning an existing backbone. Every entry must reference its parent backbone and document what changed. Do not duplicate the base model's registry data by hand; link it by ID.

## Required metadata

```yaml
id: text.bert.sentiment.example
parent_backbone: text.bert.base.example
fine_tuning_dataset: <dataset-id>
fine_tuning_split: train
objective: <verified objective>
method: full_finetune  # full_finetune | lora | adapter | other
checkpoint_uri: <model card or artifact URI>
revision: <pinned revision>
license: <verified license>
```

Use placeholders only in templates; fill them before merging a real registry entry.

## Rules

- Distinguish task-finetuned weights from a base pretrained model.
- Record the exact train/validation/test split and prevent evaluation leakage.
- If fine-tuning uses LoRA/adapters, register the adapter config under `LoRA/` and link it here rather than merging the adapter invisibly.
- Never publish weights or datasets without confirming their licence and sharing permissions.
- Include a reproducible evaluation command and the unadapted parent-backbone baseline.
