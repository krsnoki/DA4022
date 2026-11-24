# Backbone Registration — LLaMA-2-7B

> Registered for: Paper #12 — Rethinking Layer Redundancy
> Registered by: <!-- YOUR NAME HERE e.g. John Doe (@johndoe) -->

## Model Card

| Field | Value |
|---|---|
| **Backbone ID** | `meta-llama/Llama-2-7b-hf` |
| **Architecture** | Decoder-only Transformer (LLaMA) |
| **Parameters** | 7B |
| **Layers** | 32 Transformer blocks |
| **Hidden size** | 4096 |
| **Attention heads** | 32 |
| **Pretraining regime** | `self_supervised` (next-token prediction) |
| **Pretraining data** | 2T tokens (publicly undisclosed mix) |
| **HuggingFace revision** | `main` |
| **Licence** | LLaMA 2 Community Licence |
| **Weights access** | Request at https://huggingface.co/meta-llama/Llama-2-7b-hf |

## Notes

- Used as the base model for depth pruning experiments in Paper #12.
- No task fine-tuning applied — raw pretrained weights used throughout.
- For CPU smoke tests, replace with a randomly initialised model of the same architecture (no download needed).
- Do not commit model weights to this repository.
