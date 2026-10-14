# Backbones and Baseline Models

## Purpose

This directory contains **reusable base models and their metadata**, not adaptation algorithms. All compression/routing/LoRA/attention experiments should point to a registered backbone ID. Avoid hard-coding checkpoint downloads and preprocessing inside paper methods.

## Recommended categories

Use these categories because “supervised vs unsupervised” alone is often too coarse:

1. `Supervised/` — weights trained using human/ground-truth labels for a task or classification objective (e.g. ImageNet-supervised vision models).
2. `Self_Supervised/` — labels are derived from the data/task objective itself (masked-language modelling, next-token prediction, masked-image reconstruction, contrastive image/text alignment). This is commonly called “unsupervised” informally, but should be recorded as self-supervised when accurate.
3. `Generative_Pretrained/` — broad family grouping for autoregressive language/diffusion/generative backbones. Record the underlying objective and training regime in metadata; it may be self-supervised.
4. `Task_Finetuned/` — a separate variant built from a parent backbone and adapted to an explicitly named dataset/task. Keep the base model and tuning provenance.

The grouping folders are for discoverability. The registry metadata is the source of truth; the same architecture may have supervised and self-supervised checkpoints.

## Starting points and model sources

Use official model libraries/model cards and pin an exact model revision. Suggested starter sources:

- [TorchVision pretrained models](https://pytorch.org/vision/stable/models.html) — supervised vision baselines such as ResNet and ViT variants.
- [timm documentation and model registry](https://huggingface.co/docs/timm/index) — broad vision model zoo, including models trained with different recipes.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — encoder, decoder-only, and multimodal model architectures and processors.
- [DINOv2 official repository](https://github.com/facebookresearch/dinov2) — self-supervised visual representation baseline.
- [MAE official repository](https://github.com/facebookresearch/mae) — masked-image-modeling baseline.
- [OpenAI CLIP repository](https://github.com/openai/CLIP) — image/text contrastive pretrained model; describe the supervision regime precisely in the model card.
- [Hugging Face model hub](https://huggingface.co/models) — search model cards, exact revisions, licence, and compatible processor/tokenizer.

These are candidate sources, not an endorsement of any one checkpoint. Review licences, terms, training-data disclosures, and hardware needs before use. Do not download the whole model zoo.

## Minimum model registry fields

Every backbone entry should record:

```yaml
id: vision.resnet50.imagenet1k.supervised
name: ResNet-50 ImageNet-1K
modality: vision                 # text | vision | audio | multimodal | other
architecture: resnet50
pretraining_regime: supervised  # supervised | self_supervised | unsupervised_legacy | mixed | unknown
objective: image classification with ImageNet-1K labels
weights_source: torchvision
weights_uri: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
revision: <exact-version-or-commit>
license: <license-from-model-card>
num_parameters: <integer-if-verified>
input_spec:
  image_size: [224, 224]
  channels: 3
  dtype: float32
preprocessing: <transform or processor id>
parent_backbone: null
fine_tuning_dataset: null
notes: Verify preprocessing and classifier-head semantics before evaluating.
```

Replace placeholders with verified values. Never guess licence, revision, parameter count, or training data.

## Stable interface

A backbone wrapper should ideally expose:

- `load(config, device, dtype)`
- `preprocess(raw_inputs)` or an associated processor/tokenizer
- `forward(inputs)` (or a documented `generate()` interface for generation models)
- `get_model_metadata()`
- `save_adaptation_state(path)` where the model's licence permits it

Keep model-specific tokenizers/processors with the model wrapper. Keep dataset iteration in `Dataloaders/`.

## Baseline rules

- Pin the checkpoint revision and all processor/tokenizer versions.
- Keep the original checkpoint unmodified; methods should create a transformed copy or a well-defined adapter.
- Make output head changes explicit (e.g. removing/replacing a classifier head).
- Record parameter count, weight storage size, expected input shape, precision, licence, and source URL.
- Separate training-time and inference-time costs.
- Include a small, deterministic smoke test before full benchmark runs.

See the category-specific README files in this directory and the [dataset protocol](../Dataloaders/README_dataloader.md).
