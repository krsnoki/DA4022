# Supervised Backbones

Store wrappers/registry entries for models whose reported pretrained weights were trained using labelled targets (or another explicitly supervised objective). Keep architecture code in its upstream dependency where possible; this folder should mainly contain stable identifiers, config adapters, and smoke tests.

## Suggested first baseline

- ResNet-50 with TorchVision ImageNet-1K weights: [official model/weights documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- Additional supervised vision architectures may be selected from the [TorchVision model list](https://pytorch.org/vision/stable/models.html) or [timm registry](https://huggingface.co/docs/timm/index).

Choose one model as the smoke-test backbone before adding many architectures. Record the exact weight enum/checkpoint and transform pipeline.

## Each entry must include

- `id`, architecture, modality, and intended task;
- training regime and labelled dataset/objective, only when verified;
- checkpoint URI/revision and licence;
- preprocessing and input contract;
- classifier/head structure and output semantics;
- a small load + forward test;
- a matching unmodified baseline benchmark.

## Suggested files

```text
Supervised/
├── README.md
├── registry.yaml
├── vision/
└── tests/
```

Do not commit checkpoints or cache directories.
