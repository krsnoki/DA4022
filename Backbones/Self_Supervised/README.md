# Self-Supervised Backbones

Register models trained with learning targets constructed from the data or from paired views/modalities rather than conventional human class labels. Typical objectives include masked-image reconstruction, masked-language modelling, next-token prediction, and contrastive representation learning.

## Suggested starting sources

- [DINOv2](https://github.com/facebookresearch/dinov2) — self-supervised visual features.
- [MAE](https://github.com/facebookresearch/mae) — masked image modelling.
- [OpenAI CLIP](https://github.com/openai/CLIP) — contrastive image/text pretraining; capture the training regime as described by its model documentation rather than calling it simply “unsupervised.”
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) and [model hub](https://huggingface.co/models) — find text encoders/decoders and pin a precise checkpoint revision. For language models, document whether the checkpoint is pretrained with masked-token or autoregressive next-token prediction.

## Metadata requirements

Record the exact objective, modality, source dataset when publicly documented, checkpoint revision, tokenizer/processor, licence, and any model-specific input requirements. Use `pretraining_regime: self_supervised` unless the source explicitly supports a different classification. If details are not documented, set them to `unknown` instead of inferring them.

## Validation

- Smoke-test loading and forward/generate behavior.
- For representation models, specify which layer/pooled output is exposed.
- For generative models, specify tokenizer, context limit if documented, and generation defaults.
- Record baseline quality before applying compression or runtime adaptations.
