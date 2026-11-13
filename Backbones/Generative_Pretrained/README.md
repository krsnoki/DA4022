# Generative Pretrained Backbones

A discovery group for autoregressive language models, diffusion models, and other generative foundation models used in inference/adaptation experiments. Most text-only autoregressive foundation models are trained with a self-supervised next-token objective; diffusion models use their documented denoising or reconstruction objective. Record the actual regime in the registry rather than treating this directory name as a training label.

## Supported model families

Start with one small and one medium checkpoint whose licence and hardware fit the project. Use the [Transformers docs](https://huggingface.co/docs/transformers/index) and [Hugging Face model hub](https://huggingface.co/models) to identify an exact checkpoint, model card, tokenizer, revision, and inference interface.

Registered models (metadata in [`../registry.yaml`](../registry.yaml)):

| Model | Card | Used by |
|---|---|---|
| Qwen3-4B (`text.qwen3-4b.post_trained.mixed`) | [`qwen3-4b.md`](qwen3-4b.md) | BASTION target, with the drafter `z-lab/Qwen3-4B-DFlash-b16` |
| Qwen3-8B (`text.qwen3-8b.post_trained.mixed`) | [`qwen3-8b.md`](qwen3-8b.md) | BASTION target, with the drafter `z-lab/Qwen3-8B-DFlash-b16` |

## Required metadata

Model ID/revision, licence, modality, architecture, objective/regime, tokenizer or processor, context/window parameters when documented, parameter count when verified, weight format, generation defaults, and supported runtime. Do not claim that a model is small/fast solely from its name; measure loading, memory, prefill, and decode separately.

## Integration rule

Expose model loading and generation, but keep dataset prompts and batches in `Dataloaders/` and method-specific transformations in their owning module. Baseline generation settings must be pinned before comparing speculative decoding, KV policies, quantization, or stopping.
