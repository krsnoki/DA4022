# Backbone model card: Qwen3.5-4B (4-bit NF4)

- **Registry ID:** `text.qwen3_5_4b.post_trained`
- **Modality:** multimodal (language model with a vision encoder); used text-only
- **Architecture:** `qwen3_5` (hybrid of gated linear-attention and full-attention layers)
- **Pretraining regime:** mixed (pre-training and post-training)
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Hub `main` on 2026-11-26; last modified 2026-03-02)
- **License/usage terms:** Apache-2.0
- **Tokenizer/processor revision:** same repository and revision as the weights
- **Parent backbone:** not verified
- **Input/output contract:** chat-template prompt with `enable_thinking=False`; `GenerativeBackbone.generate` returns text, token IDs, decoded token count and stop status; `score_next_options` returns log-softmax over option labels
- **Weight size:** about 9.3 GB of bf16 safetensors on disk (the download log shows 9.32 GB reconstructed). Loaded as 4-bit NF4 with bf16 compute via bitsandbytes; `get_model_metadata()['num_parameters']` reports 2.59 billion because bitsandbytes packs two 4-bit weights per stored element, so that count is not the model's parameter count.

## Loading (used for the reported SCPS run)

```yaml
backbone:
  type: hf
  id: Qwen/Qwen3.5-4B
  revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  dtype: bfloat16
  device: cuda
  kwargs:
    model_class: AutoModelForMultimodalLM
    quantization: bnb-4bit
```

Tested environment: Windows 10.0.26200, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 610.91, Python 3.11.18, torch 2.11.0+cu128, transformers 5.24.1, accelerate 1.16.0, bitsandbytes 0.50.2. Load time about 28 to 75 s from a local cache.

## Caveats

- **Slow kernels on Windows.** Transformers falls back to reference PyTorch kernels for the linear-attention layers (`flash-linear-attention` and `causal-conv1d` need Triton). Expect several seconds per short generation call on a laptop GPU.
- **Memory.** Weights plus activations peak at about 4.0 to 4.8 GB, slightly above a 4 GB card, so the driver spills into shared memory. Batches of 16 samples ran out of memory; batches of 4 worked.
- **First load on Windows** may fail with `The paging file is too small` (OS error 1455); a retry, or a larger page file, fixes it.
- **Accuracy with thinking disabled** and an answer-only prompt is much lower than the published reasoning-mode score (38.7% greedy on the SCPS MMLU-Pro sample).
