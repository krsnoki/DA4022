# Backbone model card: Qwen2.5-0.5B-Instruct

- **Registry ID:** `text.qwen2_5_0_5b_instruct.self_supervised`
- **Modality:** text
- **Architecture:** decoder-only transformer (`Qwen2ForCausalLM`), 24 layers, hidden size 896, 14 attention heads, 2 key/value heads (grouped-query attention), head dimension 64, rotary position embeddings
- **Pretraining regime:** self_supervised
- **Training objective/data:** next-token prediction followed by instruction tuning; training data details are not disclosed in the model card (unknown)
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- **License/usage terms:** Apache-2.0 (from the model card)
- **Tokenizer/processor revision:** tokenizer shipped in the same repository and revision; prompts use the model's chat template
- **Parent backbone (if fine-tuned):** Qwen/Qwen2.5-0.5B
- **Input/output contract:** `input_ids` `[B, N]` int64 -> logits `[B, N, 151936]`; `past_key_values` holds per layer keys and values of shape `[B, 2, N, 64]`; configured context length 32768 tokens
- **Parameter count / weight size:** 494,032,768 parameters, stored as BF16 safetensors (about 0.99 GB)

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

rev = "7ae557604adf67be50417f59c2c2f167def9a775"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision=rev)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision=rev, torch_dtype=torch.float16).to("cuda").eval()
```

Use `torch.float32` on CPU. Weights are downloaded to the Hugging Face cache and are not committed.

## Preprocessing and outputs

A single user message is wrapped with `tokenizer.apply_chat_template(..., add_generation_prompt=True)` and tokenised without extra special tokens. Generation in the KV-eviction experiments is greedy, batch size 1, stopping at the model's EOS ids.

## Smoke test

```bash
python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/smoke_cpu.yaml
```

Prints one line per configuration; the `full` row is the unmodified model.

## Caveats

- Small model: long-context behaviour is weaker than the 7-8B models used in the papers that reference it, so it is a functional stand-in and not a replacement for those results.
- With only 2 KV heads per layer, per-head eviction decisions are coarser than in models with more KV heads.
- Loading, memory, prefill and decode must be measured on the target device; nothing here is a claim about speed.
