# Backbone model card: Qwen3-1.7B

- **Registry ID:** `text.qwen3_1_7b.instruct`
- **Modality:** text
- **Architecture:** decoder-only transformer (Hugging Face class `Qwen3ForCausalLM`), grouped-query attention with 16 query heads and 8 key/value heads, 28 layers, 32,768-token context length.
- **Pretraining regime:** self_supervised, then instruction/RL post-training (see below).
- **Training objective/data:** Pretrained with next-token prediction, then post-trained (instruction tuning plus RL) per the official model card. Qwen3-1.7B supports switching between "thinking" and "non-thinking" generation within the same model via the chat template's `enable_thinking` flag; it is a finetune of `Qwen/Qwen3-1.7B-Base`. Further pretraining/post-training data details are not published.
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen3-1.7B. Revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **License/usage terms:** Apache-2.0 (per the repository's `license` tag).
- **Tokenizer/processor revision:** same repository and revision as the weights (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`).
- **Parent backbone (if fine-tuned):** `Qwen/Qwen3-1.7B-Base` (the repository's `base_model` tag).
- **Input/output contract:** `input_ids (B, T) int64` -> `logits (B, T, vocab) float`. For chat/reasoning use, prompts go through `tokenizer.apply_chat_template(..., enable_thinking=True)`; with thinking enabled the generation contains a `<think>...</think>` block (token id `151668` closes it) followed by the final answer. `enable_thinking=False` suppresses the think block entirely.
- **Parameter count / weight size:** 2,031,739,904 parameters, all BF16, per the repository's `safetensors` metadata (`parameters.BF16` / `total`). This is the released weight dtype; we do not commit the weights.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=REV)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", revision=REV, torch_dtype="float16"
)
```

We load in `float16`, not the released `bfloat16`, because our Kaggle runtime is an NVIDIA T4. The T4 (Turing, sm_75) has no native bf16 compute support, so running the model in bf16 either falls back to slow emulation or errors depending on the kernel; fp16 runs on the T4's tensor cores.

## Preprocessing and outputs

Raw text/chat turns go through the model's own tokenizer and chat template; no separate processor. For reasoning evaluation, build the prompt with `enable_thinking=True` and use the documented thinking-mode sampling parameters rather than greedy decoding:

| Mode | Temperature | Top-p | Top-k | Min-p |
|---|---|---|---|---|
| Thinking (`enable_thinking=True`) | 0.6 | 0.95 | 20 | 0 |

These are the values from the model card's "Best Practices" section (`generation_config.json` defaults); greedy decoding is explicitly discouraged there, since it can cause endless repetition.

To separate the reasoning trace from the final answer, split the generated token ids on the last occurrence of token id `151668` (`</think>`): everything before it is `thinking_content`, everything from it onward is `content`.

## Smoke test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=REV)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", revision=REV, torch_dtype="float16")
text = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=True,
)
inputs = tok([text], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
assert out.shape[0] == 1
```

Expected output: a `(1, prompt_len + generated_len)` id tensor; decoding the new tokens should contain a `<think>` block followed by a short answer. This downloads the actual weights (about 4.1 GB in bf16/fp16, two safetensors shards), so it is not run as part of the automated test suite; run it manually once on hardware with the weights cached.

## Caveats

- We run this model only via Kaggle's T4 GPU (16 GB). fp16 inference fits comfortably at this size, but T4 lacks bf16, so any numbers compared against a bf16 run elsewhere are not bit-identical.
- Weights are not committed to this repository; the loading command above downloads them to the Hugging Face cache on first use.
- Pretraining/post-training data composition and the RL recipe are not disclosed beyond the high-level "Pretraining & Post-training" description on the model card.

## Registry entry

`Backbones/registry.yaml` does not exist on `main` yet (as of this PR), so this entry is documentation only; add it to the real registry file once one exists.

```yaml
id: text.qwen3_1_7b.instruct
name: Qwen3-1.7B
modality: text
architecture: Qwen3ForCausalLM  # decoder-only, GQA (16 Q heads / 8 KV heads), 28 layers
pretraining_regime: mixed       # self_supervised pretraining, then instruction/RL post-training
objective: next-token prediction pretraining, then instruction/RL post-training; data not disclosed
weights_source: huggingface
weights_uri: https://huggingface.co/Qwen/Qwen3-1.7B
revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
license: Apache-2.0
num_parameters: 2031739904       # safetensors total, BF16
input_spec:
  context_length: 32768
  weight_precision: BF16 (released); we run float16 on T4
preprocessing: tokenizer in the same repository at revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; chat template supports enable_thinking
parent_backbone: Qwen/Qwen3-1.7B-Base
fine_tuning_dataset: unknown (not disclosed)
notes: Thinking-mode sampling per the model card -- temperature 0.6, top_p 0.95, top_k 20, min_p 0. We run fp16 on Kaggle T4 16 GB since the T4 has no native bf16 support. Weights not committed.
```
