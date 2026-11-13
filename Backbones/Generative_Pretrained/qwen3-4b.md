# Backbone model card: Qwen3-4B

- **Registry ID:** `text.qwen3-4b.post_trained.mixed` (see [`../registry.yaml`](../registry.yaml))
- **Modality:** text
- **Architecture:** `Qwen3ForCausalLM`, a decoder-only transformer. 36 layers, hidden size 2560, intermediate size 9728, 32 query heads and 8 KV heads (GQA), head dim 128, vocab 151,936, tied input/output embeddings, RoPE theta 1e6 (from `config.json`).
- **Pretraining regime:** mixed
- **Training objective/data:** causal language model. The model card lists "Training Stage: Pretraining & Post-training" and names `Qwen/Qwen3-4B-Base` as the base model. The [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) describes training data and post-training; those details were not re-verified for this card.
- **Weights URI and exact revision:** <https://huggingface.co/Qwen/Qwen3-4B> @ `1cfa9a7208912126459214e8b04321603b3df60c`
- **License/usage terms:** Apache-2.0 (model card front matter and the `LICENSE` file in the repo)
- **Tokenizer/processor revision:** tokenizer from the same repo and revision. EOS token `<|im_end|>` (id 151645).
- **Parent backbone (if fine-tuned):** none in this registry. The upstream base model is `Qwen/Qwen3-4B-Base`, which is not registered.
- **Input/output contract:** `input_ids` `LongTensor[1, L]` (batch size 1 in BASTION) → `logits` `bf16[1, L, 151936]`. Generation goes through the speculative-decoding loops in `Speculative_Decoding/BASTION`, not `model.generate`.
- **Parameter count / weight size:** 4,022,468,096 parameters, counted from the safetensors headers and confirmed by loading. 8,044,936,192 bytes in bf16 safetensors (3 shards).

## Paired draft model

For BASTION and DFlash, this target is paired with `z-lab/Qwen3-4B-DFlash-b16` @ `b74e3a329c4d963783143b1e970d95b002be72bd` (registry ID `text.qwen3-4b-dflash-b16.drafter.unknown`, MIT). The drafter is a block-diffusion draft model trained by z-lab (DFlash, [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)). It has 5 layers, block size 16, and 537,427,200 parameters. It reads target hidden states from layers [1, 9, 17, 25, 33] and uses the target's embedding and LM head. Its model card does not state the training objective or data.

## Loading

Tested on CPU, offline, from `Speculative_Decoding/BASTION/upstream` with the BASTION venv:

```bash
cd Speculative_Decoding/BASTION/upstream
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 ../.venv/bin/python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlashDraftModel
m = "Qwen/Qwen3-4B"; rev = "1cfa9a7208912126459214e8b04321603b3df60c"
tok = AutoTokenizer.from_pretrained(m, revision=rev)
model = AutoModelForCausalLM.from_pretrained(m, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
draft = DFlashDraftModel.from_pretrained("z-lab/Qwen3-4B-DFlash-b16", revision="b74e3a329c4d963783143b1e970d95b002be72bd",
                                         dtype=torch.bfloat16, attn_implementation="sdpa").eval()
text = tok.apply_chat_template([{"role": "user", "content": "What is 2+3?"}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(text, return_tensors="pt")
with torch.no_grad():
    print(tuple(ids.shape), tuple(model(ids).logits.shape), draft.block_size)
EOF
```

The BASTION harness loads the model the same way, without `revision=`. It resolves the cached snapshot listed above, then moves the model to `cuda`.

## Preprocessing and outputs

- **Non-thinking mode.** Prompts are built with `apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`, which ends the prompt with `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Thinking mode is not used. The DFlash drafters were trained for non-thinking mode, and upstream BASTION raises an error if `--enable-thinking` is passed for Qwen3-4B/8B.
- Messages contain user turns only (no system prompt). For multi-turn MT-Bench, the turn-1 assistant message is the AR baseline's output (upstream behaviour).
- Generation stops at `tokenizer.eos_token_id` (151645) or at `max_new_tokens` (2048 in this reproduction).
- The model card's recommended sampling defaults (`generation_config.json`: temperature 0.6, top-p 0.95, top-k 20) are **not** used. This reproduction decodes greedily (temperature 0).
- Weights are bf16. Attention is SDPA for every mode in this reproduction, because flash-attn is not installed.

## Smoke test

The loading command above, run on CPU, printed `(1, 19) (1, 19, 151936) 16`: input shape, logits shape, drafter block size.

## Caveats

- The model card gives 32,768 tokens of native context (131,072 with YaRN); `config.json` has `max_position_embeddings` 40960. BASTION prompts plus 2048 new tokens stay inside this.
- The cost-model and calibration caches in BASTION are keyed per (GPU, model). Results depend on the GPU and are not portable across machines.
- Weights are about 8 GB in bf16. Do not commit them. They live in the local Hugging Face cache.
