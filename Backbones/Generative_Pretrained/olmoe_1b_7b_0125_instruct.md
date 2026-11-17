# Backbone model card: OLMoE-1B-7B-0125-Instruct

- **Registry ID:** text.olmoe-1b-7b-0125-instruct
- **Modality:** text
- **Architecture:** Decoder-only Mixture-of-Experts transformer (Hugging Face class `OlmoeForCausalLM`). 16 layers, hidden size 2048, 64 experts per MoE layer, 8 experts active per token, max 4096 positions. The router applies softmax over all 64 experts, then takes the top-8. The 8 weights are not renormalized (`norm_topk_prob: false`). Vocabulary size 50304 (`vocab_size` in config.json).
- **Pretraining regime:** self_supervised (next-token prediction), followed by instruction tuning.
- **Training objective/data:** Next-token prediction pretraining, then post-training. The model card states the models were trained on the Dolma dataset ("a mix of publicly available, synthetic and human-created datasets"). Post-training, as stated on the card: (1) supervised finetuning on an OLMo-specific variant of the Tulu 3 dataset (`allenai/tulu-3-sft-olmo-2-mixture`); (2) DPO on `allenai/olmo-2-1124-13b-preference-mix`; (3) RLVR on `allenai/RLVR-GSM`. Other training details (compute, filtering, exact mixture sizes): unknown.
- **Weights URI and exact revision:** https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct. Revision: `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
- **License/usage terms:** Apache-2.0, for research and educational use. The model card also states that the model "has been fine-tuned using a dataset mix with outputs generated from third party models" and is therefore subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) as well. Not gated.
- **Tokenizer/processor revision:** Shipped in the same repository at the same revision (`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`). Chat template applied with `tokenizer.apply_chat_template`.
- **Parent backbone (if fine-tuned):** `allenai/OLMoE-1B-7B-0125-DPO`, per the card's "Finetuned from" field. The card's model tree names the base checkpoint `allenai/OLMoE-1B-7B-0125`; the card contains both statements, and the DPO checkpoint is the stated immediate parent.
- **Input/output contract:** Token ids in, next-token logits out. We use greedy decoding with a KV cache.
- **Parameter count / weight size:** 6,919,161,856 parameters (BF16 safetensors). Repository size about 13.8 GB. Active parameters per token: unknown (the model card gives only "7B params"; the "1B" in the model name is not quantified on the card).

## Loading

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0125-Instruct",
    revision="b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
    torch_dtype=torch.bfloat16,
)
```

The repository loader is `HFCausalLMBackbone.load("text.olmoe-1b-7b-0125-instruct")` in `Backbones/Generative_Pretrained/hf_causal_lm.py`. It is the current way to load this backbone; the `from_pretrained` call above shows the underlying Hugging Face call.

Versions: transformers 5.19.0, torch 2.13.0+cu129.

## Preprocessing and outputs

Tokenization and the chat template are applied with `tokenizer.apply_chat_template` from the same repository and revision. Input is a sequence of token ids. Output is next-token logits of shape `[batch, sequence, 50304]`. Decoding is greedy with a KV cache. Output fields beyond logits: unknown.

## Smoke test

```bash
python -m Mixture_of_Experts.experiments.xshare.smoke_test --batch-size 8
```

The script loads the model, generates for 8 GSM8K questions, and checks that XShare with every expert kept reproduces the baseline output exactly. It prints `SMOKE_OK` on success.

## Caveats

- This model runs with plain PyTorch through Hugging Face transformers, not a serving engine. Throughput numbers are therefore not comparable with the XShare paper's vLLM numbers.
- Hardware used: one NVIDIA RTX PRO 4500 Blackwell (32 GB), driver 570.211.
- Further training details (data, compute, and filtering) are unknown.

## Registry entry

```yaml
id: text.olmoe-1b-7b-0125-instruct
name: OLMoE-1B-7B-0125-Instruct
modality: text                   # text | vision | audio | multimodal | other
architecture: OlmoeForCausalLM   # decoder-only Mixture-of-Experts transformer, 16 layers, hidden 2048, 64 experts, top-8 active, vocab 50304
pretraining_regime: self_supervised
objective: next-token prediction pretraining (Dolma), then SFT (Tulu 3 OLMo-specific mixture), DPO (olmo-2-1124-13b-preference-mix), RLVR (RLVR-GSM)
weights_source: huggingface
weights_uri: https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct
revision: b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e
license: apache-2.0 (also subject to the Gemma Terms of Use; see the licence line above)
num_parameters: 6919161856       # BF16 safetensors; active parameters per token: unknown
input_spec:
  format: token ids; chat template applied with tokenizer.apply_chat_template
  weight_precision: BF16
preprocessing: tokenizer shipped in the same repository at revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e
parent_backbone: allenai/OLMoE-1B-7B-0125-DPO (card "Finetuned from"; model tree lists allenai/OLMoE-1B-7B-0125 as base)
fine_tuning_dataset: allenai/tulu-3-sft-olmo-2-mixture (SFT); allenai/olmo-2-1124-13b-preference-mix (DPO); allenai/RLVR-GSM (RLVR)
notes: Run with plain PyTorch via Hugging Face transformers, not a serving engine; throughput is not comparable with the XShare paper's vLLM numbers. Hardware: NVIDIA RTX PRO 4500 Blackwell (32 GB), driver 570.211. Smoke test: python -m Mixture_of_Experts.experiments.xshare.smoke_test --batch-size 8 (prints SMOKE_OK).
