# Backbone model card: Gemma 2 2B (pretrained)

- **Registry ID:** `text.gemma_2_2b.base`
- **Modality:** text
- **Architecture:** decoder-only transformer (`Gemma2ForCausalLM`), GQA, explicit `head_dim`, tied embeddings, logit soft-capping
- **Pretraining regime:** self_supervised (next-token prediction; see the Gemma 2 report for distillation details)
- **Weights URI and exact revision:** https://huggingface.co/google/gemma-2-2b. Revision: **unpinned, set `backbone.revision`**.
- **License/usage terms:** Gemma Terms of Use. **Gated**: each user must accept the terms on Hugging Face and authenticate (`huggingface-cli login`). Never commit tokens.
- **Tokenizer/processor revision:** same repo and revision as the weights
- **Parent backbone (if fine-tuned):** none
- **Input/output contract:** `input_ids (B, T) int64` → `logits (B, T, vocab) float`
- **Parameter count / weight size:** record from the loaded model

## Loading

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", revision=REV, torch_dtype="bfloat16")
```

Requires `transformers>=4.42`.

## Smoke test

`python -m pytest Quantization_and_Pruning/tests/test_cafeq_model.py -k gemma2` uses a tiny random Gemma 2 config; no download.

## Caveats

This is the paper's main model (Tables 1–4, 9). Use `configs/cafeq/gemma2_2b_w4_perchannel_paper.yaml`; it expects a GPU.
