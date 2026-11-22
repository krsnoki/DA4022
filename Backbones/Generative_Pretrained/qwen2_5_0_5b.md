# Backbone model card: Qwen2.5-0.5B (base)

- **Registry ID:** `text.qwen2_5_0_5b.base`
- **Modality:** text
- **Architecture:** decoder-only transformer (`Qwen2ForCausalLM`), GQA, biases on q/k/v projections
- **Pretraining regime:** self_supervised (next-token prediction). Training data per the official model card.
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen2.5-0.5B. Revision: **unpinned, set `backbone.revision`**.
- **License/usage terms:** see the model card (listed as Apache-2.0; verify when pinning)
- **Tokenizer/processor revision:** same repo and revision as the weights
- **Parent backbone (if fine-tuned):** none (base model)
- **Input/output contract:** `input_ids (B, T) int64` → `logits (B, T, vocab) float`
- **Parameter count / weight size:** record from the loaded model (`sum(p.numel() for p in model.parameters())`); do not copy from the name

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", revision=REV)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", revision=REV, torch_dtype="float32")
```

## Preprocessing and outputs

Raw text goes through the model's tokenizer. Used here for WikiText-2 perplexity only; no chat template.

## Smoke test

`python -m pytest Quantization_and_Pruning/tests/test_cafeq_model.py -k qwen2` uses a tiny *random* Qwen2 config with the same layout. It does not download these weights.

## Caveats

Chosen as the CPU-feasible stand-in for the paper's Qwen2.5-7B (Table 5). Results on it are `approximate` with respect to the paper.
