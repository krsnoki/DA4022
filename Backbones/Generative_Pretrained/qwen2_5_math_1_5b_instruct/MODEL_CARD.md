# Backbone model card: Qwen2.5-Math-1.5B-Instruct

- **Registry ID:** `llm.qwen2_5_math_1_5b_instruct.generative`
- **Modality:** `text`
- **Architecture:** `Qwen2ForCausalLM`
- **Pretraining regime:** `mixed` (self-supervised next-token pretraining followed by math instruction post-training)
- **Training objective/data:** Mathematical reasoning via chain-of-thought (CoT) and tool-integrated reasoning (TIR); exact `fine_tuning_dataset` composition is `unknown` (not itemized in full upstream).
- **Weights URI and exact revision:** `https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct`, `revision: aafeb0fc6f22cbf0eaeed126eff8be45b0360a35` (the commit `main` resolved to for the NVIDIA L4 benchmark runs; last upstream change 2024-09-23).
- **License/usage terms:** `apache-2.0`
- **Tokenizer/processor revision:** Bundled `AutoTokenizer` from `Qwen/Qwen2.5-Math-1.5B-Instruct` at the same pinned `revision: aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`.
- **Parent backbone (if fine-tuned):** `Qwen/Qwen2.5-Math-1.5B`
- **Input/output contract:** `preprocess()` accepts a raw `str` or chat message list `list[dict[str, str]]` (context limit `32768` tokens) and returns `{"input_ids": ..., "attention_mask": ...}`. `forward()` returns causal LM outputs (`logits` over vocabulary). `generate()` returns `list[dict[str, Any]]` with keys `text`, `num_tokens`, `token_ids`, and `latency_ms` (max documented generation tokens `8192`; DIPA default `max_new_tokens=512`, `temperature=0.7`, `top_p=0.8`, `repetition_penalty=1.05`).
- **Parameter count / weight size:** `1,540,000,000` parameters (`1.54B`), default `dtype: float16` (~3.08 GB raw FP16 weights).

## Loading

```python
from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct import (
    BackboneConfig,
    Qwen25Math15BInstructBackbone,
)

backbone = Qwen25Math15BInstructBackbone(
    BackboneConfig(dtype="float16", device="auto")
).load()
```

## Preprocessing and outputs

- **Chat template & tokenization (`preprocess`):** When given `list[dict[str, str]]`, the wrapper invokes `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` if available (falling back to joining `msg["content"]`), then runs `tokenizer(text, return_tensors="pt")` to produce `input_ids` and `attention_mask` of shape `[1, prompt_len]` (deriving an all-ones mask if the tokenizer omits `attention_mask`).
- **Causal LM forward (`forward`):** Passes the tokenized input dictionary directly into `model(**inputs)` for logits or gradient-based inspection without modifying base weights.
- **Generation (`generate`):** Runs `model.generate(...)` under `torch.no_grad()` with `do_sample=(temperature > 0)` (forwarding `temperature` and `top_p` only when sampling), `repetition_penalty`, `max_new_tokens`, and `num_return_sequences`. Slices off the leading `prompt_len` tokens per returned sequence, decodes via `tokenizer.decode(..., skip_special_tokens=True)`, and returns a list of records `{"text": str, "num_tokens": int, "token_ids": list[int], "latency_ms": float}`.

## Smoke test

Run the CPU unit and contract test suite (uses injected CPU/mock models and tokenizers without downloading weights or requiring `transformers` locally):

```bash
pytest Backbones/tests/test_qwen2_5_math_wrapper.py
```

Expected result: all tests pass, validating `BackboneConfig` invariants, `Backbones/registry.yaml` parity, chat-template preprocessing, `forward()`, `generate()` prompt-slicing and sampling controls, `save_adaptation_state()`, and clean-subprocess import isolation (`transformers`, `torch`, `Dataloaders`, `Adaptive_Stopping` absent on wrapper import).

## Caveats

- **Target hardware:** Intended for Kaggle Tesla T4 (16 GB VRAM) in `float16` (or `bfloat16` on Ampere+ GPUs). Do not run full-scale `Qwen2.5-Math-1.5B-Instruct` sweeps on CPU.
- **Pinned revision:** `revision` is pinned to `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35` in `Backbones/registry.yaml`, `BackboneConfig`, and `Adaptive_Stopping/configs/dipa.yaml`. Rows in `Performance_Metrics/results/dipa_math500_L4.csv` record `backbone_revision = main`, which resolved to this SHA at run time.
- **Domain specialization & unknown tuning corpus:** `Qwen2.5-Math-1.5B-Instruct` is a math-specialized checkpoint whose exact `fine_tuning_dataset` mixture is `unknown`; do not generalize math benchmark latency/quality to non-mathematical domains.
- **Runtime dependency:** `load()` requires `torch` and `transformers`; `transformers` is not installed in the minimal local CPU test environment.
