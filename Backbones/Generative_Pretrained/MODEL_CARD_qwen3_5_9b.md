# Backbone model card: Qwen3.5-9B

- **Registry ID:** `text.qwen3_5_9b.post_trained`
- **Modality:** multimodal (causal language model with a vision encoder); used text-only by SCPS
- **Architecture:** `qwen3_5`, 32 layers, hidden size 4096, padded token embedding 248320 (model card)
- **Pretraining regime:** mixed (pre-training and post-training, per the model card)
- **Training objective/data:** next-token pre-training followed by post-training; data details are in the Qwen3.5 blog, not verified here
- **Weights URI and exact revision:** https://huggingface.co/Qwen/Qwen3.5-9B, revision not pinned yet (latest commit seen on the Hub file list: `c202236`, short hash). Pin the full hash in the experiment config before reporting.
- **License/usage terms:** Apache-2.0 (model card)
- **Tokenizer/processor revision:** same repository and revision as the weights
- **Parent backbone (if fine-tuned):** `Qwen/Qwen3.5-9B-Base`
- **Input/output contract:** text prompt built with the tokenizer chat template (`enable_thinking=False`), output text plus token IDs through `GenerativeBackbone.generate`; next-token label log-probabilities through `score_next_options`
- **Parameter count / weight size:** 9B (model card, rounded); 19.3 GB repository size on the Hub, safetensors in 4 shards

## Loading

```python
from Backbones.Generative_Pretrained.vllm_causal_lm import VLLMCausalLM
bb = VLLMCausalLM("Qwen/Qwen3.5-9B", revision="<full-commit-hash>", dtype="bfloat16", max_model_len=16384)
```

Not run in this repository's CI (no GPU).

## Preprocessing and outputs

Prompts come from `Adaptive_Stopping/common/mcq.py` (answer-only MCQ). `render_prompt` applies the chat template and appends any partial assistant text. `generate` returns text, token IDs, decoded token count and whether the sample ended on a stop token. `score_next_options` returns log-softmax over option labels of the next-token log-probabilities.

## Smoke test

```bash
SCPS_VLLM_MODEL=Qwen/Qwen3.5-9B python -m pytest -q Adaptive_Stopping/tests/test_backbones_optional.py -k vllm
```

Expected: 1 passed (two samples of at most 16 tokens; option probabilities sum to 1).

## Caveats

- The Hub lists the model as `image-text-to-text`. With the Transformers wrapper you may need `model_class: AutoModelForImageTextToText` in `backbone.kwargs`.
- vLLM's `max_logprobs` caps how many next-token log-probabilities are returned; labels outside the top list get a floor value and are counted in `missing_label_events`.
- The SCPS paper does not name the GPU, package versions or serving command, so serving-level differences can change token counts and latency.
