# Backbone model card: DeepSeek-R1-Distill-Qwen-1.5B

- **Registry ID:** `text.deepseek-r1-distill-qwen-1.5b` (see [`Backbones/registry.yaml`](../registry.yaml))
- **Modality:** text
- **Architecture:** `Qwen2ForCausalLM`, decoder-only transformer. It has 28 layers, hidden size 1536, MLP size 8960, 12 query heads and 2 KV heads (GQA). The LM head is untied and the vocabulary has 151,936 tokens (from `config.json`).
- **Pretraining regime:** mixed. The parent model was pretrained with next-token prediction and then specialised for maths. This checkpoint was then fine-tuned on reasoning traces.
- **Training objective/data:** supervised fine-tuning (distillation) of Qwen2.5-Math-1.5B on about 800k samples curated with DeepSeek-R1 (model card, section 7). That data is not released.
- **Weights URI and exact revision:** https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- **License/usage terms:** MIT (model card and `LICENSE` at the pinned revision). The card notes that the Qwen-based distills derive from the Qwen-2.5 series, which is originally under Apache 2.0.
- **Tokenizer/processor revision:** same repository and revision (`tokenizer.json`, `tokenizer_config.json`, chat template).
- **Parent backbone (if fine-tuned):** `text.qwen2.5-math-1.5b` (Qwen/Qwen2.5-Math-1.5B @ `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`, Apache-2.0). It is registered for provenance only and is not evaluated here.
- **Input/output contract:**
  - Input: chat messages, rendered with the model's chat template. The rendered prompt ends with `<｜Assistant｜><think>\n`, so generation starts inside the thinking block.
  - Output: generated token IDs. The reasoning comes first, then `</think>`, then the final answer, then EOS `<｜end▁of▁sentence｜>` (id 151643).
- **Parameter count / weight size:**
  - 1,777,088,000 parameters (safetensors header of the pinned revision). `HFCausalLMBackbone.count_parameters()` recounts them at load time, and the count is written into every run manifest.
  - 3,554,214,621 bytes for `model.safetensors` (bf16).

## Loading

```bash
python -c "from Backbones.Task_Finetuned.hf_causal_lm import HFCausalLMBackbone as B; b = B.from_registry('text.deepseek-r1-distill-qwen-1.5b', device='cuda'); print(b.get_model_metadata())"
```

Weights download to the Hugging Face cache on first use (set `HF_HOME` to choose the location). They are never committed.

## Preprocessing and outputs

- **Prompt:** `HFCausalLMBackbone.preprocess(messages)` applies the chat template and tokenises with `add_special_tokens=False`, because the template already adds BOS.
- **Prompts per task:** built in `Dataloaders/` (`math_cot_boxed.v1`: the question plus "Please reason step by step, and put your final answer within \boxed{}."). No system prompt is used, following the model card's usage notes.
- **Generation defaults** (`generation_config.json`): sampling with temperature 0.6 and top-p 0.95, plus the `transformers` default top-k (50).

## Smoke test

```bash
python -m Adaptive_Stopping.experiments.st_es.generate --dataset-id text.gsm8k.train --num-samples 8 --offset 1000 --max-new-tokens 4096 --run-dir /tmp/st_es_smoke
```

This writes 8 traces and a manifest containing the counted parameters and the GPU name. On an RTX 4060 Laptop GPU (8 GB) it takes about 1.5 minutes including model load.

## Caveats

- The paper uses DS-R1-Distill-Llama-8B, DS-R1-Distill-Qwen-14B and QwQ-32B. This 1.5B model is a stand-in that fits an 8 GB GPU, so results are not directly comparable.
- The model card advises against a system prompt and against greedy decoding (risk of endless repetition). Generation uses sampling, and long traces are cut at the experiment's `max_new_tokens`.
- On Windows (WDDM), exceeding dedicated GPU memory spills into shared system memory instead of raising OOM, and throughput then drops several-fold. The batch budgets in the experiment scripts leave headroom for that reason.
