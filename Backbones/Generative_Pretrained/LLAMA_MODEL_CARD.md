# Llama causal language model backbones

| Registry ID | Weights and objective | Revision | Tokenizer |
|---|---|---|---|
| `llama.tiny.synthetic` | Seeded random initialization for offline CPU checks | `v1` | Synthetic vocabulary of 128 token IDs |
| `meta-llama/Llama-2-7b-hf` | Meta's base Llama 2 7B, self-supervised next-token prediction | `01c7f73d771dfac7d292323805ebc428287df4f9` | Fast tokenizer from the same pinned snapshot |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | TinyLlama 1.1B with next-token pretraining, supervised chat fine-tuning and DPO | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | Fast tokenizer from the same pinned snapshot |

All entries expose Hugging Face `LlamaForCausalLM` with eager attention under Transformers 4.48.3. The fixture has two layers, hidden width 32, FFN width 64, four query heads, two KV heads, a 128-position context, tied embeddings, and nonuniform RMSNorm gains. The 7B checkpoint uses safetensors and has a 4096-token context. Access requires approval under the Llama 2 Community License. Pass `cache_dir` to store model and tokenizer files outside tracked paths. [Meta model card and access terms](https://huggingface.co/meta-llama/Llama-2-7b-hf)

TinyLlama Chat is ungated and Apache-2.0 licensed. It has 22 layers, a hidden width of 2048, an FFN width of 5632, 32 query heads, four KV heads, and a 2048-token context. The T4 profile loads its safetensors checkpoint in FP16 and scores the raw WikiText-2 stream without a chat template. [TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) · [Pinned architecture](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json) · [T4 benchmark](../../Quantization_and_Pruning/README_qp.md#tinyllama-on-a-t4)

`load_llama(id, device=..., dtype=..., seed=...)` returns a `LlamaBackbone` with the evaluation-mode model, tokenizer and checkpoint metadata. `seed` controls fixture initialization. Each call loads a separate model instance. Inputs are integer token IDs and a boolean attention mask of shape `[B,T]`. Generation uses the model's upstream interface. Experiments pin generation settings in their own configuration.

`backbone.adapter` exposes embeddings, decoder layers, the final norm and the LM head. `norms(layer_idx)` and `projections(layer_idx)` give per-layer access. The `post_rope(layer_idx, fn)` and `gated_mlp_output(layer_idx, fn)` context managers install instance-local transforms for Q and new K `[B,H,T,D]` and the MLP intermediate `[B,T,F]`, respectively. Both restore the original behavior on context exit. [Transformers Llama source](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/llama/modeling_llama.py)
