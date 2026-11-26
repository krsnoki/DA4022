# Qwen3-0.6B model card (AgentKV integration)

- **Registry ID / weights source:** Qwen/Qwen3-0.6B
- **Source and license:** https://huggingface.co/Qwen/Qwen3-0.6B, Apache-2.0
- **Model and tokenizer revision:** c1899de289a04d12100db370d81485cdf75e47ca
- **Modality / architecture:** text, Qwen3 causal language model with grouped-query attention
- **Regime:** pretrained and post-trained; exact training datasets not registered here
- **Objective:** autoregressive next-token prediction; see upstream card for post-training details
- **Parent checkpoint / fine-tuning dataset:** not specified in this local wrapper
- **Parameter count:** read from the loaded model rather than inferred from its name
- **Wrapper:** `qwen3.load(device, dtype, model_id, revision)`
- **Processor:** pinned AutoTokenizer; model is loaded with SDPA attention
- **Input/output:** unpadded token IDs at batch one; causal-LM logits and dynamic cache
- **Adaptation:** frozen weights, no fine-tuning; cache policy lives in Attention

The pretrained checkpoint has not been downloaded or evaluated locally. CPU
tests instantiate tiny random Qwen3 configurations, not this checkpoint. Cloud
loading occurs through the experiment runner after its CUDA availability check.
The synthetic experiment explicitly serializes Qwen-style chat markers; it does
not claim equivalence to every upstream tool-use/chat-template convention.
