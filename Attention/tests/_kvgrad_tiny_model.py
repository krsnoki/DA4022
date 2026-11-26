"""
Shared test helper: a tiny, randomly initialised Qwen3 model with its own tokenizer, built offline.

Used by the KVgrad tests (no downloads, CPU only). Not a test module itself.
"""

from pathlib import Path

import torch

WORDS = [
    "the",
    "grass",
    "is",
    "green",
    "sky",
    "blue",
    "sun",
    "yellow",
    "here",
    "we",
    "go",
    "there",
    "and",
    "back",
    "again",
    "secret",
    "passkey",
    "number",
    "is",
]


def make_tiny_model(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<bos>", "<eos>"])
    corpus = [" ".join(WORDS)] * 20 + [str(i) for i in range(100)] + ["Repeat the previous context : ? ."]
    tokenizer.train_from_iterator(corpus, trainer)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"
    )
    hf_tokenizer.save_pretrained(path)

    config = Qwen3Config(
        vocab_size=hf_tokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.eos_token_id,
        tie_word_embeddings=False,
        attention_bias=True,
        rope_theta=1_000_000.0,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    # kvpress evicts a KV pair by replacing its key with a "fake key" that every query ignores, which
    # requires the queries of a head to lie in a common half-space. Trained LLMs satisfy this; random
    # weights do not. A bias on the slowest-rotating RoPE dimension (index head_dim/2 - 1) reproduces it.
    with torch.no_grad():
        for layer in model.model.layers:
            bias = torch.zeros_like(layer.self_attn.q_proj.bias)
            bias[config.head_dim // 2 - 1 :: config.head_dim] = 10.0
            layer.self_attn.q_proj.bias.copy_(bias)
    model.save_pretrained(path)
