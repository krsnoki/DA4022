import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CharTokenizer:
    """Tiny tokenizer for CPU tests: specials 0-3, then printable ASCII."""

    specials = ["<pad>", "<think>", "</think>", "<|im_end|>"]

    def __init__(self):
        self.chars = [chr(c) for c in range(32, 127)] + ["\n"]
        self.vocab = {t: i for i, t in enumerate(self.specials)}
        for i, c in enumerate(self.chars):
            self.vocab[c] = len(self.specials) + i
        self.inv = {i: t for t, i in self.vocab.items()}
        self.eos_token_id = self.vocab["<|im_end|>"]
        self.unk_token_id = None
        self.pad_token_id = 0

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab[c] for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            t = self.inv[int(i)]
            if skip_special_tokens and t in self.specials:
                continue
            out.append(t)
        return "".join(out)


@pytest.fixture
def char_tokenizer():
    return CharTokenizer()


@pytest.fixture
def tiny_qwen3():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(CharTokenizer()),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    # zero logits for </think> and <|im_end|> so greedy decoding keeps thinking
    with torch.no_grad():
        for t in ("</think>", "<|im_end|>"):
            model.lm_head.weight[CharTokenizer().vocab[t]].zero_()
    return model
