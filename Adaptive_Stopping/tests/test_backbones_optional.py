"""Backbone wrapper tests that need optional dependencies.

* HF: builds a tiny randomly initialized Llama and a word-level tokenizer
  in memory, so no download is needed. Skipped without torch/transformers.
* vLLM: set ``SCPS_VLLM_MODEL`` (e.g. ``Qwen/Qwen3.5-9B``) on a GPU machine.
"""

import math
import os

import pytest

from Adaptive_Stopping.common.budget import Budget
from Adaptive_Stopping.methods.budget_policy.scps.controller import SCPSController

LABELS = list("ABCDEFGHIJ")
ITEM = {"question_id": 0, "question": "Which letter?", "options": [f"opt {c}" for c in LABELS], "answer": "C"}


def tiny_hf_backbone():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    words = ("The answer is ( ) . Which letter ? Options : Question Reply with only the of correct option "
             "following a multiple choice question opt Before answering carefully check whether correct "
             "\n A B C D E F G H I J").split()
    for w in words + LABELS:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    hf_tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                                   pad_token_id=0, eos_token_id=1, bos_token_id=None)
    model = transformers.LlamaForCausalLM(cfg)
    from Backbones.Generative_Pretrained.hf_causal_lm import HFCausalLM

    return HFCausalLM("tiny-random-llama", model=model, tokenizer=hf_tok, use_chat_template=False)


def test_hf_generate_and_score():
    bb = tiny_hf_backbone()
    gens = bb.generate(bb.render_prompt("Which letter ?"), n=3, max_tokens=7, temperature=0.6, seed=1)
    assert len(gens) == 3
    assert all(1 <= g.num_tokens <= 7 for g in gens)
    assert all(g.finished or g.num_tokens == 7 for g in gens)
    again = bb.generate(bb.render_prompt("Which letter ?"), n=3, max_tokens=7, temperature=0.6, seed=1)
    assert [g.token_ids for g in gens] == [g.token_ids for g in again]  # seeded
    sc = bb.score_next_options(bb.render_prompt("Which letter ?", "The answer is ("), LABELS)
    assert set(sc.logprobs) == set(LABELS)
    assert sum(math.exp(v) for v in sc.logprobs.values()) == pytest.approx(1.0, abs=1e-6)
    meta = bb.get_model_metadata()
    assert meta["num_parameters"] > 0 and meta["backend"] == "transformers"


def test_scps_runs_on_hf_backbone():
    bb = tiny_hf_backbone()
    budget = Budget(4, 6, 24, 200)
    res = SCPSController(bb, budget).run(ITEM)
    led = res.ledger
    assert led["rollouts"] <= 4 and led["decode_tokens"] <= 24 and led["total_tokens"] <= 200
    assert res.metadata["violations"] == []


@pytest.mark.skipif(not os.environ.get("SCPS_VLLM_MODEL"), reason="set SCPS_VLLM_MODEL on a GPU machine")
def test_vllm_generate_and_score():
    from Backbones.Generative_Pretrained.vllm_causal_lm import VLLMCausalLM

    bb = VLLMCausalLM(os.environ["SCPS_VLLM_MODEL"], max_model_len=4096)
    from Adaptive_Stopping.common.mcq import build_mcq_prompt

    user = build_mcq_prompt(ITEM["question"], ITEM["options"])
    gens = bb.generate(bb.render_prompt(user), n=2, max_tokens=16, temperature=0.6, seed=1)
    assert len(gens) == 2 and all(g.num_tokens <= 16 for g in gens)
    sc = bb.score_next_options(bb.render_prompt(user, "The answer is ("), LABELS)
    assert sum(math.exp(v) for v in sc.logprobs.values()) == pytest.approx(1.0, abs=1e-6)
