"""Sample selection must equal upstream _limit_dataset (random.seed(0) + shuffle + first k)."""
from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

import repro_lib as rl  # noqa: E402
from bastion import benchmark as ub  # noqa: E402


@pytest.mark.parametrize("n,cap", [(1319, 64), (500, 64), (80, 40), (30, 30), (30, 64), (164, 16), (10, None)])
def test_selection_matches_upstream(n, cap):
    data = [{"turns": [f"q{i}"]} for i in range(n)]
    random.seed(0)  # what upstream _run_transformers does before _limit_dataset
    expected = ub._limit_dataset(list(data), cap)
    idx = rl.upstream_selection_indices(n, cap)
    assert [data[i] for i in idx] == expected


def test_selection_does_not_touch_global_random():
    random.seed(123)
    state = random.getstate()
    rl.upstream_selection_indices(1000, 10)
    assert random.getstate() == state


def test_from_pretrained_does_not_consume_python_random(tmp_path):
    """Upstream seeds python random before *loading models*, then shuffles. Our selection
    assumes model loading draws nothing from python's global RNG (checked on tiny models)."""
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dflash.model import DFlashDraftModel

    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp_path / "t")
    dcfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    dcfg.block_size = 4
    dcfg.num_target_layers = 2
    dcfg.dflash_config = {"mask_token_id": 63, "target_layer_ids": [0]}
    DFlashDraftModel(dcfg).save_pretrained(tmp_path / "d")
    random.seed(0)
    before = random.getstate()
    Qwen3ForCausalLM.from_pretrained(tmp_path / "t", attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    DFlashDraftModel.from_pretrained(tmp_path / "d", attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    assert random.getstate() == before


def test_cached_tokenizer_load_does_not_consume_python_random():
    from transformers import AutoTokenizer

    for repo in ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"):
        random.seed(0)
        before = random.getstate()
        try:
            AutoTokenizer.from_pretrained(repo, local_files_only=True)
        except Exception:
            continue
        assert random.getstate() == before
        return
    pytest.skip("no Qwen3 tokenizer in the local HF cache")
