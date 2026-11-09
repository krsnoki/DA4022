"""Shared fixtures: CPU only, synthetic data, no downloads."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "analysis", ROOT / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # tests never touch the GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Parameters for synthetic cost-model tests: an upstream-known (model, GPU) pair.
COST_MODEL = "qwen3-8b"
COST_GPU = "a100"


@pytest.fixture
def calib_env(tmp_path, monkeypatch):
    """Write a calibration cache like upstream's and point CALIBRATION_JSON_PATH at it."""

    def _make(alpha_m=1.2, beta_m=0.004, alpha_c=1.5, beta_c=0.002):
        path = tmp_path / "calibration_test.json"
        path.write_text(json.dumps({
            "fit_parameters": {"alpha_m": alpha_m, "beta_m": beta_m, "alpha_c": alpha_c, "beta_c": beta_c},
            "fit_version": "sigma_sequence_v1", "sigma_scale": 30.0, "attn_impl": "sdpa",
        }))
        monkeypatch.setenv("CALIBRATION_JSON_PATH", str(path))
        return path

    return _make


@pytest.fixture(autouse=True)
def _no_method_env(monkeypatch):
    for name in ("ADAPTIVE_MIN_TREE_SIZE", "ADAPTIVE_BUCKET_STEP"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def tiny_models():
    """Randomly initialised tiny Qwen3 target + DFlash drafter (fp32, CPU)."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dflash.model import DFlashDraftModel

    torch.manual_seed(1234)
    vocab, hidden = 96, 32
    tcfg = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
        tie_word_embeddings=False, attn_implementation="sdpa",
    )
    target = Qwen3ForCausalLM(tcfg).eval()
    dcfg = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
        attn_implementation="sdpa",
    )
    dcfg.block_size = 4
    dcfg.num_target_layers = 3
    dcfg.dflash_config = {"mask_token_id": vocab - 1, "target_layer_ids": [0, 1]}
    draft = DFlashDraftModel(dcfg).eval()
    # Sharpen the target so greedy argmax margins are large (avoids fp32 near-ties).
    with torch.no_grad():
        target.lm_head.weight.mul_(8.0)
    return target, draft
