"""CPU tests for the Hugging Face integration and the backbone wrapper.

They build a tiny, randomly initialised OLMoE (2 layers, 8 experts, top-2), so
they need ``transformers`` but no download and no GPU. They check that the
router replacement leaves the model unchanged when nothing is pruned, prunes
only at decode steps, and respects finished sequences.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from Backbones.Generative_Pretrained.hf_causal_lm import (  # noqa: E402
    BackboneSpec,
    HFCausalLMBackbone,
)
from Mixture_of_Experts.execution_layouts.xshare_hf import XShareRouter, install_xshare  # noqa: E402
from Mixture_of_Experts.routing.xshare import XShareConfig  # noqa: E402

N_EXPERTS, TOP_K = 8, 2


def tiny_model(seed: int = 0):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(seed)
    config = OlmoeConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=64,
        norm_topk_prob=False,
    )
    model = OlmoeForCausalLM(config).eval()
    # Larger router weights give clearly separated expert scores.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.normal_(0.0, 1.0)
    return model


def prefill_and_step(model, input_ids, controller=None, valid_rows=None):
    """Return (prefill logits, decode-step logits) for a batch without padding."""
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        if controller is not None:
            controller.valid_rows = valid_rows
        step = model(input_ids=next_token, past_key_values=out.past_key_values, use_cache=True)
        if controller is not None:
            controller.valid_rows = None
    return out.logits, step.logits


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, 97, (4, 6), generator=generator)


def keep_all() -> XShareConfig:
    return XShareConfig(top_k=TOP_K, warmup_k0=0, budget=N_EXPERTS, gate_norm="softmax_topk")


def aggressive() -> XShareConfig:
    return XShareConfig(top_k=TOP_K, warmup_k0=1, budget=0, gate_norm="softmax_topk")


def test_every_router_is_wrapped_once():
    """Catches a layer being skipped, or a second install double-wrapping routers."""
    model = tiny_model()
    install_xshare(model, None)
    assert all(isinstance(layer.mlp.gate, XShareRouter) for layer in model.model.layers)
    with pytest.raises(RuntimeError):
        install_xshare(model, None)


def test_baseline_wrapper_changes_nothing(inputs):
    """With no config the wrapper must call the original router: identical logits."""
    reference = tiny_model()
    wrapped = copy.deepcopy(reference)
    install_xshare(wrapped, None)
    ref_prefill, ref_step = prefill_and_step(reference, inputs)
    got_prefill, got_step = prefill_and_step(wrapped, inputs)
    assert torch.equal(ref_prefill, got_prefill)
    assert torch.equal(ref_step, got_step)


def test_keep_all_matches_baseline(inputs):
    """XShare's own routing path with every expert kept must equal the model's router."""
    reference = tiny_model()
    wrapped = copy.deepcopy(reference)
    controller = install_xshare(wrapped, keep_all())
    _, ref_step = prefill_and_step(reference, inputs)
    _, got_step = prefill_and_step(wrapped, inputs, controller)
    assert torch.allclose(ref_step, got_step, atol=1e-5, rtol=0)
    assert controller.stats()["mean_kept_experts"] == N_EXPERTS


def test_prefill_is_never_pruned(inputs):
    """Pruning applies at decode steps only, as in the paper."""
    reference = tiny_model()
    wrapped = copy.deepcopy(reference)
    controller = install_xshare(wrapped, aggressive())
    ref_prefill, _ = prefill_and_step(reference, inputs)
    got_prefill, _ = prefill_and_step(wrapped, inputs, controller)
    assert torch.equal(ref_prefill, got_prefill)


def test_decode_step_is_pruned(inputs):
    """An aggressive config must shrink the kept set at decode steps."""
    model = tiny_model()
    controller = install_xshare(model, aggressive())
    prefill_and_step(model, inputs, controller)
    stats = controller.stats()
    assert stats["layer_calls"] == len(model.model.layers)
    # Warm-up keeps each token's top-1: at most 4 experts, at least top_k = 2.
    assert TOP_K <= stats["mean_kept_experts"] <= 4
    assert stats["mean_activated_experts"] <= stats["mean_kept_experts"]


def test_finished_rows_do_not_count(inputs):
    """Rows marked finished must be excluded from the token count and the selection."""
    model = tiny_model()
    controller = install_xshare(model, aggressive())
    valid = torch.tensor([True, True, False, False])
    prefill_and_step(model, inputs, controller, valid)
    stats = controller.stats()
    assert stats["mean_tokens_per_step"] == 2
    assert stats["mean_kept_experts"] <= 2  # top-1 of two tokens, topped up to top_k = 2


def test_config_must_match_router(inputs):
    """A config written for another top-k must fail loudly, not route silently."""
    model = tiny_model()
    install_xshare(model, XShareConfig(top_k=3, warmup_k0=1, budget=0, gate_norm="softmax_topk"))
    with pytest.raises(ValueError):
        prefill_and_step(model, inputs)


def test_backbone_metadata_and_adaptation_state(tmp_path):
    """The wrapper reports the expert layout and saves only configuration, no weights."""
    spec = BackboneSpec("test.tiny-olmoe", "tiny", "none", "none", "float32", "none")
    backbone = HFCausalLMBackbone(spec, tiny_model(), tokenizer=None, load_time_s=0.0)
    meta = backbone.get_model_metadata()
    assert meta["n_experts"] == N_EXPERTS
    assert meta["top_k"] == TOP_K
    assert meta["num_parameters"] == sum(p.numel() for p in backbone.model.parameters())
    assert meta["device"] == "cpu"
    path = backbone.save_adaptation_state(tmp_path / "state.json", {"warmup_k0": 1, "budget": 8})
    saved = json.loads(path.read_text())
    assert saved["adaptation"] == {"warmup_k0": 1, "budget": 8}
    assert saved["backbone"]["id"] == "test.tiny-olmoe"


def test_registry_entry_is_pinned():
    """The registry must pin the exact revision the results were produced with."""
    pytest.importorskip("yaml")
    from Backbones.Generative_Pretrained.hf_causal_lm import load_spec

    spec = load_spec("text.olmoe-1b-7b-0125-instruct")
    assert spec.revision == "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
    assert spec.hf_path == "allenai/OLMoE-1B-7B-0125-Instruct"
    with pytest.raises(ValueError):
        load_spec("text.gpt-oss-20b.mxfp4")
