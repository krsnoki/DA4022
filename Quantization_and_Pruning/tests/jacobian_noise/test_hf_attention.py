"""Integration with Hugging Face models (tiny random Qwen2 and SigLIP, CPU)."""

import os

import pytest
import torch

from Backbones.registry import load_backbone
from Quantization_and_Pruning.quantization.qat.jacobian_noise import (
    ATTN_IMPL_NAME, JacobianNoiseConfig, attach_jacobian_noise, detach_jacobian_noise,
    jacobian_noise_attention_forward,
)


def tiny_qwen(seed=0):
    torch.manual_seed(seed)
    m, _, _ = load_backbone("test.tiny_qwen2.random")
    return m


def tiny_siglip(seed=0):
    torch.manual_seed(seed)
    m, _, _ = load_backbone("test.tiny_siglip.random")
    return m


def lm_batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (2, 12), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 9:] = 0  # right padding
    return ids, mask


def test_eval_matches_eager_qwen():
    m = tiny_qwen().eval()
    ids, mask = lm_batch()
    ref = m(input_ids=ids, attention_mask=mask).logits
    ctrl = attach_jacobian_noise(m, JacobianNoiseConfig(mode="jacobian"))
    assert m.config._attn_implementation == ATTN_IMPL_NAME
    assert ctrl.num_layers == 2
    out = m(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(out, ref)


def test_train_mode_without_noise_matches_eager():
    m = tiny_qwen().train()
    ids, mask = lm_batch()
    ref = m(input_ids=ids, attention_mask=mask).logits
    attach_jacobian_noise(m, JacobianNoiseConfig(mode="none"))
    assert torch.allclose(m(input_ids=ids, attention_mask=mask).logits, ref, atol=1e-6)


def test_noise_only_in_training():
    m = tiny_qwen()
    ids, mask = lm_batch()
    attach_jacobian_noise(m, JacobianNoiseConfig(mode="constant", constant_sigma=1.0, warmup_steps=0))
    m.eval()
    e1, e2 = m(input_ids=ids).logits, m(input_ids=ids).logits
    assert torch.equal(e1, e2)
    m.train()
    t1, t2 = m(input_ids=ids).logits, m(input_ids=ids).logits
    assert not torch.allclose(t1, t2)


def test_noise_is_scaled_per_row_by_sigma():
    """With zero Q, scores are 0, so softmax(Z~) = softmax(sigma_i * eps) for a known eps."""

    class Dummy(torch.nn.Module):
        num_key_value_groups = 1

    mod = Dummy().train()
    ctrl = attach_ctrl = None
    from Quantization_and_Pruning.quantization.qat.jacobian_noise import JacobianNoiseController

    ctrl = JacobianNoiseController(JacobianNoiseConfig(mode="constant", constant_sigma=0.7, warmup_steps=0))
    mod._jn_controller, mod._jn_layer_idx = ctrl, 0
    eps = torch.randn(1, 2, 3, 4)
    ctrl.sample_noise = lambda like: eps
    q = torch.zeros(1, 2, 3, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.eye(4).expand(1, 2, 4, 4)
    out, probs = jacobian_noise_attention_forward(mod, q, k, v, None, scaling=1.0)
    assert torch.allclose(probs, torch.softmax(0.7 * eps, -1), atol=1e-6)


def test_padding_mask_respected_with_noise():
    m = tiny_qwen().train()
    ids, mask = lm_batch()
    attach_jacobian_noise(m, JacobianNoiseConfig(mode="constant", constant_sigma=5.0, warmup_steps=0))
    captured = []
    attn = m.model.layers[0].self_attn
    h = attn.register_forward_hook(lambda mod, i, o: captured.append(o[1]))
    m(input_ids=ids, attention_mask=mask)
    h.remove()
    probs = captured[0]
    assert torch.isfinite(probs).all()
    assert torch.all(probs[1, :, :, 9:][:, :9] == 0)  # real queries never attend to padding
    assert torch.all(torch.triu(probs[0, 0], diagonal=1) == 0)  # causal


def test_statistics_collected_and_grad_flows():
    m = tiny_qwen().train()
    ids, mask = lm_batch()
    ctrl = attach_jacobian_noise(m, JacobianNoiseConfig(mode="jacobian", warmup_steps=1, update_interval=1))
    ctrl.set_query_mask(mask)
    assert ctrl.collecting
    loss = m(input_ids=ids, attention_mask=mask, labels=ids).loss
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    ctrl.step()
    assert ctrl.history and len(ctrl.history[-1]["sigma_per_layer"]) == 2


def test_gradient_checkpointing_replays_same_noise():
    ids, mask = lm_batch()
    grads = []
    for ckpt in (False, True):
        m = tiny_qwen(seed=3).train()
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        attach_jacobian_noise(m, JacobianNoiseConfig(mode="constant", constant_sigma=0.5, warmup_steps=0))
        torch.manual_seed(123)
        m(input_ids=ids, attention_mask=mask, labels=ids, use_cache=False).loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_detach_restores_eager_and_saves_clean_config(tmp_path):
    m = tiny_qwen()
    attach_jacobian_noise(m, JacobianNoiseConfig())
    detach_jacobian_noise(m)
    assert m.config._attn_implementation == "eager"
    assert not any(hasattr(mod, "_jn_controller") for mod in m.modules())
    m.save_pretrained(tmp_path)
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(tmp_path)  # loads without our registration


def test_siglip_vision_only_noise_and_eval_equivalence():
    m = tiny_siglip().eval()
    x = torch.randn(2, 3, 32, 32)
    ref = m.get_image_features(pixel_values=x)
    ctrl = attach_jacobian_noise(m, JacobianNoiseConfig(mode="jacobian"), include="vision_model")
    assert ctrl.num_layers == 2 and all("vision_model" in n for n in ctrl.layer_names)
    assert torch.allclose(m.get_image_features(pixel_values=x), ref, atol=1e-6)
    m.train()
    a, b = m.get_image_features(pixel_values=x), m.get_image_features(pixel_values=x)
    assert not torch.allclose(a, b)
    ids = torch.randint(0, 256, (2, 16))
    t1, t2 = m.get_text_features(input_ids=ids), m.get_text_features(input_ids=ids)
    assert torch.allclose(t1, t2)  # text tower has no noise


def test_no_attention_modules_raises():
    with pytest.raises(ValueError):
        attach_jacobian_noise(torch.nn.Linear(2, 2))
