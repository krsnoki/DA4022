"""Noise schedule: warm-up, update timing, sigma formula, ablation modes, masking, state."""

import math

import pytest
import torch

from Quantization_and_Pruning.quantization.qat.jacobian_noise import (
    JacobianNoiseConfig, JacobianNoiseController, softmax_jacobian_fro_sq,
)


def probs(B=2, H=3, Q=5, K=5, seed=0, sharp=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(B, H, Q, K, generator=g) * sharp, -1)


def make(mode="jacobian", **kw):
    base = dict(mode=mode, warmup_steps=2, update_interval=3)
    base.update(kw)
    return JacobianNoiseController(JacobianNoiseConfig(**base))


def run_until_update(ctrl, p, layer=0):
    """Feed `p` on every collecting step until the first sigma update happened."""
    for _ in range(50):
        if ctrl.collecting:
            ctrl.accumulate(layer, p)
        before = len(ctrl.history)
        ctrl.step()
        if len(ctrl.history) > before:
            return
    raise AssertionError("no update happened")


def test_config_validation():
    with pytest.raises(ValueError):
        JacobianNoiseConfig(mode="bogus")
    with pytest.raises(ValueError):
        JacobianNoiseConfig(sigma_min=1.0, sigma_max=0.5)
    with pytest.raises(ValueError):
        JacobianNoiseConfig.from_dict({"alpha": 0.5, "typo": 1})
    assert JacobianNoiseConfig.from_dict(None).alpha == 0.5


def test_paper_defaults():
    c = JacobianNoiseConfig()
    assert (c.alpha, c.sigma_min, c.sigma_max, c.warmup_steps, c.warmup_sigma, c.update_interval) == (
        0.5, 0.005, 10.0, 100, 0.1, 100)


def test_warmup_sigma_and_update_timing():
    ctrl = make(warmup_steps=2, update_interval=3)
    dev = torch.device("cpu")
    assert torch.allclose(ctrl.sigma_for(0, 4, dev), torch.full((4,), 0.1))
    # last_step collection: collect on the optimiser step right before each boundary
    pattern = []
    for _ in range(9):
        pattern.append(ctrl.collecting)
        if ctrl.collecting:
            ctrl.accumulate(0, probs())
        ctrl.step()
    # boundaries after steps 2, 5, 8 -> collect at step indices 1, 4, 7
    assert pattern == [False, True, False, False, True, False, False, True, False]
    assert [h["step"] for h in ctrl.history] == [2, 5, 8]


def test_sigma_formula_matches_paper():
    ctrl = make(alpha=0.5)
    p = probs(Q=6, K=6)
    run_until_update(ctrl, p)
    expected = 0.5 * softmax_jacobian_fro_sq(p).mean(dim=(0, 1)).sqrt()
    got = ctrl.sigma_for(0, 6, torch.device("cpu"))
    assert torch.allclose(got, expected.clamp(0.005, 10.0), atol=1e-6)


def test_sigma_clamped_for_one_hot_rows():
    ctrl = make(alpha=0.5)
    p = torch.zeros(1, 1, 3, 3)
    p[..., 0] = 1.0
    run_until_update(ctrl, p)
    assert torch.allclose(ctrl.sigma_for(0, 3, torch.device("cpu")), torch.full((3,), 0.005))


def test_window_collection_averages_interval():
    ctrl = make(collect="window", warmup_steps=0, update_interval=2)
    a, b = probs(seed=1), probs(seed=2, sharp=5.0)
    for p in (a, b):
        assert ctrl.collecting
        ctrl.accumulate(0, p)
        ctrl.step()
    mean = (softmax_jacobian_fro_sq(a).mean((0, 1)) + softmax_jacobian_fro_sq(b).mean((0, 1))) / 2
    assert torch.allclose(ctrl.sigma_for(0, 5, torch.device("cpu")), (0.5 * mean.sqrt()).clamp(0.005, 10), atol=1e-6)


def test_query_mask_excludes_padding_rows():
    ctrl = make()
    p = probs(B=2, Q=4, K=4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    ctrl.set_query_mask(mask)
    run_until_update(ctrl, p)
    jsq = softmax_jacobian_fro_sq(p).mean(1)  # [B, Q], mean over heads
    exp = torch.stack([jsq[:, 0].mean(), jsq[:, 1].mean(), jsq[0, 2], jsq[0, 3]])
    assert torch.allclose(ctrl.sigma_for(0, 4, torch.device("cpu")), (0.5 * exp.sqrt()).clamp(0.005, 10), atol=1e-6)


def test_unseen_positions_fall_back_to_layer_mean():
    ctrl = make()
    run_until_update(ctrl, probs(Q=3, K=3))
    s = ctrl.sigma_for(0, 6, torch.device("cpu"))
    assert torch.allclose(s[3:], s[:3].mean().expand(3), atol=1e-6)


def test_jacobian_layer_mode_is_constant_per_layer():
    ctrl = make("jacobian_layer")
    p = probs(Q=6, K=6)
    run_until_update(ctrl, p)
    s = ctrl.sigma_for(0, 6, torch.device("cpu"))
    assert torch.allclose(s, s[0].expand(6))
    assert s[0].item() == pytest.approx(0.5 * math.sqrt(float(softmax_jacobian_fro_sq(p).mean())), rel=1e-5)


def test_uniform_matched_has_same_rms_as_jacobian():
    p0, p1 = probs(seed=3, sharp=0.5), probs(seed=4, sharp=4.0)
    jac, uni = make("jacobian"), make("uniform_matched")
    for c in (jac, uni):
        for _ in range(50):
            if c.collecting:
                c.accumulate(0, p0)
                c.accumulate(1, p1)
            n = len(c.history)
            c.step()
            if len(c.history) > n:
                break
    dev = torch.device("cpu")
    js = torch.cat([jac.sigma_for(0, 5, dev), jac.sigma_for(1, 5, dev)])
    us = torch.cat([uni.sigma_for(0, 5, dev), uni.sigma_for(1, 5, dev)])
    assert torch.allclose(us, us[0].expand_as(us))
    assert us.pow(2).mean().item() == pytest.approx(js.pow(2).mean().item(), rel=1e-5)
    assert js.std() > 0  # the Jacobian schedule really is non-uniform


def test_none_and_constant_modes():
    dev = torch.device("cpu")
    assert make("none").sigma_for(0, 4, dev) is None
    assert not make("none").collecting
    c = make("constant", constant_sigma=0.3, warmup_steps=0)
    assert torch.allclose(c.sigma_for(0, 4, dev), torch.full((4,), 0.3))
    assert not c.collecting


def test_disabled_context_and_measure():
    ctrl = make()
    with ctrl.disabled():
        assert ctrl.sigma_for(0, 3, torch.device("cpu")) is None
    assert ctrl.sigma_for(0, 3, torch.device("cpu")) is not None
    with ctrl.measure():
        assert ctrl.collecting and not ctrl.injects_noise
        ctrl.accumulate(0, probs())
        m = ctrl.mean_jacobian_sq()[0][:5]
    assert torch.allclose(m, softmax_jacobian_fro_sq(probs()).mean((0, 1)), atol=1e-6)


def test_state_dict_roundtrip():
    ctrl = make()
    run_until_update(ctrl, probs())
    state = ctrl.state_dict()
    new = JacobianNoiseController()
    new.load_state_dict(state)
    dev = torch.device("cpu")
    assert new.global_step == ctrl.global_step
    assert torch.allclose(new.sigma_for(0, 5, dev), ctrl.sigma_for(0, 5, dev))


def test_too_long_sequence_raises():
    ctrl = make(max_positions=4)
    with pytest.raises(ValueError):
        ctrl.sigma_for(0, 5, torch.device("cpu"))
