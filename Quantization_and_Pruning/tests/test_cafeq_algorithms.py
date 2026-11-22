"""Unit tests for the three CafeQ components on small synthetic matrices (CPU, no downloads)."""
import math

import pytest

torch = pytest.importorskip("torch")

from Quantization_and_Pruning.quantization.common import (  # noqa: E402
    QuantSpec,
    block_diag_dense,
    quantize_dequantize,
    random_block_diagonal,
    relative_l2,
)
from Quantization_and_Pruning.quantization.weight_only.cafeq import (  # noqa: E402
    CafeQLinear,
    adaptive_rounding,
    adaptive_rounding_vo,
    effective_weight,
    learn_block_transform,
    learn_paired_transform,
    surrogate_loss,
    transform_weight,
    vo_paired_error,
)
from Quantization_and_Pruning.quantization.weight_only.cafeq.config import (  # noqa: E402
    BlockTransformConfig,
    PairedTransformConfig,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _outlier_weight(out_f=48, in_f=32, seed=0):
    """LLM-like weight: small Gaussian entries plus a few large-magnitude input channels."""
    g = _gen(seed)
    w = torch.randn(out_f, in_f, generator=g) * 0.02
    idx = torch.randperm(in_f, generator=g)[:3]
    w[:, idx] *= 15
    return w


# ------------------------------------------------------------------ Section 3.1 (single matrix)

def test_transform_and_effective_weight_are_consistent():
    w = torch.randn(12, 16, generator=_gen())
    blocks = random_block_diagonal(16, 4, "rotation", _gen(1)) + 0.2 * torch.randn(4, 4, 4, generator=_gen(2))
    m = block_diag_dense(blocks)
    wt = transform_weight(w, blocks)
    assert torch.allclose(wt, w @ m.t(), atol=1e-5)  # W_lin M^T  ==  (M W_paper)^T
    inv = torch.linalg.inv(blocks)
    assert torch.allclose(effective_weight(wt, inv), w, atol=1e-4)  # no quantization -> exact


def test_grouped_surrogate_upper_bounds_stochastic_error():
    """E||M^-1 Q_stoch(MW) - W||^2 <= grouped bound (the derivation behind Eq. 4)."""
    w = torch.randn(24, 32, generator=_gen(3))
    w[0, 3] = 12.0
    blocks = random_block_diagonal(32, 8, "rotation", _gen(4)) + 0.3 * torch.randn(4, 8, 8, generator=_gen(5))
    inv = torch.linalg.inv(blocks)
    spec = QuantSpec(3, 8)
    wt = transform_weight(w, blocks)
    g = _gen(6)
    errs = [
        float(((effective_weight(quantize_dequantize(wt, spec, stochastic=True, generator=g), inv) - w) ** 2).sum())
        for _ in range(1500)
    ]
    mc = sum(errs) / len(errs) / float((w ** 2).sum())
    bound = float(surrogate_loss(w, blocks, spec, "grouped_bound"))
    assert mc <= bound * 1.02


@pytest.mark.parametrize("loss", ["paper_eq4", "grouped_bound"])
def test_learned_block_transform_beats_rtn(loss):
    w = _outlier_weight()
    spec = QuantSpec(4)
    cfg = BlockTransformConfig(block_size=8, iterations=300, lr=1e-2, loss=loss)
    res = learn_block_transform(w, spec, cfg, generator=_gen(7))
    info = res.info
    assert res.blocks.shape == (4, 8, 8) and res.weight_q.shape == w.shape
    assert info["surrogate_final"] <= info["surrogate_init"] + 1e-12   # keep_best
    assert info["surrogate_final"] < 0.9 * info["surrogate_init"]      # the optimiser actually moved
    assert info["rel_l2_final"] <= info["rel_l2_init"] + 1e-9          # fallback_if_worse
    assert info["rel_l2_final"] < info["rel_l2_rtn"]
    assert math.isclose(relative_l2(effective_weight(res.weight_q, res.inv_blocks), w), info["rel_l2_final"], rel_tol=1e-4)


def test_cafeq_linear_forward_matches_effective_weight():
    w = _outlier_weight(16, 32, seed=1)
    bias = torch.randn(16, generator=_gen(8))
    res = learn_block_transform(w, QuantSpec(4), BlockTransformConfig(block_size=8, iterations=20), generator=_gen(9))
    mod = CafeQLinear.build(res.weight_q, bias, res.inv_blocks)
    x = torch.randn(3, 7, 32, generator=_gen(10))
    expected = x @ effective_weight(res.weight_q, res.inv_blocks).t() + bias
    assert mod(x).shape == (3, 7, 16)
    assert torch.allclose(mod(x), expected, atol=1e-4)


# ------------------------------------------------------------------ Appendix A (paired transform)

def test_paired_transform_preserves_product_and_lowers_pseudo_loss():
    g = _gen(11)
    w1 = torch.randn(32, 8, generator=g) * 0.05
    w1[:, 2] *= 10  # outlier direction in the shared dimension
    w2 = torch.randn(8, 40, generator=g) * 0.05
    cfg = PairedTransformConfig(iterations=300, lr=1e-2)
    res = learn_paired_transform(w1, w2, cfg)
    assert res.m.shape == (8, 8)
    assert torch.allclose((w1 @ res.m) @ (res.m_inv @ w2), w1 @ w2, atol=1e-4)
    assert res.info["pseudo_final"] <= res.info["pseudo_init"] + 1e-9
    assert res.info["pseudo_final"] < res.info["pseudo_init"]


@pytest.mark.parametrize("pseudo_loss", ["logsumexp", "sum_sq", "sum_sq_weighted"])
def test_paired_pseudo_losses_run(pseudo_loss):
    w1, w2 = torch.randn(16, 4, generator=_gen(12)), torch.randn(4, 16, generator=_gen(13))
    cfg = PairedTransformConfig(iterations=20, pseudo_loss=pseudo_loss, orth_weight=0.1, init="rotation")
    res = learn_paired_transform(w1, w2, cfg, generator=_gen(14))
    assert math.isfinite(res.info["pseudo_final"])


# ------------------------------------------------------------------ Appendix B (adaptive rounding)

def test_exports_are_not_shadowed_by_submodules():
    import Quantization_and_Pruning.quantization.weight_only.cafeq as cafeq
    import Quantization_and_Pruning.quantization.weight_only.cafeq.apply  # noqa: F401  loads every submodule

    for name in cafeq.__all__:
        obj = getattr(cafeq, name)
        assert not isinstance(obj, type(cafeq)), f"cafeq.{name} resolved to a module"


def test_paper_toy_example():
    """Appendix B: 1-bit grid {0, 1}; PQE drops from 0.64 to 0.36."""
    w = torch.diag(torch.tensor([1.0, 0.6]))

    def q01(x):
        return torch.clamp(torch.round(x), 0, 1)

    _, _, info = adaptive_rounding(w, w, q01, q01, iterations=1)
    p_norm = float(torch.linalg.norm(w @ w))
    assert math.isclose(info["rel_pqe_independent"] * p_norm, 0.64, abs_tol=1e-6)
    assert math.isclose(info["rel_pqe_final"] * p_norm, 0.36, abs_tol=1e-6)


def _brute_force_vo_error(wv_h, wo_h, wv, wo, n_heads, n_kv, dh):
    r = n_heads // n_kv
    num = den = 0.0
    for h in range(n_heads):
        g = h // r
        vs, hs = slice(g * dh, (g + 1) * dh), slice(h * dh, (h + 1) * dh)
        ref = wv[vs].double().t() @ wo[:, hs].double().t()
        approx = wv_h[vs].double().t() @ wo_h[:, hs].double().t()
        num += float(((approx - ref) ** 2).sum())
        den += float((ref ** 2).sum())
    return math.sqrt(num / den)


def test_vo_error_and_adaptive_rounding_with_gqa():
    n_heads, n_kv, dh, d_model = 4, 2, 8, 32
    g = _gen(15)
    wv = torch.randn(n_kv * dh, d_model, generator=g)
    wo = torch.randn(d_model, n_heads * dh, generator=g)
    spec = QuantSpec(3)
    qv, qo = quantize_dequantize(wv, spec), quantize_dequantize(wo, spec)
    fast = vo_paired_error(qv, qo, wv, wo, n_heads, n_kv, dh)
    assert math.isclose(fast, _brute_force_vo_error(qv, qo, wv, wo, n_heads, n_kv, dh), rel_tol=1e-6)

    v_hat, o_hat, info = adaptive_rounding_vo(wv, wo, n_heads, n_kv, dh, spec, spec, iterations=4)
    assert v_hat.shape == wv.shape and o_hat.shape == wo.shape and v_hat.dtype == wv.dtype
    assert math.isclose(info["rel_pqe_independent"], fast, rel_tol=1e-6)
    assert info["rel_pqe_final"] <= info["rel_pqe_independent"] + 1e-12
    assert math.isclose(vo_paired_error(v_hat, o_hat, wv, wo, n_heads, n_kv, dh), info["rel_pqe_final"], rel_tol=1e-5)
    # outputs stay on the base rounding grid: re-quantizing is a no-op
    assert torch.allclose(quantize_dequantize(v_hat, spec), v_hat, atol=1e-5)
    assert torch.allclose(quantize_dequantize(o_hat, spec), o_hat, atol=1e-5)
