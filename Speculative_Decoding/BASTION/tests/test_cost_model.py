"""Cost model + controller tests (upstream functions, synthetic calibration, CPU)."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import COST_GPU, COST_MODEL  # noqa: E402

from bastion import cost_model, tree_draft  # noqa: E402


@pytest.mark.parametrize("model", ["qwen3-4b", "qwen3-8b", "llama3.1-8b-instruct"])
@pytest.mark.parametrize("ctx", [0, 64, 1024, 8192])
def test_latency_estimate_monotone_and_convex_in_n(calib_env, model, ctx):
    calib_env()
    est = cost_model.build_verified_latency_estimator(context_tokens=ctx, model_name=model, gpu_type="a100")
    n = np.arange(1, 2049)
    t = np.array([est.estimate(int(k)) for k in n])
    assert np.all(np.diff(t) >= -1e-15), "C(N) must be non-decreasing"
    assert np.all(np.diff(t, 2) >= -1e-12), "max of two convex quadratics must be convex"
    assert np.all(t > 0)
    # next_delta is what the controller uses.
    assert est.next_delta(10) == pytest.approx(est.estimate(11) - est.estimate(10))


def test_uncalibrated_equals_calibrated_with_identity_params(calib_env):
    calib_env(alpha_m=1.0, beta_m=0.0, alpha_c=1.0, beta_c=0.0)
    est = cost_model.build_verified_latency_estimator(context_tokens=256, model_name=COST_MODEL, gpu_type=COST_GPU)
    for s in (1, 16, 300, 512):
        mem, comp = cost_model.compute_cost_model_latencies(model_name=COST_MODEL, gpu_type=COST_GPU, s=s, c=256)
        assert est.estimate(s) == pytest.approx(max(mem, comp), rel=1e-9)


def _rho_sequence(tree, logits):
    logp = torch.log_softmax(logits[0], -1).float()  # float32, as upstream ranks nodes
    par, tok, dep = tree.parent_indices.tolist(), tree.token_ids.tolist(), tree.depths.tolist()
    rho = []
    for i in range(len(tok)):
        lp, j = 0.0, i
        while par[j] >= 0:
            lp += float(logp[dep[j] - 1, tok[j]])
            j = par[j]
        rho.append(math.exp(lp))
    return rho


@pytest.mark.parametrize("seed,scale", [(0, 2.0), (1, 4.0), (2, 6.0), (3, 8.0)])
def test_controller_stops_at_global_max_of_estimated_speedup(calib_env, seed, scale):
    """Prop. 2.4 + implementation: with bucket step 1 (min_tree_size=1) the adaptive
    builder returns argmax_N S(N), S(N) = A(N) / (T_draft + T_aux + C(N)), A = sum of rho."""
    calib_env()
    d, nv, ctx = 0.006, 0.002, 512
    logits = torch.randn(1, 15, 500, generator=torch.Generator().manual_seed(seed)) * scale
    n_max = 4096
    adaptive = tree_draft.build_adaptive_best_tree_from_draft_logits(
        root_token_id=torch.tensor(1), draft_logits=logits, context_prefix_length=ctx, draft_latency_s=d,
        non_verify_latency_s=nv, cost_model_name=COST_MODEL, cost_gpu_type=COST_GPU,
        min_tree_size=1, max_tree_size=n_max)
    n_star = adaptive.token_ids.numel()
    full = tree_draft.build_adaptive_best_tree_from_draft_logits(
        root_token_id=torch.tensor(1), draft_logits=logits, context_prefix_length=ctx, draft_latency_s=d,
        non_verify_latency_s=nv, cost_model_name=COST_MODEL, cost_gpu_type=COST_GPU,
        min_tree_size=n_max, max_tree_size=n_max)
    rho = np.array(_rho_sequence(full, logits))
    assert np.all(rho[1:] <= rho[:-1] * (1 + 1e-9))  # concave A (Prop. 2.3)
    A = np.cumsum(rho)
    est = cost_model.build_verified_latency_estimator(context_tokens=ctx, model_name=COST_MODEL, gpu_type=COST_GPU)
    C = np.array([est.estimate(k) for k in range(1, n_max + 1)])
    S = A / (d + nv + C)
    # Unimodality under the proposition's assumptions: after the first decrease S never increases again.
    first_drop = int(np.argmax(np.diff(S) < 0))
    assert np.all(np.diff(S)[first_drop:] <= 1e-15)
    assert n_star < n_max, "test logits should give an interior optimum"
    assert n_star == int(np.argmax(S)) + 1
    assert torch.equal(adaptive.token_ids, full.token_ids[:n_star])


@pytest.mark.parametrize("seed,scale", [(0, 3.0), (4, 6.0), (5, 9.0)])
def test_default_controller_rounds_down_to_bucket_of_32(calib_env, seed, scale):
    """Upstream default (min_tree_size=32, fixed_interval buckets of 32): the returned
    tree has max(32, 32 * floor(N*/32)) nodes, where N* is the stopping point above."""
    calib_env()
    d, nv, ctx = 0.006, 0.002, 512
    logits = torch.randn(1, 15, 500, generator=torch.Generator().manual_seed(seed)) * scale
    kw = dict(root_token_id=torch.tensor(1), draft_logits=logits, context_prefix_length=ctx, draft_latency_s=d,
              non_verify_latency_s=nv, cost_model_name=COST_MODEL, cost_gpu_type=COST_GPU)
    n_star = tree_draft.build_adaptive_best_tree_from_draft_logits(min_tree_size=1, max_tree_size=8192, **kw).token_ids.numel()
    n_def = tree_draft.build_adaptive_best_tree_from_draft_logits(**kw).token_ids.numel()
    assert n_def == max(32, 32 * (n_star // 32))
