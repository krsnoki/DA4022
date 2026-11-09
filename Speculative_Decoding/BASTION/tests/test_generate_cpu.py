"""End-to-end CPU smoke test of upstream generation with tiny random models.

Exercises the tree-size recorder, the fixed-budget path used by the harness
(bastion_generate(min_tree_size=N, max_tree_size=N)), and losslessness at T=0
in fp32 (the real runs use bf16 on GPU, where divergences are possible).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import COST_GPU, COST_MODEL  # noqa: E402

import repro_lib  # noqa: E402
from bastion import tree_draft  # noqa: E402
from dflash.model import dflash_generate  # noqa: E402


def _bastion(target, draft, ids, budget=None, max_new=24):
    kw = dict(cost_model_name=COST_MODEL, cost_gpu_type=COST_GPU, draft_latency_s=0.004, non_verify_latency_s=0.002)
    if budget is not None:
        kw.update(min_tree_size=budget, max_tree_size=budget)
    return tree_draft.bastion_generate(draft, target=target, input_ids=ids, max_new_tokens=max_new,
                                       stop_token_ids=None, temperature=0.0, return_stats=False, **kw)


@pytest.fixture
def prompt():
    return torch.tensor([[3, 17, 42, 8, 11, 5, 23]])


def test_recorder_reports_fixed_budget_sizes(tiny_models, calib_env, prompt):
    calib_env()
    target, draft = tiny_models
    rec = repro_lib.TreeSizeRecorder().install()
    try:
        assert getattr(tree_draft.build_adaptive_best_tree_from_draft_logits, "_repro_wrapped", False)
        for budget in (32, 64, 200):
            with rec.recording():
                _bastion(target, draft, prompt, budget)
            assert rec.sizes, "builder was not called through the wrapper"
            assert all(s == budget for s in rec.sizes), rec.sizes
            assert all(m == budget for m in rec.requested_min)
            assert all(c >= prompt.shape[1] for c in rec.contexts)
        with rec.recording():
            _bastion(target, draft, prompt, None)
        # Adaptive: >= min (32) and a multiple of the fixed-interval bucket (32) unless == max.
        assert rec.sizes and all(s >= 32 and (s % 32 == 0 or s == 8192) for s in rec.sizes), rec.sizes
        # Recorder inactive -> nothing appended.
        n = len(rec.sizes)
        _bastion(target, draft, prompt, 32)
        assert len(rec.sizes) == n
    finally:
        rec.uninstall()
    assert not getattr(tree_draft.build_adaptive_best_tree_from_draft_logits, "_repro_wrapped", False)


def test_wrapper_does_not_change_outputs(tiny_models, calib_env, prompt):
    calib_env()
    target, draft = tiny_models
    ref = _bastion(target, draft, prompt, None)
    rec = repro_lib.TreeSizeRecorder().install()
    try:
        with rec.recording():
            out = _bastion(target, draft, prompt, None)
    finally:
        rec.uninstall()
    assert torch.equal(ref, out)


@pytest.mark.parametrize("budget", [None, 32, 128])
def test_lossless_fp32(tiny_models, calib_env, prompt, budget):
    calib_env()
    target, draft = tiny_models
    ar = dflash_generate(draft, target=target, input_ids=prompt, max_new_tokens=24, stop_token_ids=None,
                         temperature=0.0, block_size=1, return_stats=False)
    dfl = dflash_generate(draft, target=target, input_ids=prompt, max_new_tokens=24, stop_token_ids=None,
                          temperature=0.0, block_size=draft.block_size, return_stats=False)
    bas = _bastion(target, draft, prompt, budget)
    assert ar.shape == dfl.shape == bas.shape
    assert torch.equal(ar, dfl)
    assert torch.equal(ar, bas)
