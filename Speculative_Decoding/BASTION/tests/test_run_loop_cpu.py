"""CPU integration test of the harness loop (records, multi-turn history, resume, fixed budgets)
with tiny random models, a fake tokenizer and upstream's generate functions."""
from __future__ import annotations

import sys
import time

import pytest

torch = pytest.importorskip("torch")

from conftest import COST_GPU, COST_MODEL  # noqa: E402

import repro_lib as rl  # noqa: E402
import run_benchmark as rb  # noqa: E402
from bastion import tree_draft  # noqa: E402
import dflash.model as dfm  # noqa: E402

EOS = 94


class FakeTokenizer:
    eos_token_id = EOS

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    def encode(self, text, return_tensors="pt"):
        return torch.tensor([[ord(ch) % 90 for ch in text[-40:]]])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + int(i) % 26) for i in ids if not (skip_special_tokens and i == EOS))


@pytest.fixture
def cpu_timing(monkeypatch):
    monkeypatch.setattr(tree_draft, "_cuda_time", time.perf_counter)
    monkeypatch.setattr(dfm, "_cuda_time", time.perf_counter)


def _selected():
    return {
        "gsm8k": {"rows": [{"turns": ["what is 2+2"]}, {"turns": ["what is 3+5"]}], "orig_indices": [7, 3],
                  "dataset_size": 10, "cap": 2},
        "mt-bench": {"rows": [{"turns": ["write a poem", "now shorter"]}], "orig_indices": [12],
                     "dataset_size": 80, "cap": 1},
    }


def _run(tmp_path, tiny_models, plan, history=None, name="run"):
    target, draft = tiny_models

    def generate(mode, ids, budget, max_new):
        if mode in ("ar", "dflash"):
            return dfm.dflash_generate(draft, target=target, input_ids=ids, max_new_tokens=max_new, stop_token_ids=[EOS],
                                       temperature=0.0, block_size=1 if mode == "ar" else draft.block_size, return_stats=True)
        kw = {} if budget is None else {"min_tree_size": budget, "max_tree_size": budget}
        return tree_draft.bastion_generate(draft, target=target, input_ids=ids, max_new_tokens=max_new, stop_token_ids=[EOS],
                                           temperature=0.0, block_size=draft.block_size, return_stats=True,
                                           cost_model_name=COST_MODEL, cost_gpu_type=COST_GPU, draft_latency_s=0.004,
                                           non_verify_latency_s=0.002, **kw)

    writer = rl.JsonlWriter(tmp_path / f"{name}.jsonl")
    recorder = rl.TreeSizeRecorder().install()
    try:
        n = rb.run_loop(datasets=["gsm8k", "mt-bench"], selected=_selected(), plan=plan, generate=generate,
                        tokenizer=FakeTokenizer(), device=torch.device("cpu"), writer=writer, recorder=recorder,
                        history_records=history or {}, base_fields={"run_name": name, "model": "tiny", "draft_model": "tiny-d",
                                                                    "max_new_tokens": 12},
                        enable_thinking=False, max_new_tokens=12, eos=EOS, block_size=draft.block_size,
                        mem_reset=lambda: None, mem_peak_mib=lambda: None, attn_impl=lambda: "sdpa")
    finally:
        recorder.uninstall()
        writer.close()
    return n, rl.load_jsonl(tmp_path / f"{name}.jsonl")


PLAN = [("ar", None), ("dflash", None), ("bastion", None), ("bastion", 32)]


def test_records_history_and_resume(tmp_path, tiny_models, calib_env, cpu_timing):
    calib_env()
    n, recs = _run(tmp_path, tiny_models, PLAN)
    assert n == len(recs) == (2 + 2) * 4
    keys = {rl.record_key(r) for r in recs}
    assert ("mt-bench", 0, 1, "bastion", 32, None) in keys and ("gsm8k", 1, 0, "dflash", None, None) in keys
    assert ("gsm8k", 0, 0, "bastion", None, "upstream") in keys  # run_loop default profile_mode
    for r in recs:
        assert r["time_per_output_token"] > 0 and r["num_output_tokens"] == len(r["output_ids"])
        assert sum(r["acceptance_lengths"]) >= r["num_output_tokens"] - 1
        if r["mode"] == "bastion":
            assert len(r["tree_sizes"]) == r["num_steps"] == len(r["acceptance_lengths"])
            if r["fixed_budget"] == 32:
                assert set(r["tree_sizes"]) == {32} and r["tree_size_mismatch_steps"] == 0
    t2 = [r for r in recs if r["dataset"] == "mt-bench" and r["turn"] == 1]
    assert t2 and all(r["history_sources"] == ["ar"] for r in t2)
    assert len({r["input_hash"] for r in t2}) == 1  # all modes saw the same turn-2 prompt
    assert {r["orig_idx"] for r in recs if r["dataset"] == "gsm8k"} == {7, 3}
    # fp32 tiny model: speculative outputs equal AR
    by = {}
    for r in recs:
        by.setdefault((r["dataset"], r["sample_idx"], r["turn"]), []).append(r["output_ids"])
    assert all(all(o == outs[0] for o in outs) for outs in by.values())

    # Full resume: nothing regenerated.
    n2, recs2 = _run(tmp_path, tiny_models, PLAN)
    assert n2 == 0 and len(recs2) == 16

    # Partial resume after losing records (incl. a truncated tail).
    path = tmp_path / "run.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:9]) + "\n" + lines[9][:20])
    n3, recs3 = _run(tmp_path, tiny_models, PLAN)
    assert n3 == 16 - 9 and len(recs3) == 16
    t2b = {r["input_hash"] for r in recs3 if r["dataset"] == "mt-bench" and r["turn"] == 1}
    assert t2b == {t2[0]["input_hash"]}


def test_sweep_without_ar_uses_external_history(tmp_path, tiny_models, calib_env, cpu_timing):
    calib_env()
    _, e1 = _run(tmp_path, tiny_models, [("ar", None)], name="e1")
    history = {(r["dataset"], r["orig_idx"], r["turn"]): r for r in e1}
    _, sw = _run(tmp_path, tiny_models, [("bastion", 64), ("bastion", None)], history=history, name="sweep")
    t2 = [r for r in sw if r["turn"] == 1]
    assert t2 and all(r["history_sources"] == ["external-ar:e1"] for r in t2)
    e1_t2 = next(r for r in e1 if r["turn"] == 1)
    assert all(r["input_hash"] == e1_t2["input_hash"] for r in t2)
    # Without history and without AR, the first mode in the plan supplies the history.
    _, sw2 = _run(tmp_path, tiny_models, [("bastion", 64)], name="sweep2")
    assert all(r["history_sources"] == ["bastion[N=64]"] for r in sw2 if r["turn"] == 1)


def test_resume_refuses_changed_selection(tmp_path, tiny_models, calib_env, cpu_timing, monkeypatch):
    """Resuming with a selection that maps sample_idx to a different problem must fail, not mix records."""
    calib_env()
    _run(tmp_path, tiny_models, [("ar", None)], name="sel")
    changed = _selected()
    changed["gsm8k"] = dict(changed["gsm8k"], rows=changed["gsm8k"]["rows"][::-1], orig_indices=[3, 7])
    monkeypatch.setattr(sys.modules[__name__], "_selected", lambda: changed)
    with pytest.raises(RuntimeError, match="orig_idx"):
        _run(tmp_path, tiny_models, [("ar", None)], name="sel")


def test_two_profile_modes_into_one_jsonl(tmp_path, tiny_models, calib_env, cpu_timing):
    """run_sweep.sh pattern: warm invocation (fixed + adaptive), then an upstream invocation into the same file."""
    calib_env()
    plan = [("bastion", 32), ("bastion", None)]
    n1, _ = _run_pm(tmp_path, tiny_models, plan, "warm")
    n2, recs = _run_pm(tmp_path, tiny_models, plan, "upstream")
    turns = 2 + 2
    assert n1 == 2 * turns and n2 == turns  # fixed-budget records are not regenerated for the second mode
    adaptive = [r for r in recs if r["fixed_budget"] is None]
    assert sorted(r["profile_mode"] for r in adaptive) == ["upstream"] * turns + ["warm"] * turns
    assert all(r["profile_mode"] == "warm" for r in recs if r["fixed_budget"] == 32)
    assert len({rl.record_key(r) for r in recs}) == len(recs) == 3 * turns
    n3, _ = _run_pm(tmp_path, tiny_models, plan, "warm")
    assert n3 == 0


def _run_pm(tmp_path, tiny_models, plan, profile_mode, name="pm"):
    import functools

    orig = rb.run_loop
    rb_run_loop = functools.partial(orig, profile_mode=profile_mode)
    try:
        rb.run_loop = rb_run_loop
        return _run(tmp_path, tiny_models, plan, name=name)
    finally:
        rb.run_loop = orig
