"""--profile-mode: warm vs upstream controller constants (profile function mocked, CPU only)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import run_benchmark as rb  # noqa: E402


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[0]["content"]

    def encode(self, text, return_tensors="pt"):
        return torch.tensor([[1, 2, 3]])


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("CALIBRATION_JSON_PATH", "placeholder")  # restored after the test
    monkeypatch.delenv("BASTION_SIGMA_SCALE", raising=False)
    (tmp_path / "calibration_b4500_qwen3-4b.json").write_text(json.dumps({
        "fit_parameters": {"alpha_m": 1.0, "beta_m": 0.0, "alpha_c": 1.0, "beta_c": 0.0},
        "fit_version": "sigma_sequence_v1", "sigma_scale": 30.0, "attn_impl": "sdpa"}))
    return tmp_path


@pytest.fixture
def fake_profile(monkeypatch):
    """Cold first call, warm afterwards (values from the GPU pilot on Qwen3-4B)."""
    calls = []
    values = [(0.0103, 0.0205), (0.00322, 0.00084), (0.00320, 0.00085)]

    def _prof(**kw):
        calls.append(kw)
        return values[min(len(calls) - 1, len(values) - 1)]

    monkeypatch.setattr(rb.ub, "profile_tree_adaptive_constants", _prof)
    return calls


def _ensure(mode):
    return rb.ensure_calibration_and_profile(
        args=SimpleNamespace(temperature=0.0, model="Qwen/Qwen3-4B"),
        target=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")),
        draft_model=object(), tokenizer=_Tok(), device=torch.device("cpu"), block_size=16,
        model_key="qwen3-4b", gpu_key="b4500", probe_turn="hello", bastion_attn_impl="sdpa", profile_mode=mode)


def test_warm_discards_first_call_and_leaves_upstream_file_untouched(cache, fake_profile):
    upstream = cache / "calibration_b4500_qwen3-4b.profile.json"
    upstream.write_text(json.dumps({"draft_latency_s": 0.0151, "non_verify_latency_s": 0.1513, "attn_impl": "sdpa"}))
    before = upstream.read_bytes()
    res = _ensure("warm")
    assert len(fake_profile) == 2
    # identical arguments for the discarded and kept call
    a, b = fake_profile
    assert {k: v for k, v in a.items() if k != "input_ids"} == {k: v for k, v in b.items() if k != "input_ids"}
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert fake_profile[1]["num_trials"] == 3 and fake_profile[1]["block_size"] == 16
    assert (res["draft_latency_s"], res["non_verify_latency_s"]) == (0.00322, 0.00084)
    assert res["profile_mode"] == "warm"
    assert res["profile_discarded_warmup"] == {"draft_latency_s": 0.0103, "non_verify_latency_s": 0.0205}
    assert res["upstream_profile_cache_contents"]["non_verify_latency_s"] == 0.1513  # recorded, not used
    warm = json.loads((cache / "calibration_b4500_qwen3-4b.profile.warm.json").read_text())
    assert warm["profile_mode"] == "warm" and warm["draft_latency_s"] == 0.00322
    assert upstream.read_bytes() == before
    # Cached: a second process does not re-profile.
    res2 = _ensure("warm")
    assert len(fake_profile) == 2 and res2["draft_latency_s"] == 0.00322


def test_warm_does_not_create_upstream_profile(cache, fake_profile):
    _ensure("warm")
    assert not (cache / "calibration_b4500_qwen3-4b.profile.json").exists()


def test_upstream_mode_uses_upstream_cache(cache, fake_profile):
    upstream = cache / "calibration_b4500_qwen3-4b.profile.json"
    upstream.write_text(json.dumps({"draft_latency_s": 0.0151, "non_verify_latency_s": 0.1513, "attn_impl": "sdpa"}))
    res = _ensure("upstream")
    assert fake_profile == [] and res["non_verify_latency_s"] == 0.1513 and res["profile_mode"] == "upstream"
    assert res["profile_discarded_warmup"] is None
    assert not (cache / "calibration_b4500_qwen3-4b.profile.warm.json").exists()


def test_upstream_mode_creates_cache_with_single_cold_call(cache, fake_profile):
    res = _ensure("upstream")
    assert len(fake_profile) == 1 and res["draft_latency_s"] == 0.0103
    assert json.loads((cache / "calibration_b4500_qwen3-4b.profile.json").read_text())["non_verify_latency_s"] == 0.0205


def test_invalid_warm_cache_refused(cache, fake_profile):
    (cache / "calibration_b4500_qwen3-4b.profile.warm.json").write_text(json.dumps(
        {"draft_latency_s": 0.003, "non_verify_latency_s": 0.001, "attn_impl": "flash_attention_2", "profile_mode": "warm"}))
    with pytest.raises(ValueError, match="attn_impl"):
        _ensure("warm")
    with pytest.raises(ValueError):
        _ensure("bogus")


def test_parser_profile_mode():
    p = rb.build_parser()
    base = ["--model", "Qwen/Qwen3-4B", "--draft-model", "d", "--datasets", "gsm8k", "--run-name", "x"]
    assert p.parse_args(base).profile_mode == "upstream"
    assert p.parse_args(base + ["--profile-mode", "warm"]).profile_mode == "warm"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--profile-mode", "hot"])


def test_summary_rows_split_by_profile_variant():
    from aggregate import summarize, table1_markdown
    from check_lossless import compare
    from score_math import score

    def rec(mode, tpot, budget=None, pm=None, ids=(1, 2, 3)):
        r = {"run_name": "s", "model": "Qwen/Qwen3-8B", "dataset": "gsm8k", "sample_idx": 0, "orig_idx": 0, "turn": 0,
             "mode": mode, "fixed_budget": budget, "time_per_output_token": tpot, "acceptance_lengths": [4],
             "num_input_tokens": 9, "num_output_tokens": 3, "max_new_tokens": 2048, "input_hash": "h",
             "output_ids": list(ids)}
        if pm is not None:
            r["profile_mode"] = pm
        return r

    recs = [rec("ar", 0.03, pm="warm"), rec("bastion", 0.005, pm="warm"), rec("bastion", 0.01, pm="upstream", ids=(1, 9)),
            rec("bastion", 0.006, budget=64, pm="warm")]
    rows = summarize(recs)
    by = {(r["method"], r["dataset"]): r for r in rows}
    assert by[("BASTION (adaptive)", "gsm8k")]["speedup"] == pytest.approx(6.0)
    assert by[("BASTION (adaptive, upstream cold profile)", "gsm8k")]["speedup"] == pytest.approx(3.0)
    assert by[("BASTION-Fixed-64", "gsm8k")]["profile_variant"] is None
    assert ("BASTION (adaptive)", "Average") in by and ("BASTION (adaptive, upstream cold profile)", "Average") in by
    md = table1_markdown(rows)
    assert "| BASTION (adaptive) |" in md and "| BASTION (adaptive, upstream cold profile) |" in md
    summary, _ = compare(recs)
    lab = {(s["method"], s["dataset"]): s for s in summary}
    assert lab[("BASTION (adaptive)", "gsm8k")]["exact_rate"] == 1.0
    assert lab[("BASTION (adaptive, upstream cold profile)", "gsm8k")]["exact_rate"] == 0.0
    sc, _ = score(recs, {"gsm8k": ["3"]}, lambda m, ids: "\\boxed{3}" if list(ids) == [1, 2, 3] else "\\boxed{4}")
    acc = {s["method"]: s["accuracy"] for s in sc}
    assert acc["BASTION (adaptive)"] == 1.0 and acc["BASTION (adaptive, upstream cold profile)"] == 0.0
