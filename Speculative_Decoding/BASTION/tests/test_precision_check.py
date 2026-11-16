"""CPU test of analysis/precision_check.py with tiny random models (no downloads, no GPU)."""
from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from conftest import COST_GPU, COST_MODEL  # noqa: E402

import precision_check as pc  # noqa: E402

PROMPTS = [{"orig_idx": 7, "input_ids": [3, 17, 42, 8, 11, 5, 23]},
           {"orig_idx": 2, "input_ids": [9, 4, 33, 61, 12]}]


@pytest.fixture
def patched(tiny_models, calib_env, monkeypatch, tmp_path):
    calib = calib_env()
    prof = tmp_path / "profile.warm.json"
    prof.write_text(json.dumps({"draft_latency_s": 0.004, "non_verify_latency_s": 0.002, "profile_mode": "warm"}))
    target, draft = tiny_models

    def fake_load(args, dtype_name, device):
        dtype = getattr(torch, pc.DTYPES[dtype_name])
        return copy.deepcopy(target).to(dtype).eval(), copy.deepcopy(draft).to(dtype).eval()

    monkeypatch.setattr(pc, "prepare_prompts", lambda args: (copy.deepcopy(PROMPTS), None))
    monkeypatch.setattr(pc, "load_models", fake_load)
    return ["--model", "tiny", "--draft-model", "tiny-draft", "--device", "cpu", "--max-new-tokens", "12",
            "--model-key", COST_MODEL, "--gpu-key", COST_GPU, "--calibration-json", str(calib),
            "--profile-json", str(prof)]


def test_end_to_end_cpu(patched, tmp_path):
    out = tmp_path / "out"
    res = pc.main(patched + ["--out-dir", str(out)])
    assert (out / "precision_check.json").exists() and (out / "precision_check.md").exists()
    expected = {f"{m}-{dt} vs ar-{dt}" for m in ("dflash", "bastion") for dt in ("bf16", "fp32")}
    assert set(res["comparisons"]) == expected | {"ar-bf16 vs ar-fp32"}
    for dt in ("bf16", "fp32"):
        assert all(len(o) == 12 for m in pc.METHODS for o in res["outputs"][dt][m])  # eos=None -> full length
    # fp32 on the sharpened tiny target: greedy speculative decoding is lossless.
    for m in ("dflash", "bastion"):
        assert res["summary"][f"{m}-fp32 vs ar-fp32"]["exact_match_rate"] == 1.0
    for rows in res["comparisons"].values():
        for r in rows:
            if r["first_divergence"] is not None:
                assert r["top1_top2_margin"] >= 0
    saved = json.loads((out / "precision_check.json").read_text())
    assert saved["bastion"]["cost_gpu_type"] == COST_GPU


def test_margin_at_matches_ar(tiny_models, tmp_path):
    target, draft = tiny_models
    p = PROMPTS[0]
    ar = pc.generate("ar", target, draft, torch.tensor([p["input_ids"]]), max_new_tokens=10, eos=None,
                     block_size=draft.block_size, bastion_kw={})
    m = pc.margin_at(target, p["input_ids"], ar[:5], ar[5], (ar[5] + 1) % 96)
    assert m["top1_top2_margin"] >= 0 and m["teacher_forced_argmax_equals_ar"]
    assert m["ar_minus_other_logit"] > 0
    row = pc.divergence_row(target, p, ar, ar[:4] + [(ar[4] + 1) % 96] + ar[5:])
    assert row["first_divergence"] == 4 and row["ref_token"] == ar[4]
    assert pc.divergence_row(target, p, ar, list(ar))["first_divergence"] is None
    s = pc.summarize([row, pc.divergence_row(target, p, ar, list(ar))])
    assert s["exact_match"] == 1 and s["first_divergences"] == [4] and s["margins"] == [row["top1_top2_margin"]]
    result = {"args": {"model": "t", "draft_model": "d", "max_new_tokens": 10, "device": "cpu"},
              "prompts": [p], "comparisons": {"x vs y": [row]}, "summary": {"x vs y": pc.summarize([row])}}
    pc.write_markdown(tmp_path / "s.md", result)
    assert "pos 4" in (tmp_path / "s.md").read_text()


def test_refuses_out_dir_in_upstream(patched):
    with pytest.raises(ValueError, match="upstream"):
        pc.main(patched + ["--out-dir", str(pc.ROOT / "upstream" / "cache" / "x")])
