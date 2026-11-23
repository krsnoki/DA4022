"""E5 latency-model analysis on synthetic probe files shaped like upstream's cache."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

import latency_model  # noqa: E402
from bastion import cost_model  # noqa: E402


def _write_cache(cache, gpu="a100", model_key="qwen3-8b", model_name="Qwen/Qwen3-8B", a_m=1.3, b_m=0.004, noise=0.0002):
    rng = np.random.default_rng(0)
    probe = cache / f"calib_probe_{gpu}_{model_key}.sigma_sequence_v1.jsonl"
    lines = []
    for s in cost_model.build_fixed_interval_grid(16, 512, 16):
        for c in cost_model.build_power_of_two_grid(64, 2048):
            mem, comp = cost_model.compute_cost_model_latencies(model_name=model_name, gpu_type=gpu, s=s, c=c)
            y = max(a_m * mem + b_m, 2.0 * comp + 0.001) + rng.normal(0, noise)
            lines.append(json.dumps({"model_name_or_path": model_name, "gpu_type": gpu, "attn_impl": "sdpa", "s": s, "c": c,
                                     "verification_latency_s": y, "verification_latency_ms": y * 1e3,
                                     "verification_latency_std_s": noise, "num_repeats": 5}))
    probe.write_text("\n".join(lines) + "\n")
    params = cost_model.fit_linear_calibration(probe, sigma_scale=30.0)
    (cache / f"calibration_{gpu}_{model_key}.json").write_text(json.dumps(
        {"fit_parameters": params, "fit_version": "sigma_sequence_v1", "sigma_scale": 30.0, "attn_impl": "sdpa"}))
    (cache / f"calibration_{gpu}_{model_key}.profile.json").write_text(json.dumps(
        {"draft_latency_s": 0.005, "non_verify_latency_s": 0.001, "attn_impl": "sdpa"}))
    return probe


def test_find_cache_sets_parses_names(tmp_path):
    for name in ["calib_probe_b4500_qwen3-8b.sigma_sequence_v1.jsonl", "calib_probe_a100_llama3.1-8b-instruct.sigma_sequence_v1.jsonl",
                 "calibration_b4500_qwen3-8b.json", "gsm8k.jsonl"]:
        (tmp_path / name).write_text("")
    sets = latency_model.find_cache_sets(tmp_path)
    assert {(s["gpu_key"], s["model_key"]) for s in sets} == {("b4500", "qwen3-8b"), ("a100", "llama3.1-8b-instruct")}
    b = next(s for s in sets if s["gpu_key"] == "b4500")
    assert b["calibration"] is not None and b["profile"] is None


def test_evaluate_and_cli(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache)
    monkeypatch.delenv("CALIBRATION_JSON_PATH", raising=False)
    rec = {"mode": "bastion", "model_key": "qwen3-8b", "gpu_key": "a100", "tree_sizes": [64, 128],
           "tree_contexts": [100, 300], "verification_latencies": [0.02, 0.025]}
    runs = tmp_path / "r.jsonl"
    runs.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "out"
    latency_model.main(["--cache-dir", str(cache), "--out-dir", str(out), "--runs", str(runs)])
    res = json.loads((out / "latency_model.json").read_text())[0]
    assert res["n_points"] == 32 * 6
    assert res["controller_estimator_max_abs_diff_s"] < 1e-9  # our formula == upstream estimator
    a = res["rmse_all"]
    assert a["rmse_cal_ms"] < a["rmse_uncal_ms"] and a["reduction_pct"] > 50
    assert a["rmse_cal_ms"] < 1.0
    assert set(res["loco_rmse_cal_ms"]) == {"64", "128", "256", "512", "1024", "2048"}
    assert res["in_situ"]["n_steps"] == 2
    assert (out / "latency_points_a100_qwen3-8b.csv").exists()
    assert "CALIBRATION_JSON_PATH" not in __import__("os").environ


def test_no_probe_files(tmp_path):
    latency_model.main(["--cache-dir", str(tmp_path / "missing"), "--out-dir", str(tmp_path / "o")])
    assert json.loads((tmp_path / "o" / "latency_model.json").read_text())["status"] == "no probe files"
