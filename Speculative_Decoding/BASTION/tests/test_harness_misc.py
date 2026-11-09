"""Harness pieces that run without a GPU: CLI parsing, GPU registration, summary printing, plots."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

import repro_lib as rl  # noqa: E402


def test_local_gpu_registration():
    info = rl.register_local_gpu_specs()
    from bastion import cost_model

    key, spec = cost_model.resolve_gpu_spec("NVIDIA RTX PRO 4500 Blackwell")
    assert key == "b4500" and spec["memory_bandwidth"] == 896e9
    assert "b4500" in info["added_gpu_specs"]
    # upstream entries untouched
    assert cost_model.resolve_gpu_spec("NVIDIA A100 80GB PCIe")[0] == "a100"


def test_parser_defaults():
    import run_benchmark

    a = run_benchmark.build_parser().parse_args(["--model", "Qwen/Qwen3-8B", "--draft-model", "z-lab/Qwen3-8B-DFlash-b16",
                                                 "--datasets", "gsm8k", "mt-bench", "--run-name", "x",
                                                 "--cap", "mt-bench=40", "--budgets", "adaptive", "32,64"])
    assert a.max_new_tokens == 2048 and a.temperature == 0.0 and a.seed == 0 and a.warmup == 0
    assert a.modes == ["ar", "dflash", "bastion"]
    assert rl.parse_budgets(a.budgets) == [None, 32, 64]
    assert run_benchmark.CACHE_DIR == rl.UPSTREAM / "cache"


def test_print_summary_runs(capsys):
    import run_benchmark

    recs = []
    for i in range(2):
        for mode, tpot, acc in [("ar", 0.02, [1, 1]), ("dflash", 0.005, [4]), ("bastion", 0.004, [6])]:
            recs.append({"dataset": "gsm8k", "sample_idx": i, "turn": 0, "mode": mode, "fixed_budget": None,
                         "time_per_output_token": tpot, "acceptance_lengths": acc})
        recs.append({"dataset": "gsm8k", "sample_idx": i, "turn": 0, "mode": "bastion", "fixed_budget": 64,
                     "time_per_output_token": 0.005, "acceptance_lengths": [5], "tree_sizes": [64]})
    run_benchmark.print_summary(recs, 16, ["gsm8k"])
    out = capsys.readouterr().out
    assert "BASTION tree-draft" in out and "5.00x" in out and "N=64" in out and "4.00x" in out


def test_plots_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    import plots

    rows = []
    for mode, sp in [("dflash", 4.0), ("bastion", 6.0)]:
        for ds in ["gsm8k", "math500", "Average"]:
            rows.append({"model": "Qwen/Qwen3-8B", "mode": mode, "dataset": ds, "fixed_budget": None, "speedup": sp,
                         "aal": 5.0, "mean_tree_size": 100.0 if mode == "bastion" else None, "datasets_included": ["gsm8k", "math500"]})
    for n in (32, 64, 128):
        for ds in ["gsm8k", "Average"]:
            rows.append({"model": "Qwen/Qwen3-8B", "mode": "bastion", "dataset": ds, "fixed_budget": n, "speedup": 5 + n / 100,
                         "aal": 5.0, "mean_tree_size": float(n), "datasets_included": ["gsm8k"]})
    s = tmp_path / "s.json"
    s.write_text(json.dumps(rows))
    plots.main(["--summary", str(s), "--sweep-summary", str(s), "--out-dir", str(tmp_path / "fig")])
    names = {p.name for p in (tmp_path / "fig").iterdir()}
    assert {"speedup_bars_Qwen3-8B.png", "budget_sweep_Qwen3-8B.png", "budget_sweep_Qwen3-8B_per_dataset.png"} <= names
