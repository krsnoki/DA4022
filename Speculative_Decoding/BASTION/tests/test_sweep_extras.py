"""sweep_extras.py on a hand-computed synthetic sweep + E1 pair."""
from __future__ import annotations

import json

import pytest

import sweep_extras

M = "Qwen/Qwen3-8B"


def rec(ds, idx, mode, tpot, budget=None, profile="warm", sizes=None):
    return {"run_name": "x", "model": M, "dataset": ds, "orig_idx": idx, "sample_idx": idx, "turn": 0, "mode": mode,
            "fixed_budget": budget, "profile_mode": profile, "time_per_output_token": tpot,
            "input_hash": f"{ds}-{idx}", "tree_sizes": list(sizes or [])}


def write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


def test_oracle_and_ratios(tmp_path):
    e1, sweep = [], []
    # gsm8k: AR 0.02; fixed-32 0.004 (5x), fixed-64 0.005 (4x); adaptive warm 0.004; cold 0.005; dflash 0.008
    # mbpp:  AR 0.02; fixed-32 0.005 (4x), fixed-64 0.004 (5x); adaptive warm 0.005; cold 0.004; dflash 0.008
    for ds, f32, f64, warm, cold in (("gsm8k", 0.004, 0.005, 0.004, 0.005), ("mbpp", 0.005, 0.004, 0.005, 0.004)):
        for i in range(2):
            e1 += [rec(ds, i, "ar", 0.02), rec(ds, i, "dflash", 0.008), rec(ds, i, "bastion", warm, sizes=[32, 64])]
            sweep += [rec(ds, i, "bastion", f32, budget=32), rec(ds, i, "bastion", f64, budget=64),
                      rec(ds, i, "bastion", warm, sizes=[32]), rec(ds, i, "bastion", cold, profile="upstream", sizes=[64])]
    sweep.append(rec("gsm8k", 9, "bastion", 0.001, budget=32))  # no AR / other configs -> skipped
    sweep_extras.main(["--sweep", write(tmp_path / "s.jsonl", sweep), "--e1", write(tmp_path / "e.jsonl", e1),
                       "--out-dir", str(tmp_path), "--prefix", "t_", "--n-boot", "50"])
    res = json.loads((tmp_path / "t_extras.json").read_text())
    assert res["n_keys"] == 4 and res["n_keys_skipped"] == 1
    assert res["best_fixed_per_dataset"] == {"gsm8k": "Fixed-32", "mbpp": "Fixed-64"}
    assert res["oracle_average"] == pytest.approx(5.0)
    avg = res["speedup_average"]
    assert avg["Fixed-32"] == pytest.approx(4.5) and avg["Adaptive warm"] == pytest.approx(4.5)
    assert avg["DFlash (E1)"] == pytest.approx(2.5)
    r = {(x["num"], x["den"]): x for x in res["ratios"]}
    assert r[("Adaptive warm", "Oracle (best fixed N per dataset)")]["point"] == pytest.approx(0.9)
    assert r[("Adaptive warm", "Adaptive warm (E1 repeat)")]["point"] == pytest.approx(1.0)
    assert r[("Adaptive warm", "DFlash (E1)")]["point"] == pytest.approx(1.8)
    trees = res["adaptive_tree_sizes"]
    assert trees["Adaptive warm"]["pooled_mean"] == pytest.approx(32)
    assert trees["Adaptive warm (E1 repeat)"]["pooled_mean"] == pytest.approx(48)
    assert trees["Adaptive upstream-cold"]["pooled_mean"] == pytest.approx(64)
