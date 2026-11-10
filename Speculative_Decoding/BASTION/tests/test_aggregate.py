"""aggregate.py math on a hand-computed synthetic JSONL."""
from __future__ import annotations

import csv
import json

import pytest

import aggregate
from aggregate import ARIndex, summarize

M = "Qwen/Qwen3-8B"


def rec(ds, idx, mode, tpot, acc=(1,), budget=None, turn=0, h=None, run="r1", sizes=None, out=100):
    r = {
        "run_name": run, "model": M, "draft_model": "d", "dataset": ds, "sample_idx": idx, "orig_idx": idx,
        "turn": turn, "mode": mode, "fixed_budget": budget, "time_per_output_token": tpot,
        "acceptance_lengths": list(acc), "num_input_tokens": 50, "num_output_tokens": out, "max_new_tokens": 2048,
        "input_hash": h or f"{ds}-{idx}-{turn}", "output_ids": [1, 2, 3], "time_to_first_token": 0.02,
        "peak_mem_allocated_mib": 1000.0,
    }
    if sizes is not None:
        r["tree_sizes"] = list(sizes)
        r["mean_tree_size"] = sum(sizes) / len(sizes)
    return r


@pytest.fixture
def records():
    return [
        # dataset A: AR tpot 0.020, 0.030 ; dflash 0.005, 0.010 ; bastion 0.004, 0.006
        rec("gsm8k", 0, "ar", 0.020), rec("gsm8k", 1, "ar", 0.030),
        rec("gsm8k", 0, "dflash", 0.005, acc=(4, 6)), rec("gsm8k", 1, "dflash", 0.010, acc=(2,)),
        rec("gsm8k", 0, "bastion", 0.004, acc=(8, 10), sizes=(64, 96)), rec("gsm8k", 1, "bastion", 0.006, acc=(6,), sizes=(32,)),
        # dataset B: AR 0.040 ; dflash 0.020 ; bastion 0.010, plus a bastion record without AR
        rec("mt-bench", 0, "ar", 0.040), rec("mt-bench", 0, "dflash", 0.020, acc=(3, 3)),
        rec("mt-bench", 0, "bastion", 0.010, acc=(5,), sizes=(128,)),
        rec("mt-bench", 1, "bastion", 0.050, acc=(1,), sizes=(32,)),
    ]


def _row(rows, ds, mode, budget=None):
    return next(r for r in rows if r["dataset"] == ds and r["mode"] == mode and r["fixed_budget"] == budget)


def test_per_dataset_speedup_and_aal(records):
    rows = summarize(records)
    d = _row(rows, "gsm8k", "dflash")
    assert d["speedup"] == pytest.approx((0.020 + 0.030) / 2 / ((0.005 + 0.010) / 2))  # 3.333
    assert d["aal"] == pytest.approx((4 + 6 + 2) / 3)  # pooled over steps, not mean of per-record means
    b = _row(rows, "gsm8k", "bastion")
    assert b["speedup"] == pytest.approx(0.025 / 0.005)
    assert b["aal"] == pytest.approx(8.0)
    assert b["mean_tree_size"] == pytest.approx((64 + 96 + 32) / 3)
    assert b["mean_tree_size_per_record"] == pytest.approx((80 + 32) / 2)
    assert b["tokens_per_s"] == pytest.approx(1 / 0.005)
    ar = _row(rows, "gsm8k", "ar")
    assert ar["speedup"] == pytest.approx(1.0) and ar["aal"] is None
    # mt-bench bastion: one record lacks AR -> excluded from speedup, counted
    mb = _row(rows, "mt-bench", "bastion")
    assert mb["n_records"] == 2 and mb["n_matched_ar"] == 1 and mb["n_no_ar"] == 1
    assert mb["speedup"] == pytest.approx(0.040 / 0.010)
    assert mb["aal"] == pytest.approx(3.0)  # (5 + 1) / 2 steps, includes the unmatched record


def test_average_and_pooled_rows(records):
    rows = summarize(records)
    avg = _row(rows, "Average", "bastion")
    assert avg["speedup"] == pytest.approx((5.0 + 4.0) / 2)  # unweighted over datasets
    assert avg["aal"] == pytest.approx((8.0 + 3.0) / 2)
    pooled = _row(rows, "Pooled", "bastion")
    # matched records only: AR (0.02, 0.03, 0.04) vs bastion (0.004, 0.006, 0.010)
    assert pooled["speedup"] == pytest.approx((0.09 / 3) / (0.02 / 3))
    assert pooled["aal"] == pytest.approx((8 + 10 + 6 + 5 + 1) / 5)
    assert avg["missing_paper_datasets"] == ["math500", "aime25", "humaneval", "mbpp", "lcb", "alpaca"]


def test_hash_mismatch_excluded(records):
    records = records + [rec("alpaca", 0, "ar", 0.03), rec("alpaca", 0, "bastion", 0.01, h="other")]
    rows = summarize(records)
    r = _row(rows, "alpaca", "bastion")
    assert r["n_hash_mismatch"] == 1 and r["speedup"] is None
    rows2 = summarize(records, hash_check=False)
    assert _row(rows2, "alpaca", "bastion")["speedup"] == pytest.approx(3.0)


def test_external_ar_reference():
    sweep = [rec("gsm8k", 0, "bastion", 0.005, budget=64, run="sweep", acc=(5,), sizes=(64,)),
             rec("gsm8k", 1, "bastion", 0.005, budget=64, run="sweep", acc=(5,), sizes=(64,)),
             rec("gsm8k", 2, "bastion", 0.005, budget=64, run="sweep", acc=(5,), sizes=(64,))]
    e1 = [rec("gsm8k", 0, "ar", 0.02, run="e1"), rec("gsm8k", 1, "ar", 0.03, run="e1")]
    rows = summarize(sweep, ARIndex(sweep, e1))
    r = _row(rows, "gsm8k", "bastion", 64)
    assert r["n_matched_ar"] == 2 and r["n_no_ar"] == 1 and r["ar_runs_used"] == ["e1"]
    assert r["speedup"] == pytest.approx(0.025 / 0.005)
    assert r["method"] == "BASTION-Fixed-64"


def test_match_on_orig_idx_not_sample_idx():
    """Different caps can order the same problems differently (aime25: cap 30 unshuffled, cap 16 shuffled)."""
    e1 = [dict(rec("aime25", 0, "ar", 0.02, run="e1"), orig_idx=0), dict(rec("aime25", 1, "ar", 0.04, run="e1"), orig_idx=1)]
    sw = [dict(rec("aime25", 0, "bastion", 0.01, budget=32, run="sw", h="aime25-1-0"), orig_idx=1)]
    r = _row(summarize(sw, ARIndex(sw, e1)), "aime25", "bastion", 32)
    assert r["n_matched_ar"] == 1 and r["speedup"] == pytest.approx(4.0)


def test_same_run_ar_preferred_over_external():
    own = [rec("gsm8k", 0, "ar", 0.02, run="a"), rec("gsm8k", 0, "dflash", 0.01, run="a", acc=(2,))]
    ext = [rec("gsm8k", 0, "ar", 0.04, run="b")]
    rows = summarize(own, ARIndex(own, ext))
    assert _row(rows, "gsm8k", "dflash")["speedup"] == pytest.approx(2.0)


def test_cli_outputs_and_da4022_schema(records, tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    (tmp_path / "run.meta.json").write_text(json.dumps({"run_name": "r1", "sessions": [{
        "da4022_commit": "abc", "upstream_commit": "1989b80", "args": {"seed": 0},
        "env": {"gpu_name": "TestGPU", "packages": {"torch": "2.8", "transformers": "4.57.1"}, "torch_cuda": "12.8"},
        "hf_revisions": {"target": {"refs_main": "rev123"}}}]}))
    out = tmp_path / "out"
    aggregate.main(["--runs", str(path), "--out-dir", str(out)])
    with open(out / "da4022_results.csv") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == aggregate.DA4022_COLUMNS
        rows = list(reader)
    assert rows and all(r["energy_j"] == "" for r in rows)
    assert all("energy=not_measured" in r["notes"] for r in rows)
    g = next(r for r in rows if r["dataset_id"].startswith("openai/gsm8k") and r["method"] == "BASTION")
    assert g["device"] == "TestGPU" and g["backbone_revision"] == "rev123" and g["commit"] == "abc"
    assert g["quality_value"] == "" and "quality=not_scored" in g["notes"]
    assert float(g["throughput"]) == pytest.approx(200.0)
    md = (out / "table1.md").read_text()
    assert "BASTION" in md and "5.00×" in md
