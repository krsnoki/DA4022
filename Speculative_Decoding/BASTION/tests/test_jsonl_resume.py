"""JSONL writer / resume logic."""
from __future__ import annotations

import json

import pytest

import repro_lib as rl


def r(idx, mode="ar", budget=None, turn=0):
    return {"dataset": "gsm8k", "sample_idx": idx, "turn": turn, "mode": mode, "fixed_budget": budget, "x": idx}


def test_write_resume_and_duplicates(tmp_path):
    p = tmp_path / "run.jsonl"
    w = rl.JsonlWriter(p)
    w.write(r(0))
    w.write(r(0, "bastion", 64))
    w.write(r(0, "bastion", None))
    with pytest.raises(KeyError):
        w.write(r(0))
    w.close()
    w2 = rl.JsonlWriter(p)
    assert w2.has(("gsm8k", 0, 0, "ar", None))
    assert w2.has(("gsm8k", 0, 0, "bastion", 64)) and w2.has(("gsm8k", 0, 0, "bastion", None))
    assert not w2.has(("gsm8k", 1, 0, "ar", None))
    assert w2.get(("gsm8k", 0, 0, "bastion", 64))["x"] == 0
    w2.write(r(1))
    w2.close()
    assert len(rl.load_jsonl(p)) == 4


def test_truncated_tail_is_repaired(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(r(0)) + "\n" + json.dumps(r(1))[:15])  # crash mid-write
    assert len(rl.load_jsonl(p)) == 1
    w = rl.JsonlWriter(p)
    assert w.has(("gsm8k", 0, 0, "ar", None)) and not w.has(("gsm8k", 1, 0, "ar", None))
    w.write(r(1))
    w.close()
    recs = rl.load_jsonl(p)
    assert [x["sample_idx"] for x in recs] == [0, 1]


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(r(0)) + "\n{bad\n" + json.dumps(r(1)) + "\n")
    with pytest.raises(json.JSONDecodeError):
        rl.load_jsonl(p)


def test_parse_caps_and_budgets():
    assert rl.parse_caps(["aime25=30", "mt-bench=40"]) == {"aime25": 30, "mt-bench": 40}
    with pytest.raises(ValueError):
        rl.parse_caps(["aime25"])
    assert rl.parse_budgets(["adaptive", "32,64", "64", "1024"]) == [None, 32, 64, 1024]
    with pytest.raises(ValueError):
        rl.parse_budgets(["1"])
