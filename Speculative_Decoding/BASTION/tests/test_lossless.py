"""check_lossless.py on synthetic cases."""
from __future__ import annotations

import pytest

from check_lossless import compare, first_divergence

M = "Qwen/Qwen3-4B"


def rec(idx, mode, ids, h=None, budget=None, turn=0):
    return {"run_name": "r", "model": M, "dataset": "gsm8k", "sample_idx": idx, "turn": turn, "mode": mode,
            "fixed_budget": budget, "output_ids": ids, "input_hash": h or f"h{idx}-{turn}", "max_new_tokens": 2048}


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2, 3], [1, 2]) == 2
    assert first_divergence([], [5]) == 0
    assert first_divergence([], []) is None


def test_compare_cases():
    ar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    records = [
        rec(0, "ar", ar), rec(0, "dflash", list(ar)), rec(0, "bastion", ar[:3] + [99] + ar[4:]),
        rec(1, "ar", [5, 6, 7, 8]), rec(1, "dflash", [5, 6, 7, 8]), rec(1, "bastion", [5, 6]),
        rec(2, "ar", [1]), rec(2, "bastion", [1], h="different-prompt"),
        rec(3, "bastion", [4, 4]),  # no AR
        rec(1, "bastion", [5, 6, 7, 8], budget=64),
    ]
    summary, per = compare(records)
    s = {(x["dataset"], x["mode"], x["fixed_budget"]): x for x in summary}
    d = s[("gsm8k", "dflash", None)]
    assert d["n_compared"] == 2 and d["n_exact"] == 2 and d["exact_rate"] == 1.0 and d["n_diverged"] == 0
    b = s[("gsm8k", "bastion", None)]
    assert b["n_records"] == 4 and b["n_compared"] == 2 and b["n_no_ar"] == 1 and b["n_prompt_mismatch"] == 1
    assert b["n_exact"] == 0 and b["exact_rate"] == 0.0
    assert b["div_pos_min"] == 2 and b["div_pos_max"] == 3 and b["div_pos_median"] == pytest.approx(2.5)
    assert b["mean_prefix_agreement"] == pytest.approx((3 / 10 + 2 / 4) / 2)
    assert b["n_length_only_diff"] == 1
    f = s[("gsm8k", "bastion", 64)]
    assert f["exact_rate"] == 1.0
    overall = s[("ALL", "bastion", None)]
    assert overall["n_compared"] == 2
