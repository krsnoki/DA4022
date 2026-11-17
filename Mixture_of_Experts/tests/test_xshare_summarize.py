"""Tests for the GSM8K sweep summary (statistics, pairing, speed-ups, CSV).

Pure Python: synthetic run files are written to a temporary folder and read
back with ``load_runs``, so no model, GPU or network is needed.
"""

from __future__ import annotations

import csv
import json
import math

import pytest

from Mixture_of_Experts.experiments.xshare.summarize import (
    CSV_COLUMNS,
    build_rows,
    load_runs,
    mcnemar_exact,
    mean_or_none,
    paired_counts,
    routing_overhead,
    sd_or_none,
    speedups,
    summarise_group,
    wilson_interval,
    write_csv,
    write_overhead_csv,
)


# ------------------------------------------------------------------ helpers


def make_run(method, batch, subset, correct_flags, tps, step_ms=10.0, energy=None, generations=None):
    """Synthetic run dict. ``correct_flags`` is a list of bools (index = position)
    or a list of (index, bool) pairs, which lets a test shuffle the record order."""
    if correct_flags and not isinstance(correct_flags[0], (tuple, list)):
        pairs = list(enumerate(correct_flags))
    else:
        pairs = [tuple(p) for p in correct_flags]
    records = []
    for pos, (index, ok) in enumerate(pairs):
        gen = f"answer {index}" if generations is None else generations[pos]
        records.append({"index": index, "correct": bool(ok), "generation": gen})
    return {
        "method": method,
        "batch_size": batch,
        "subset": subset,
        "records": records,
        "decode_tokens_per_s": tps,
        "decode_step_median_ms": step_ms,
        "decode_step_p95_ms": step_ms * 1.5,
        "request_latency_median_s": 2.0,
        "request_latency_p95_s": 3.0,
        "peak_memory_allocated_mb": 1000.0 + subset,
        "routing": {
            "mean_kept_experts": 20.0,
            "mean_activated_experts": 18.0,
            "mean_tokens_per_step": 3.5,
        },
        "energy": energy,
    }


def save(folder, run):
    name = f"xshare-gsm8k-{run['method']}-c{run['batch_size']}-s{run['subset']}.json"
    (folder / name).write_text(json.dumps(run), encoding="utf-8")


def energy_dict(jpt, above=None):
    return {"j_per_token": jpt, "j_per_token_above_idle": above}


BASE_S0 = [True, True, False, False]
BASE_S1 = [True, False, False, False]
METH_S0 = [True, False, True, False]
METH_S1 = [True, True, False, False]


def standard_groups(tmp_path, method="xshare_a"):
    for s, flags, tps in ((0, BASE_S0, 100.0), (1, BASE_S1, 200.0)):
        save(tmp_path, make_run("baseline", 4, s, flags, tps, step_ms=10.0 + 2 * s))
    for s, flags, tps in ((0, METH_S0, 110.0), (1, METH_S1, 200.0)):
        save(tmp_path, make_run(method, 4, s, flags, tps))
    return load_runs(tmp_path)


# ------------------------------------------------------------- statistics


def test_wilson_empty_sample_is_nan():
    """Would catch a ZeroDivisionError or a made-up interval when n == 0."""
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_half_known_values():
    """Would catch a wrong z or formula (50/100 must give about 0.4038 to 0.5962)."""
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=5e-4)
    assert hi == pytest.approx(0.5962, abs=5e-4)


def test_wilson_zero_successes_lower_bound_zero():
    """Would catch a negative lower bound (or a zero upper bound) when k == 0."""
    lo, hi = wilson_interval(0, 20)
    assert 0.0 <= lo <= 1e-12
    assert hi > 0.0


def test_wilson_all_successes_upper_bound_one():
    """Would catch an upper bound above 1 (or a lower bound of 1) when k == n."""
    lo, hi = wilson_interval(20, 20)
    assert hi <= 1.0
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert lo < 1.0


@pytest.mark.parametrize("k,n", [(0, 5), (1, 5), (3, 7), (50, 100), (99, 100), (10, 10)])
def test_wilson_contains_point_estimate(k, n):
    """Would catch an interval that does not contain the observed proportion."""
    lo, hi = wilson_interval(k, n)
    assert lo <= k / n <= hi


def test_wilson_narrows_with_more_data():
    """Would catch an interval that ignores n (same width for 10 and 1000 questions)."""
    lo_s, hi_s = wilson_interval(5, 10)
    lo_l, hi_l = wilson_interval(500, 1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_mcnemar_no_discordant_pairs():
    """Would catch a division by zero or p != 1 when nothing differs."""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_known_values():
    """Would catch a wrong tail sum or a missing factor 2 in the exact test."""
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    assert mcnemar_exact(1, 9) == 0.021484375


def test_mcnemar_symmetric():
    """Would catch a p-value that depends on which method is called b and which c."""
    assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)


def test_mcnemar_even_split_is_capped_at_one():
    """Would catch a p-value above 1 for a perfectly even split (2 * tail > 1)."""
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_stays_in_unit_interval():
    """Would catch any p-value outside [0, 1] over a grid of counts."""
    for b in range(0, 12):
        for c in range(0, 12):
            assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_mean_or_none_propagates_missing():
    """Would catch a mean silently computed over a list with a missing value."""
    assert mean_or_none([]) is None
    assert mean_or_none([1.0, None]) is None
    assert mean_or_none([1.0, 3.0]) == 2.0


def test_sd_or_none_needs_two_values():
    """Would catch an sd reported for one subset, or with a missing value."""
    assert sd_or_none([5.0]) is None
    assert sd_or_none([]) is None
    assert sd_or_none([1.0, None]) is None
    assert sd_or_none([1.0, 3.0]) == pytest.approx(math.sqrt(2.0))


# ----------------------------------------------------------- run grouping


def test_load_runs_groups_by_batch_method_and_subset(tmp_path):
    """Would catch runs filed under the wrong (batch, method) key or subset."""
    groups = standard_groups(tmp_path)
    assert set(groups) == {(4, "baseline"), (4, "xshare_a")}
    assert sorted(groups[(4, "baseline")]) == [0, 1]


def test_pooled_accuracy_over_subsets(tmp_path):
    """Would catch accuracy computed per subset or averaged instead of pooled."""
    groups = standard_groups(tmp_path)
    row = summarise_group(4, "baseline", groups[(4, "baseline")], groups[(4, "baseline")])
    assert row["n_questions"] == 8
    assert row["n_subsets"] == 2
    assert row["accuracy"] == pytest.approx(3 / 8)
    assert row["acc_ci_low"] <= 3 / 8 <= row["acc_ci_high"]


def test_paired_counts_match_by_subset_and_index(tmp_path):
    """Would catch discordant counts taken per subset or mixing up b and c."""
    groups = standard_groups(tmp_path)
    b, c = paired_counts(groups[(4, "xshare_a")], groups[(4, "baseline")])
    # subset 0: idx1 base-only, idx2 method-only; subset 1: idx1 method-only.
    assert (b, c) == (1, 2)


def test_paired_counts_ignore_record_order(tmp_path):
    """Would catch positional matching of records instead of matching on (subset, index)."""
    save(tmp_path, make_run("baseline", 4, 0, BASE_S0, 100.0))
    shuffled = [(3, False), (2, True), (1, False), (0, True)]  # same answers as METH_S0
    save(tmp_path, make_run("xshare_a", 4, 0, shuffled, 110.0))
    groups = load_runs(tmp_path)
    assert paired_counts(groups[(4, "xshare_a")], groups[(4, "baseline")]) == (1, 1)


def test_paired_counts_do_not_mix_subsets(tmp_path):
    """Would catch the same question index in different subsets being treated as one question."""
    save(tmp_path, make_run("baseline", 4, 0, [True], 100.0))
    save(tmp_path, make_run("baseline", 4, 1, [False], 100.0))
    save(tmp_path, make_run("xshare_a", 4, 0, [False], 100.0))
    save(tmp_path, make_run("xshare_a", 4, 1, [True], 100.0))
    groups = load_runs(tmp_path)
    assert paired_counts(groups[(4, "xshare_a")], groups[(4, "baseline")]) == (1, 1)


def test_speedups_are_computed_per_subset(tmp_path):
    """Would catch a speed-up from pooled means instead of per-subset ratios."""
    groups = standard_groups(tmp_path)
    sp = speedups(groups[(4, "xshare_a")], groups[(4, "baseline")])
    assert sp == pytest.approx([10.0, 0.0])
    row = summarise_group(4, "xshare_a", groups[(4, "xshare_a")], groups[(4, "baseline")])
    assert row["speedup_pct_mean"] == pytest.approx(5.0)
    assert row["speedup_pct_sd"] == pytest.approx(math.sqrt(50.0))


def test_method_row_has_comparison_columns(tmp_path):
    """Would catch a method row with a wrong accuracy change or McNemar result."""
    groups = standard_groups(tmp_path)
    row = summarise_group(4, "xshare_a", groups[(4, "xshare_a")], groups[(4, "baseline")])
    assert row["acc_change_pt"] == pytest.approx(100 * (4 / 8 - 3 / 8))
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (1, 2)
    assert row["mcnemar_p"] == pytest.approx(mcnemar_exact(1, 2))


def test_baseline_row_has_no_comparison_columns(tmp_path):
    """Would catch the baseline being compared against itself (change 0, p 1, speed-up 0)."""
    groups = standard_groups(tmp_path)
    row = summarise_group(4, "baseline", groups[(4, "baseline")], groups[(4, "baseline")])
    for column in ("acc_change_pt", "mcnemar_b", "mcnemar_c", "mcnemar_p",
                   "speedup_pct_mean", "speedup_pct_sd"):
        assert row.get(column) is None


def test_tps_mean_sd_and_peak_memory(tmp_path):
    """Would catch peak memory averaged instead of maximised, or a wrong tokens/s spread."""
    groups = standard_groups(tmp_path)
    row = summarise_group(4, "baseline", groups[(4, "baseline")], None)
    assert row["tps_mean"] == pytest.approx(150.0)
    assert row["tps_sd"] == pytest.approx(math.sqrt(5000.0))
    assert row["peak_memory_mb"] == 1001.0
    assert row["step_median_ms"] == pytest.approx(11.0)


def test_build_rows_puts_baseline_first_within_batch(tmp_path):
    """Would catch rows sorted alphabetically so a method precedes its baseline."""
    save(tmp_path, make_run("aaa_method", 4, 0, BASE_S0, 100.0))
    save(tmp_path, make_run("baseline", 4, 0, BASE_S0, 100.0))
    save(tmp_path, make_run("zzz_method", 4, 0, BASE_S0, 100.0))
    save(tmp_path, make_run("baseline", 2, 0, BASE_S0, 100.0))
    save(tmp_path, make_run("aaa_method", 2, 0, BASE_S0, 100.0))
    rows = build_rows(load_runs(tmp_path))
    order = [(r["batch_size"], r["method"]) for r in rows]
    assert order == [
        (2, "baseline"), (2, "aaa_method"),
        (4, "baseline"), (4, "aaa_method"), (4, "zzz_method"),
    ]


# ------------------------------------------------------- routing overhead


def keep_all_folder(tmp_path, differ):
    for s, flags in ((0, BASE_S0), (1, BASE_S1)):
        save(tmp_path, make_run("baseline", 4, s, flags, 100.0, step_ms=10.0 + 2 * s))
        gens = [f"answer {i}" for i in range(4)]
        if differ and s == 1:
            gens[2] = "something else"
        save(tmp_path, make_run("xshare_keep_all", 4, s, flags, 100.0, step_ms=11.0 + 3 * s,
                                generations=gens))
    return load_runs(tmp_path)


def test_routing_overhead_identical_generations(tmp_path):
    """Would catch an overhead that is not keep-all minus baseline, or false generation diffs."""
    out = routing_overhead(keep_all_folder(tmp_path, differ=False))
    assert len(out) == 1
    assert out[0]["batch_size"] == 4
    # Baseline 10 and 12 ms, keep-all 11 and 14 ms: differences 1 and 2.
    assert out[0]["overhead_ms"] == pytest.approx(1.5)
    assert out[0]["n_matched"] == 8
    assert out[0]["n_differ"] == 0


def test_routing_overhead_counts_differing_generation(tmp_path):
    """Would catch a differing generation being missed or counted more than once."""
    out = routing_overhead(keep_all_folder(tmp_path, differ=True))
    assert out[0]["n_differ"] == 1


def test_routing_overhead_skipped_without_keep_all(tmp_path):
    """Would catch a KeyError or an empty row when there is no keep-all run."""
    groups = standard_groups(tmp_path)
    assert routing_overhead(groups) == []


# ----------------------------------------------------------------- energy


def test_energy_missing_in_one_subset_gives_none(tmp_path):
    """Would catch J/token averaged over only the subsets that have a measurement."""
    save(tmp_path, make_run("baseline", 4, 0, BASE_S0, 100.0, energy=energy_dict(1.0, 0.5)))
    save(tmp_path, make_run("baseline", 4, 1, BASE_S1, 100.0, energy=None))
    groups = load_runs(tmp_path)
    row = summarise_group(4, "baseline", groups[(4, "baseline")], None)
    assert row["j_per_token"] is None
    assert row["j_per_token_above_idle"] is None


def test_energy_present_in_all_subsets_is_averaged(tmp_path):
    """Would catch the wrong energy field being read or not averaged over subsets."""
    save(tmp_path, make_run("baseline", 4, 0, BASE_S0, 100.0, energy=energy_dict(1.0, 0.5)))
    save(tmp_path, make_run("baseline", 4, 1, BASE_S1, 100.0, energy=energy_dict(3.0, 1.5)))
    groups = load_runs(tmp_path)
    row = summarise_group(4, "baseline", groups[(4, "baseline")], None)
    assert row["j_per_token"] == pytest.approx(2.0)
    assert row["j_per_token_above_idle"] == pytest.approx(1.0)


# -------------------------------------------------------------------- CSV


def test_write_csv_header_and_empty_cells(tmp_path):
    """Would catch a changed column set or None written as the text 'None'."""
    groups = standard_groups(tmp_path)
    rows = build_rows(groups)
    out = tmp_path / "summary.csv"
    write_csv(rows, out)
    with out.open(newline="", encoding="utf-8") as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == CSV_COLUMNS
    assert len(lines) == 1 + len(rows)
    base = dict(zip(CSV_COLUMNS, lines[1]))
    assert base["method"] == "baseline"
    for column in ("acc_change_pt", "mcnemar_b", "mcnemar_c", "mcnemar_p",
                   "speedup_pct_mean", "speedup_pct_sd", "j_per_token"):
        assert base[column] == ""
    assert base["accuracy"] == "0.375"
    assert all(cell != "None" for line in lines for cell in line)


def test_routing_overhead_finds_every_keep_all_variant(tmp_path):
    """Would catch a keep-all control with a suffix (e.g. a warm-up variant) being ignored."""
    for s in (0, 1):
        save(tmp_path, make_run("baseline", 4, s, BASE_S0, 100.0, step_ms=10.0))
        save(tmp_path, make_run("xshare_keep_all", 4, s, BASE_S0, 100.0, step_ms=11.0))
        save(tmp_path, make_run("xshare_keep_all_k0-4", 4, s, BASE_S0, 100.0, step_ms=12.5))
    out = routing_overhead(load_runs(tmp_path))
    by_method = {o["method"]: o for o in out}
    assert set(by_method) == {"xshare_keep_all", "xshare_keep_all_k0-4"}
    assert by_method["xshare_keep_all"]["overhead_ms"] == pytest.approx(1.0)
    assert by_method["xshare_keep_all_k0-4"]["overhead_ms"] == pytest.approx(2.5)


def test_write_overhead_csv(tmp_path):
    """Would catch a missing header or a None written as text in the overhead file."""
    path = tmp_path / "routing_overhead.csv"
    write_overhead_csv(
        [{"batch_size": 8, "method": "xshare_keep_all", "overhead_ms": 4.0, "n_matched": 600, "n_differ": 0},
         {"batch_size": 32, "method": "xshare_keep_all", "overhead_ms": None, "n_matched": 0, "n_differ": 0}],
        path,
    )
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["batch_size", "method", "overhead_ms", "n_matched", "n_differ"]
    assert rows[0]["overhead_ms"] == "4.0"
    assert rows[1]["overhead_ms"] == ""
