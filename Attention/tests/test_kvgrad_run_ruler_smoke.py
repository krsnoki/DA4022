"""
CPU smoke test for Attention/experiments/kvgrad/run_ruler.py.

Builds a tiny, randomly initialised Qwen3 model and a toy RULER-style dataset in a temporary
folder (no downloads, no GPU), runs every press through the benchmark script and checks that
predictions, metrics and the summary CSV are written with the expected structure. It also checks
resuming: a second call must skip finished configurations.

Run from the repository root:
    python -m pytest Attention/tests/test_kvgrad_run_ruler_smoke.py -q
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kvpress")

from _kvgrad_tiny_model import WORDS, make_tiny_model

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "kvgrad" / "run_ruler.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_ruler", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_toy_ruler(path: Path) -> None:
    rows = []
    for t, task in enumerate(["niah_single_1", "qa_1"]):
        for i in range(3):
            # every row has a different context (the score cache is keyed by the context)
            context = " ".join(WORDS * (4 + t)) + f" the secret number is {10 + i} " + " ".join(WORDS * 4)
            rows.append(
                {
                    "context": context,
                    "question": " what is the secret number ?",
                    "answer_prefix": " answer :",
                    "answer": [str(10 + i)],
                    "task": task,
                    "max_new_tokens": 4,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("kvgrad_smoke")
    make_tiny_model(root / "model")
    _make_toy_ruler(root / "toy_ruler.parquet")
    return root


def test_run_ruler_end_to_end(tiny_setup):
    run_ruler = _load_script()
    out_dir = tiny_setup / "results"
    argv = [
        "--model",
        str(tiny_setup / "model"),
        "--data_file",
        str(tiny_setup / "toy_ruler.parquet"),
        "--device",
        "cpu",
        "--per_task",
        "2",
        "--presses",
        "no_press",
        "kvzip",
        "kvzip_plus",
        "kvgrad",
        "--compression_ratios",
        "0.5",
        "--warmup",
        "1",
        "--output_dir",
        str(out_dir),
    ]
    summary_file = run_ruler.main(argv)

    # Same 2 x 2 = 4 sampled rows for every press
    samples = pd.read_csv(out_dir / "samples_2pertask_seed42.csv")
    assert len(samples) == 4 and samples["task"].value_counts().to_dict() == {"niah_single_1": 2, "qa_1": 2}

    expected_runs = ["no_press_cr0.000", "kvzip_cr0.500", "kvzip_plus_cr0.500", "kvgrad_cr0.500"]
    for name in expected_runs:
        predictions = pd.read_csv(out_dir / name / "predictions.csv")
        assert sorted(predictions["row_id"]) == sorted(samples["row_id"])
        assert (predictions["total_ms"] > 0).all()
        assert (predictions["decode_ms"] >= 0).all() and (predictions["prefill_compress_ms"] > 0).all()
        metrics = json.loads((out_dir / name / "metrics.json").read_text())
        assert set(metrics["string_match_per_task"]) == {"niah_single_1", "qa_1"}
        assert 0.0 <= metrics["string_match_macro"] <= 100.0
        # 2 layers x 2 KV heads x 16 dims x (K and V) x 4 bytes = 512 bytes per token
        expected_full = metrics["input_tokens_median"] * 512 / 2**20
        assert metrics["kv_cache_mb_full_median"] == pytest.approx(expected_full, abs=0.05)
        assert metrics["kv_cache_mb_kept_median"] <= metrics["kv_cache_mb_full_median"]

    summary = pd.read_csv(summary_file)
    assert list(summary.columns) == run_ruler.SUMMARY_COLUMNS
    assert sorted(summary["method"]) == sorted(["no_press", "kvzip", "kvzip_plus", "kvgrad"])
    assert summary["energy_j"].isna().all()  # never invent energy numbers

    # Resuming: a second call must skip every finished configuration and keep the same summary
    run_ruler.main(argv)
    assert pd.read_csv(summary_file).equals(summary)


def test_scoring_matches_kvpress_ruler_metric():
    run_ruler = _load_script()
    df = pd.DataFrame(
        {
            "task": ["niah_single_1", "niah_single_1", "qa_1", "qa_1"],
            "predicted_answer": ["The number is 42.", "no idea", " Paris\n", "London"],
            "answer": [["42"], ["7"], ["Paris", "paris, France"], ["Rome"]],
        }
    )
    # niah uses string_match_all, qa uses string_match_part: 1 hit out of 2 for each task
    assert run_ruler.ruler_scores(df) == {"niah_single_1": 50.0, "qa_1": 50.0}


def test_config_rejects_unknown_press():
    run_ruler = _load_script()
    with pytest.raises(ValueError, match="Unknown press"):
        run_ruler.load_config(["--presses", "not_a_press"])


def test_run_ruler_kvgrad_variants_share_scores(tiny_setup):
    """Our variant presses run through the script; the first configuration replays each context once and
    every later configuration reuses those scores. Our bound_max must answer exactly like kvpress's kvgrad."""
    run_ruler = _load_script()
    run_ruler.SCORE_CACHE.clear()
    out_dir = tiny_setup / "results_variants"
    argv = [
        "--model",
        str(tiny_setup / "model"),
        "--data_file",
        str(tiny_setup / "toy_ruler.parquet"),
        "--device",
        "cpu",
        "--per_task",
        "2",
        "--presses",
        "kvgrad",
        "kvgrad_bound_max",
        "kvgrad_bound_sum",
        "kvgrad_taylor_max",
        "kvgrad_taylor_sum",
        "--compression_ratios",
        "0.5",
        "0.75",
        "--warmup",
        "1",
        "--output_dir",
        str(out_dir),
    ]
    summary = pd.read_csv(run_ruler.main(argv))
    assert len(summary) == 10

    def metrics(name):
        return json.loads((out_dir / name / "metrics.json").read_text())

    assert metrics("kvgrad_bound_max_cr0.500")["scoring_cached_fraction"] == 0.0  # replayed every context
    for name in ["kvgrad_bound_max_cr0.750", "kvgrad_taylor_sum_cr0.500", "kvgrad_taylor_sum_cr0.750"]:
        assert metrics(name)["scoring_cached_fraction"] == 1.0  # reused the scores, no replay
    assert metrics("kvgrad_cr0.500")["scoring_cached_fraction"] == 0.0  # kvpress press: never cached

    for ratio in ["0.500", "0.750"]:
        ours = pd.read_csv(out_dir / f"kvgrad_bound_max_cr{ratio}" / "predictions.csv").sort_values("row_id")
        ref = pd.read_csv(out_dir / f"kvgrad_cr{ratio}" / "predictions.csv").sort_values("row_id")
        assert ours["predicted_answer"].tolist() == ref["predicted_answer"].tolist()
    assert summary["notes"].str.contains("scores reused without replay").sum() == 7
