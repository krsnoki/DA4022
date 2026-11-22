"""Unit and CLI integration tests for the checkpointed DIPA benchmark runner."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from Adaptive_Stopping.experiments.run_dipa_benchmark import (
    _build_allocator,
    append_pool_entry,
    build_response_pool,
    evaluate_pool,
    load_pool_jsonl,
    run_benchmark,
)
from Adaptive_Stopping.methods.dipa.types import AllocationConfig
from Dataloaders.evaluation.math500 import Math500LoaderConfig, load_math500
from Performance_Metrics import METRICS_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Adaptive_Stopping" / "configs" / "dipa.yaml"
FIXTURE_PATH = (
    REPO_ROOT / "Dataloaders" / "evaluation" / "math500" / "sample_fixture.jsonl"
)


class _StubBackbone:
    """CPU stub mimicking Qwen25Math15BInstructBackbone.generate()."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        raw_inputs: str | list[dict[str, str]],
        num_return_sequences: int = 1,
        seed: int | None = None,
        **gen_kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "raw_inputs": raw_inputs,
                "num_return_sequences": num_return_sequences,
                "seed": seed,
            }
        )
        return [
            {
                "text": r"Reasoning... \boxed{5}",
                "num_tokens": 12 + idx,
                "token_ids": list(range(12 + idx)),
                "latency_ms": 8.0 + idx,
            }
            for idx in range(num_return_sequences)
        ]


class TestResponsePoolCheckpointing:
    """Tests for JSONL checkpointing, incremental resume, and HF chunking."""

    def test_checkpoint_resume_only_generates_missing_instances(
        self, tmp_path: Path
    ) -> None:
        records = load_math500(Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH))
        assert len(records) == 4
        pool_path = tmp_path / "pool.jsonl"

        sentinel_attempt = {
            "text": r"Precomputed \boxed{5}",
            "num_tokens": 7,
            "latency_ms": 3.5,
            "extracted_answer": "5",
            "is_correct": True,
        }
        for rec in records[:2]:
            append_pool_entry(
                pool_path,
                {
                    "instance_id": rec["instance_id"],
                    "prompt": rec["prompt"],
                    "ground_truth": rec["ground_truth"],
                    "metadata": rec["metadata"],
                    "attempts": [dict(sentinel_attempt) for _ in range(4)],
                },
            )

        assert len(pool_path.read_text(encoding="utf-8").splitlines()) == 2

        pool_entries, peak_mem = build_response_pool(
            records=records,
            mode="mock",
            pool_size=4,
            pool_path=pool_path,
            seed=42,
        )
        assert peak_mem is None
        assert set(pool_entries.keys()) == {r["instance_id"] for r in records}
        assert all(len(e["attempts"]) == 4 for e in pool_entries.values())

        lines = pool_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        reloaded = load_pool_jsonl(pool_path)
        for rec in records[:2]:
            assert reloaded[rec["instance_id"]]["attempts"] == [
                sentinel_attempt for _ in range(4)
            ]

    def test_hf_mode_chunked_generation_and_verification(
        self, tmp_path: Path
    ) -> None:
        records = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=2, seed=42)
        )
        pool_path = tmp_path / "hf_pool.jsonl"
        stub = _StubBackbone()

        entries, _ = build_response_pool(
            records=records,
            mode="hf",
            pool_size=5,
            pool_path=pool_path,
            backbone=stub,
            gen_batch_size=3,
            seed=42,
        )
        assert [c["num_return_sequences"] for c in stub.calls] == [3, 2, 3, 2]
        assert all(isinstance(c["raw_inputs"], list) for c in stub.calls)
        for rec in records:
            attempts = entries[rec["instance_id"]]["attempts"]
            assert len(attempts) == 5
            expected_correct = rec["ground_truth"] == "5"
            assert all(a["is_correct"] is expected_correct for a in attempts)
            assert all(a["extracted_answer"] == "5" for a in attempts)


class TestBenchmarkRunnerAndEvaluation:
    """End-to-end tests for run_benchmark, eval_only, errors, and CLI execution."""

    def test_run_benchmark_mock_csv_schema_and_determinism(
        self, tmp_path: Path
    ) -> None:
        methods = ("uniform", "easy2hard", "hard2easy", "dipa_mgl", "dipa_mgc")
        multipliers = (0.5, 1.0, 2.0)
        seeds = (42, 43)

        csv_a = tmp_path / "run_a.csv"
        csv_b = tmp_path / "run_b.csv"
        pool_a = tmp_path / "pool_a.jsonl"
        pool_b = tmp_path / "pool_b.jsonl"

        rows_a = run_benchmark(
            config_path=CONFIG_PATH,
            mode="mock",
            local_jsonl=FIXTURE_PATH,
            pool_size=4,
            gen_batch_size=2,
            pool_path=pool_a,
            methods=methods,
            budget_multipliers=multipliers,
            seeds=seeds,
            output_csv=csv_a,
            overwrite_csv=True,
            print_summary=False,
        )
        rows_b = run_benchmark(
            config_path=CONFIG_PATH,
            mode="mock",
            local_jsonl=FIXTURE_PATH,
            pool_size=4,
            gen_batch_size=2,
            pool_path=pool_b,
            methods=methods,
            budget_multipliers=multipliers,
            seeds=seeds,
            output_csv=csv_b,
            overwrite_csv=True,
            print_summary=False,
        )

        expected_count = len(methods) * len(multipliers) * len(seeds)
        assert len(rows_a) == expected_count
        assert len(rows_b) == expected_count

        with csv_a.open("r", encoding="utf-8", newline="") as handle:
            reader = list(csv.reader(handle))
        assert reader[0] == list(METRICS_COLUMNS)
        assert len(reader) == expected_count + 1

        with csv_a.open("r", encoding="utf-8", newline="") as handle:
            dict_rows_a = list(csv.DictReader(handle))
        with csv_b.open("r", encoding="utf-8", newline="") as handle:
            dict_rows_b = list(csv.DictReader(handle))

        for row in dict_rows_a:
            assert row["energy_j"] == "not_measured"
            assert row["status"] == "smoke_test"
            assert row["quality_metric"] == "coverage"
            cov = float(row["quality_value"])
            assert 0.0 <= cov <= 1.0
            assert row["peak_memory_mb"] == ""
            assert row["input_length"] == ""
            assert float(row["latency_median_ms"]) > 0.0
            assert float(row["throughput"]) > 0.0
            assert "budget_multiplier=" in row["notes"]
            assert "pool_size=4" in row["notes"]
            assert "lambda_mode=dynamic" in row["notes"]
            assert "lambda_value=5" in row["notes"]

        assert dict_rows_a == dict_rows_b

    def test_eval_only_uses_existing_pool_without_regeneration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pool_path = tmp_path / "existing_pool.jsonl"
        out_csv = tmp_path / "eval_only.csv"

        records = load_math500(Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH))
        build_response_pool(
            records=records,
            mode="mock",
            pool_size=4,
            pool_path=pool_path,
            seed=42,
        )
        pool_bytes_before = pool_path.read_bytes()

        def _forbid_loader(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("load_math500 should not run in eval_only mode")

        monkeypatch.setattr(
            "Adaptive_Stopping.experiments.run_dipa_benchmark.load_math500",
            _forbid_loader,
        )

        rows = run_benchmark(
            config_path=CONFIG_PATH,
            mode="mock",
            pool_path=pool_path,
            eval_only=True,
            methods=("uniform", "dipa_mgl"),
            budget_multipliers=(1.0,),
            seeds=(42,),
            output_csv=out_csv,
            overwrite_csv=True,
            print_summary=False,
        )
        assert len(rows) == 2
        assert all(r.status == "smoke_test" for r in rows)
        assert pool_path.read_bytes() == pool_bytes_before
        assert out_csv.is_file()

        with pytest.raises(FileNotFoundError):
            run_benchmark(
                config_path=CONFIG_PATH,
                pool_path=tmp_path / "missing_pool.jsonl",
                eval_only=True,
                output_csv=None,
                print_summary=False,
            )

    def test_unknown_method_raises_value_error(self, tmp_path: Path) -> None:
        records = load_math500(Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH))
        pool_entries, _ = build_response_pool(records=records, mode="mock", pool_size=2)
        with pytest.raises(ValueError, match="Unknown method"):
            evaluate_pool(
                pool_entries=pool_entries,
                methods=("uniform", "nonexistent_method"),
                budget_multipliers=(1.0,),
                seeds=(42,),
                config_dict={},
            )

        with pytest.raises(ValueError, match="Unknown method"):
            run_benchmark(
                config_path=CONFIG_PATH,
                mode="mock",
                local_jsonl=FIXTURE_PATH,
                methods=("bad_method",),
                output_csv=tmp_path / "unused.csv",
                print_summary=False,
            )

    def test_pool_wraparound_records_note(self) -> None:
        records = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=2, seed=42)
        )
        pool_entries, _ = build_response_pool(records=records, mode="mock", pool_size=1)
        for entry in pool_entries.values():
            entry["attempts"][0]["is_correct"] = False

        rows = evaluate_pool(
            pool_entries=pool_entries,
            methods=("uniform",),
            budget_multipliers=(2.0,),
            seeds=(42,),
            config_dict={},
        )
        assert len(rows) == 1
        assert "pool_wrapped=" in rows[0].notes

    def test_cli_subprocess_mock_mode_and_import_isolation(
        self, tmp_path: Path
    ) -> None:
        pool_path = tmp_path / "cli_pool.jsonl"
        out_csv = tmp_path / "cli_out.csv"

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "Adaptive_Stopping.experiments.run_dipa_benchmark",
                "--mode",
                "mock",
                "--local-jsonl",
                str(FIXTURE_PATH),
                "--pool-size",
                "4",
                "--gen-batch-size",
                "2",
                "--pool-path",
                str(pool_path),
                "--budget-multipliers",
                "0.5,1",
                "--seeds",
                "42,43",
                "--output-csv",
                str(out_csv),
                "--overwrite-csv",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        assert "Method" in proc.stdout
        assert "dipa_mgl" in proc.stdout
        assert "b=0.5" in proc.stdout
        assert out_csv.is_file()

        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 5 * 2 * 2

        isolation_script = (
            "import sys\n"
            "import Adaptive_Stopping.experiments.run_dipa_benchmark\n"
            "for mod in ('torch', 'transformers', 'datasets'):\n"
            "    assert mod not in sys.modules, f'{mod} imported at module level'\n"
        )
        subprocess.run(
            [sys.executable, "-c", isolation_script],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )

    def test_dipa_mgc_uses_const_input_prior_and_explores_long_prompts(self) -> None:
        alloc = _build_allocator(
            "dipa_mgc",
            AllocationConfig(total_budget=20, input_proxy="mql", gen_proxy="mgl", seed=42),
        )
        assert alloc.config.input_proxy == "const"
        assert alloc.config.gen_proxy == "mgc"

        long_records = [
            {
                "instance_id": f"long_{i}",
                "prompt": " ".join(f"word{k}" for k in range(50)),
                "ground_truth": "42",
                "metadata": {"level": 3},
            }
            for i in range(20)
        ]
        pool_entries, _ = build_response_pool(
            records=long_records, mode="mock", pool_size=4, seed=42
        )
        # Force the first 2 attempts on every problem to fail with distinct answers so
        # an unfixed mql+mgc scale mismatch would lock onto a single arm.
        for i, entry in enumerate(pool_entries.values()):
            entry["attempts"][0]["is_correct"] = False
            entry["attempts"][0]["extracted_answer"] = f"wrong_a_{i}"
            entry["attempts"][1]["is_correct"] = False
            entry["attempts"][1]["extracted_answer"] = f"wrong_b_{i}"
            entry["attempts"][2]["is_correct"] = True
            entry["attempts"][2]["extracted_answer"] = "42"

        rows = evaluate_pool(
            pool_entries=pool_entries,
            methods=("dipa_mgc",),
            budget_multipliers=(1.0,),
            seeds=(42,),
            config_dict={"allocation": {"input_proxy": "mql"}},
        )
        assert len(rows) == 1
        assert "pool_wrapped=" not in rows[0].notes

    def test_lambda_overrides_propagate_to_notes_and_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_csv = tmp_path / "override.csv"
        pool_path = tmp_path / "override_pool.jsonl"
        shared_cfg = {
            "backbone_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "dataset_id": "HuggingFaceH4/MATH-500",
            "split": "test",
            "allocation": {
                "input_proxy": "mql",
                "gen_proxy": "mgl",
                "lambda_mode": "dynamic",
                "lambda_value": 5.0,
                "min_difficulty": 1e-6,
                "seeds": [42],
                "budget_multipliers": [1.0],
            },
        }
        monkeypatch.setattr(
            "Adaptive_Stopping.experiments.run_dipa_benchmark._load_yaml_config",
            lambda _path: shared_cfg,
        )

        rows = run_benchmark(
            config_path=CONFIG_PATH,
            mode="mock",
            local_jsonl=FIXTURE_PATH,
            pool_size=4,
            gen_batch_size=2,
            pool_path=pool_path,
            methods=("uniform", "dipa_mgl"),
            budget_multipliers=(1.0,),
            seeds=(42,),
            lambda_mode="fixed",
            lambda_value=2.5,
            output_csv=out_csv,
            overwrite_csv=True,
            print_summary=False,
        )
        assert len(rows) == 2
        for row in rows:
            assert "lambda_mode=fixed" in row.notes
            assert "lambda_value=2.5" in row.notes
        assert shared_cfg["allocation"]["lambda_mode"] == "dynamic"
        assert shared_cfg["allocation"]["lambda_value"] == 5.0

        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) == 2
        for csv_row in csv_rows:
            assert "lambda_mode=fixed" in csv_row["notes"]
            assert "lambda_value=2.5" in csv_row["notes"]

        for bad_kwargs in (
            {"lambda_mode": "invalid"},
            {"lambda_value": 0.0},
            {"lambda_value": -1.0},
        ):
            with pytest.raises(ValueError):
                run_benchmark(
                    config_path=CONFIG_PATH,
                    mode="mock",
                    local_jsonl=FIXTURE_PATH,
                    pool_path=pool_path,
                    output_csv=None,
                    print_summary=False,
                    **bad_kwargs,
                )

        cli_csv = tmp_path / "cli_override.csv"
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "Adaptive_Stopping.experiments.run_dipa_benchmark",
                "--mode",
                "mock",
                "--local-jsonl",
                str(FIXTURE_PATH),
                "--pool-size",
                "4",
                "--gen-batch-size",
                "2",
                "--pool-path",
                str(pool_path),
                "--methods",
                "uniform,dipa_mgl",
                "--budget-multipliers",
                "1",
                "--seeds",
                "42",
                "--lambda-mode",
                "fixed",
                "--lambda-value",
                "1.5",
                "--output-csv",
                str(cli_csv),
                "--overwrite-csv",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        with cli_csv.open("r", encoding="utf-8", newline="") as handle:
            cli_rows = list(csv.DictReader(handle))
        assert len(cli_rows) == 2
        for cli_row in cli_rows:
            assert "lambda_mode=fixed" in cli_row["notes"]
            assert "lambda_value=1.5" in cli_row["notes"]
