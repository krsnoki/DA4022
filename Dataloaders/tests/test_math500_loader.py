"""Offline tests for MATH-500 dataloader, registry metadata, and isolation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from Dataloaders.evaluation.math500 import (
    DEFAULT_SYSTEM_PROMPT,
    Math500LoaderConfig,
    Math500Record,
    format_math500_messages,
    get_dataset_metadata,
    load_math500,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "Dataloaders" / "registry.yaml"
FIXTURE_PATH = (
    REPO_ROOT / "Dataloaders" / "evaluation" / "math500" / "sample_fixture.jsonl"
)


class TestRegistryAndMetadata:
    """Tests for ``Dataloaders/registry.yaml`` and ``get_dataset_metadata``."""

    def test_registry_yaml_schema_matches_metadata(self) -> None:
        with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle)

        assert isinstance(registry, dict)
        assert "datasets" in registry
        assert isinstance(registry["datasets"], list)

        matches = [
            entry
            for entry in registry["datasets"]
            if isinstance(entry, dict) and entry.get("id") == "math.math500.test"
        ]
        assert len(matches) == 1
        entry = matches[0]

        expected = get_dataset_metadata()
        assert entry == expected
        assert entry["name"] == "MATH-500 evaluation test split"
        assert entry["modality"] == "text"
        assert entry["task"] == "mathematical_reasoning"
        assert entry["source_url"] == "https://huggingface.co/datasets/HuggingFaceH4/MATH-500"
        assert entry["license"] == "mit"
        assert entry["role"] == "evaluation"
        assert entry["split"] == "test"
        assert entry["preprocessing_id"] == "dipa.math_cot_boxed.v1"
        assert entry["seed"] == 42
        assert entry["num_examples"] == 500
        assert entry["labels_available"] is True


class TestMath500ConfigAndFormatting:
    """Validation, freezing, and Appendix D.2 message formatting tests."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"split": "train"},
            {"split": "validation"},
            {"split": "dev"},
            {"num_samples": -1},
            {"num_samples": -100},
            {"num_samples": True},
        ],
    )
    def test_invalid_config_raises_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            Math500LoaderConfig(**kwargs)

    def test_config_is_frozen(self) -> None:
        cfg = Math500LoaderConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.split = "train"  # type: ignore[misc]

    def test_format_math500_messages_default_and_custom(self) -> None:
        problem = "Compute $2 + 2$."
        default_msgs = format_math500_messages(problem)
        assert default_msgs == [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        assert "\\boxed{}" in DEFAULT_SYSTEM_PROMPT

        custom_msgs = format_math500_messages(problem, system_prompt="Custom prompt")
        assert custom_msgs == [
            {"role": "system", "content": "Custom prompt"},
            {"role": "user", "content": problem},
        ]

    def test_math500_record_fields(self) -> None:
        rec = Math500Record(
            unique_id="id1",
            problem="1+1?",
            answer="2",
            solution="1+1=2",
            subject="Algebra",
            level=1,
        )
        assert rec.unique_id == "id1"
        assert rec.level == 1


class TestLoadMath500OfflineAndSubsampling:
    """Offline JSONL fixture loading, subsampling, and edge-case tests."""

    def test_load_sample_fixture_exact_schema_and_keys(self) -> None:
        items = load_math500(Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH))
        assert load_math500(Math500LoaderConfig(local_jsonl_path=str(FIXTURE_PATH))) == items
        assert len(items) == 4

        expected_top_keys = {
            "instance_id",
            "prompt",
            "messages",
            "ground_truth",
            "metadata",
        }
        expected_meta_keys = {"subject", "level", "solution", "split", "dataset_id"}

        seen_ids: set[str] = set()
        for item in items:
            assert set(item.keys()) == expected_top_keys
            assert isinstance(item["instance_id"], str) and item["instance_id"]
            assert isinstance(item["prompt"], str) and item["prompt"]
            assert isinstance(item["ground_truth"], str) and item["ground_truth"]
            assert item["messages"] == format_math500_messages(item["prompt"])

            meta = item["metadata"]
            assert isinstance(meta, dict)
            assert set(meta.keys()) == expected_meta_keys
            assert isinstance(meta["subject"], str) and meta["subject"]
            assert isinstance(meta["level"], int)
            assert isinstance(meta["solution"], str) and meta["solution"]
            assert meta["split"] == "test"
            assert meta["dataset_id"] == "HuggingFaceH4/MATH-500"
            seen_ids.add(item["instance_id"])

        assert len(seen_ids) == 4

    def test_deterministic_seeded_subsampling(self) -> None:
        run_a = load_math500(
            Math500LoaderConfig(
                local_jsonl_path=FIXTURE_PATH, num_samples=2, seed=42
            )
        )
        run_b = load_math500(
            Math500LoaderConfig(
                local_jsonl_path=FIXTURE_PATH, num_samples=2, seed=42
            )
        )
        assert run_a == run_b
        assert len(run_a) == 2

        sampled_orders = {
            tuple(
                item["instance_id"]
                for item in load_math500(
                    Math500LoaderConfig(
                        local_jsonl_path=FIXTURE_PATH, num_samples=2, seed=seed
                    )
                )
            )
            for seed in (42, 43, 44, 45)
        }
        assert len(sampled_orders) > 1

    def test_num_samples_zero_and_greater_than_len(self) -> None:
        empty = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=0)
        )
        assert empty == []

        exact = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=4)
        )
        oversized = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=100)
        )
        full = load_math500(
            Math500LoaderConfig(local_jsonl_path=FIXTURE_PATH, num_samples=None)
        )
        assert exact == full
        assert oversized == full
        assert len(oversized) == 4

    def test_missing_local_jsonl_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        with pytest.raises(FileNotFoundError):
            load_math500(Math500LoaderConfig(local_jsonl_path=missing))

        with pytest.raises(FileNotFoundError):
            load_math500(Math500LoaderConfig(local_jsonl_path=tmp_path))

    def test_missing_datasets_raises_runtime_error_when_local_path_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(RuntimeError, match="datasets"):
            load_math500(Math500LoaderConfig())

    def test_huggingface_path_lazy_imports_and_subsamples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []
        fake_rows = [
            {
                "unique_id": f"hf/{idx}",
                "problem": f"Compute {idx} + 1.",
                "answer": str(idx + 1),
                "solution": f"{idx} + 1 = {idx + 1}",
                "subject": "Algebra",
                "level": 1,
            }
            for idx in range(4)
        ]

        def fake_load_dataset(
            dataset_id: str, split: str, revision: str
        ) -> list[dict[str, Any]]:
            calls.append((dataset_id, split, revision))
            return fake_rows

        monkeypatch.setitem(
            sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
        )
        items = load_math500(Math500LoaderConfig(num_samples=2, seed=42))
        assert calls == [("HuggingFaceH4/MATH-500", "test", "main")]
        assert len(items) == 2
        assert items[0]["metadata"]["split"] == "test"


class TestImportIsolation:
    """Clean-subprocess tests verifying zero coupling and lazy imports."""

    def test_importing_loader_does_not_import_datasets_or_peer_packages(self) -> None:
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pp}"
            if existing_pp
            else str(REPO_ROOT)
        )
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "from Dataloaders.evaluation.math500 import (\n"
            "    Math500LoaderConfig,\n"
            "    get_dataset_metadata,\n"
            "    load_math500,\n"
            ")\n"
            "items = load_math500(\n"
            "    Math500LoaderConfig(local_jsonl_path=Path('Dataloaders/evaluation/math500/sample_fixture.jsonl'))\n"
            ")\n"
            "assert len(items) == 4\n"
            "assert get_dataset_metadata()['id'] == 'math.math500.test'\n"
            "forbidden = ('datasets', 'torch', 'Backbones', 'Adaptive_Stopping')\n"
            "loaded = [m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden)]\n"
            "assert not loaded, f'Unexpected forbidden imports: {loaded}'\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )
