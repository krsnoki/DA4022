"""Tests for the registry-based text-prompt loader (offline: synthetic sets + temp files)."""

from __future__ import annotations

import hashlib
import json

import pytest

from Dataloaders.evaluation import text_prompts as tp

REQUIRED = {"name", "modality", "task", "license", "version", "role", "split", "num_examples",
            "preprocessing_id", "seed", "labels_available"}


def test_registry_entries_have_required_fields() -> None:
    reg = tp.load_registry()
    for ds_id, entry in reg.items():
        assert REQUIRED <= set(entry), ds_id
        assert ("download" in entry) != ("generator" in entry), ds_id
        if "download" in entry:
            assert len(entry["download"]["sha256"]) == 64 and entry["download"]["url"].startswith("https://")


def test_sampling_is_deterministic_and_seed_dependent() -> None:
    a = tp.load_prompts("synthetic.prompts", 8, seed=42)
    b = tp.load_prompts("synthetic.prompts", 8, seed=42)
    c = tp.load_prompts("synthetic.prompts", 8, seed=7)
    assert [r.id for r in a] == [r.id for r in b]
    assert [r.id for r in a] != [r.id for r in c]
    assert len({r.id for r in a}) == 8


def test_synthetic_arithmetic_has_references() -> None:
    recs = tp.load_prompts("synthetic.arithmetic", 3, seed=1)
    assert all(r.reference and "####" in r.reference for r in recs)


def test_fetch_verifies_sha256(tmp_path, monkeypatch) -> None:
    rows = [{"q": "one?", "a": "1"}, {"q": "two?", "a": "2"}]
    src = tmp_path / "data.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    entry = {"version": "localtest0000", "num_examples": 2,
             "download": {"url": src.as_uri(), "sha256": good, "format": "jsonl",
                          "prompt_field": "q", "id_field": None, "reference_field": "a"}}
    monkeypatch.setenv("DA4022_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(tp, "get_entry", lambda _id: entry)
    recs = tp.load_all("local.test")
    assert [(r.id, r.prompt, r.reference) for r in recs] == [("0", "one?", "1"), ("1", "two?", "2")]

    entry["download"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        tp.load_all("local.test")


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        tp.get_entry("no.such.dataset")
