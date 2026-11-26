"""Configs, quality metric and an end-to-end benchmark run on the mock backend."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from ..config import build_moa, build_staircase, load_config
from ..quality import exact_match, extract_prediction, extract_reference, format_gsm8k
from ..run_benchmark import METRICS_COLUMNS, build_variants, main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "kaggle_2xT4.yaml", "mock.yaml"])
def test_configs_are_valid(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    moa = build_moa(cfg, seed=1)
    build_staircase(cfg).validate(len(moa.proposers))
    variants = build_variants(cfg)
    assert variants[0].id == "baseline"
    assert len(variants) == 1 + 4 * 3 + 1  # baseline + sweep + prefix-cache variant
    if cfg["backend"]["type"] == "openai_compat":
        servers = cfg["servers"]
        assert set(cfg["moa"]["proposers"]) | {cfg["moa"]["aggregator"]} <= set(servers)
        assert all(len(s["revision"]) == 40 for s in servers.values())
        assert len({s["port"] for s in servers.values()}) == len(servers)
        per_gpu: dict[str, float] = {}
        for s in servers.values():
            per_gpu[s["gpu"]] = per_gpu.get(s["gpu"], 0) + s["gpu_memory_utilization"]
        assert all(total <= 0.95 for total in per_gpu.values())


def test_config_models_match_backbone_registry() -> None:
    from .conftest import ROOT

    reg = yaml.safe_load((ROOT / "Backbones/Generative_Pretrained/registry.yaml").read_text())["backbones"]
    for name in ("default.yaml", "kaggle_2xT4.yaml"):
        for srv in load_config(CONFIGS / name)["servers"].values():
            entry = reg[srv["backbone_id"]]
            assert entry["revision"] == srv["revision"]
            assert entry["weights_uri"].endswith(srv["hf_model"])


def test_staircase_defaults_match_paper_appendix_a() -> None:
    sc = build_staircase(load_config(CONFIGS / "default.yaml"))
    assert (sc.first_chunk_size, sc.second_chunk_size, sc.chunk_size,
            sc.aggregator_chunk_size, sc.redundancy) == (8, 128, 256, 128, 2)


def test_gsm8k_extraction_and_exact_match() -> None:
    ref = "Janet sells 16 - 3 - 4 = 9 eggs.\n#### 1,018"
    assert extract_reference(ref) == "1018"
    assert extract_prediction("so the total is 1018.0\n#### 1,018.") == "1018"
    assert extract_prediction("The answer is $42.") == "42"
    assert extract_prediction("no digits here") is None
    assert exact_match("... #### 1018", ref) and not exact_match("#### 1017", ref)
    assert format_gsm8k("Q?").endswith("Q?") and "####" in format_gsm8k("Q?")


def test_end_to_end_benchmark_on_mock(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "mock.yaml")
    fast = {"base_latency_s": 0.002, "prefill_s_per_token": 0.000005,
            "decode_s_per_token": 0.0002, "answer_tokens": 150}
    cfg["mock_models"] = {k: fast for k in cfg["mock_models"]}
    cfg["experiment"].update(output_dir=str(tmp_path / "out"), warmup_requests=1)
    cfg["benchmark"].update(prompt_sets=[{"dataset_id": "synthetic.prompts", "num_samples": 2}],
                            sweep={"first_chunk_size": [8], "redundancy": [1]})
    cfg["quality"].update(num_samples=2)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = main(["--config", str(path)])
    out = tmp_path / "out"
    with open(out / "metrics.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == METRICS_COLUMNS
        rows = list(reader)
    methods = sorted(r["method"] for r in rows)
    # latency: baseline, staircase_f8_r1, prefix-cache; quality: baseline + default staircase (f8_r2)
    assert methods == sorted(["baseline_moa", "staircase_f8_r1", "staircase_pc_f8_r2",
                              "baseline_moa", "staircase_f8_r2"])
    for r in rows:
        assert r["energy_j"] == "" and r["peak_memory_mb"] == ""  # not measured -> empty, never 0
        assert float(r["latency_median_ms"]) > 0 and r["status"] == "mock_dry_run"
    lat = {r["method"]: float(r["latency_median_ms"]) for r in rows if r["quality_metric"] == ""}
    assert lat["staircase_f8_r1"] < lat["baseline_moa"]
    raw = [json.loads(line) for line in (out / "raw_requests.jsonl").read_text().splitlines()]
    assert len(raw) == 3 * 2 + 2 * 2 and all(r["status"] == "ok" for r in raw)
    assert (out / "ttft_vs_first_chunk.png").exists() and (out / "summary.json").exists()
    assert summary["summaries"]
