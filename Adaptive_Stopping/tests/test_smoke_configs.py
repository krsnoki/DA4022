"""Smoke configs: same experiment as the main configs on smaller subsets, and they can never touch the main results.

Two hazards found while reproducing the pipeline (docs/D_reproduction_and_review.md):
a smoke run must exercise exactly the main settings, and its report must not overwrite
the main result folder or add rows to the metrics CSV.
"""

from pathlib import Path

import pytest
import yaml

from Adaptive_Stopping.experiments.st_es import run_all

CONFIGS = Path("Adaptive_Stopping/configs/st_es")
PAIRS = [("gsm8k_smoke.yaml", "gsm8k.yaml"), ("math500_smoke.yaml", "math500.yaml")]
# The only fields a smoke config may change.
SIZE_FIELDS = {("calibration", "num_samples"), ("evaluation", "num_samples"), ("live", "num_samples")}
SMOKE_ONLY = {"report_name", "write_metrics"}


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def flatten(cfg: dict, prefix: tuple = ()) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and k != "sampling":
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


@pytest.mark.parametrize("smoke,main", PAIRS, ids=[p[0] for p in PAIRS])
def test_smoke_config_differs_from_main_only_in_subset_sizes(smoke, main):
    s, m = flatten(load(smoke)), flatten(load(main))
    changed = {k for k in s.keys() | m.keys() if s.get(k) != m.get(k)}
    assert changed - {(k,) for k in SMOKE_ONLY} <= SIZE_FIELDS, changed
    for k in SIZE_FIELDS:
        assert 0 < s[k] < m[k], k


@pytest.mark.parametrize("smoke,main", PAIRS, ids=[p[0] for p in PAIRS])
def test_smoke_report_cannot_overwrite_main_results(smoke, main):
    s, m = load(smoke), load(main)
    assert s["write_metrics"] is False
    main_folders = {load(n).get("report_name", load(n)["evaluation"]["dataset_id"]) for _, n in PAIRS}
    assert s["report_name"] not in main_folders
    assert m.get("write_metrics", True) is True and "report_name" not in m


@pytest.mark.parametrize("smoke", [p[0] for p in PAIRS])
def test_smoke_run_passes_the_run_dir_to_every_stage(smoke, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_all, "run", lambda mod, *args: calls.append((mod, [str(a) for a in args])))
    run_all.main(["--config", str(CONFIGS / smoke), "--run-dir", str(tmp_path)])
    assert [m for m, _ in calls] == ["generate", "generate", "replay_sweep", "replay_sweep", "live_benchmark"]
    for mod, args in calls:
        assert args[args.index("--run-dir") + 1] == str(tmp_path), mod
