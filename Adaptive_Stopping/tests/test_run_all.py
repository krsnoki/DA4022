"""run_all wiring: every stage receives the same --run-dir, and the configs pass validation (no GPU, nothing run)."""

from pathlib import Path

import pytest
import yaml

from Adaptive_Stopping.experiments.st_es import run_all

CONFIGS = sorted(Path("Adaptive_Stopping/configs/st_es").glob("*.yaml"))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_configs_pass_validation(config):
    run_all.check_config(yaml.safe_load(config.read_text(encoding="utf-8")))


def test_every_stage_gets_the_run_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_all, "run", lambda mod, *args: calls.append((mod, [str(a) for a in args])))
    run_all.main(["--config", "Adaptive_Stopping/configs/st_es/gsm8k.yaml", "--run-dir", str(tmp_path)])
    assert [m for m, _ in calls] == ["generate", "generate", "replay_sweep", "replay_sweep", "live_benchmark"]
    for mod, args in calls:
        assert "--run-dir" in args and args[args.index("--run-dir") + 1] == str(tmp_path), mod
    selected = [a for _, args in calls for a in args if a.endswith("selected.json")]
    assert selected and all(a.startswith(str(tmp_path)) for a in selected)


def test_committed_outputs_record_no_absolute_paths(tmp_path):
    from Adaptive_Stopping.experiments.st_es.io_utils import run_relative

    inside = tmp_path / "replay" / "ds" / "grid" / "selected.json"
    assert run_relative(inside, tmp_path) == "<run_dir>/replay/ds/grid/selected.json"
    assert run_relative(tmp_path.parent / "elsewhere" / "selected.json", tmp_path) == "selected.json"
