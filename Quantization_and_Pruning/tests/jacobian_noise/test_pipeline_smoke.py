"""End-to-end CPU smoke test of the training and evaluation scripts."""

import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CFG = os.path.join(ROOT, "Quantization_and_Pruning", "configs", "jacobian_noise")
sys.path.insert(0, os.path.join(ROOT, "Quantization_and_Pruning", "experiments", "jacobian_noise"))


@pytest.mark.parametrize("mode", ["none", "jacobian", "uniform_matched"])
def test_lm_train_then_quantized_eval(tmp_path, mode):
    import eval_quant
    import train_lm

    out = train_lm.main(["--config", os.path.join(CFG, "smoke_lm.yaml"), "--set",
                         f"experiment.output_dir={tmp_path}", f"noise.mode={mode}"])
    run = json.load(open(os.path.join(out, "run.json")))
    assert run["config"]["noise"]["mode"] == mode
    if mode != "none":
        assert run["noise_history"], "sigma was never updated"
    eval_quant.main(["--config", os.path.join(CFG, "smoke_eval_lm.yaml"), "--set",
                     f"experiment.output_dir={tmp_path}/eval", f"model.path={out}/checkpoint",
                     "quant.weight_method=gptq"])
    rows = list(csv.DictReader(open(tmp_path / "eval" / "results.csv")))
    assert [r["method"].split(" | ")[1] for r in rows] == ["FP16", "gptq-W8A8", "gptq-W4A4", "gptq-W4A4QK4"]
    assert all(r["status"] == "smoke-test" and float(r["quality_value"]) > 0 for r in rows)


def test_siglip_train_then_quantized_eval(tmp_path):
    import eval_quant
    import train_siglip

    out = train_siglip.main(["--config", os.path.join(CFG, "smoke_siglip.yaml"), "--set",
                             f"experiment.output_dir={tmp_path}"])
    eval_quant.main(["--config", os.path.join(CFG, "smoke_eval_siglip.yaml"), "--set",
                     f"experiment.output_dir={tmp_path}/eval", f"model.path={out}/checkpoint"])
    rows = list(csv.DictReader(open(tmp_path / "eval" / "results.csv")))
    assert len(rows) == 4 and all(0 <= float(r["quality_value"]) <= 1 for r in rows)
    diag = json.loads(open(tmp_path / "eval" / "jacobian_diagnostics.jsonl").readline())
    assert 0 < diag["mean_jacobian_fro_sq"] <= 0.25
