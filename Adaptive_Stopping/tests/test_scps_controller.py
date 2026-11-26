"""CPU smoke tests for the SCPS controller, its stopping policy and the baselines."""

import dataclasses

import pytest

from Adaptive_Stopping.baselines.self_consistency import SelfConsistency
from Adaptive_Stopping.baselines.tree_prefix import TreePrefix, TreePrefixConfig, cosine
from Adaptive_Stopping.common.budget import Budget, Ledger
from Adaptive_Stopping.common.policy import NeverStop
from Adaptive_Stopping.methods.budget_policy.scps.controller import SCPSConfig, SCPSController, SCPSStoppingPolicy
from Backbones.Generative_Pretrained.mock_mcq_lm import MockMCQBackbone, NoScoreMockBackbone
from Dataloaders.synthetic.mock_mcq import make_synthetic_mcq

PAPER_BUDGET = Budget(16, 192, 3072, 30000)


@pytest.fixture(scope="module")
def items():
    return make_synthetic_mcq(24, seed=3)


def run_all(method, items):
    return [method.run(it) for it in items]


def check_budget(res, budget):
    led = res.ledger
    assert led["rollouts"] <= budget.max_rollouts
    assert led["decode_tokens"] <= budget.decode_token_limit
    assert led["total_tokens"] <= budget.total_token_limit
    assert res.metadata["violations"] == []


def test_scps_respects_paper_budget_and_returns_metadata(items):
    ctrl = SCPSController(MockMCQBackbone(items), PAPER_BUDGET, SCPSConfig())
    for res in run_all(ctrl, items):
        check_budget(res, PAPER_BUDGET)
        assert res.method == "scps"
        assert res.stop_reason in {"no_positive_action", "answer_class_quota", "max_rounds", "budget_exhausted"}
        assert res.rounds == len(res.trace) <= SCPSConfig().max_rounds
        assert res.unique_answers() <= SCPSConfig().answer_class_quota + max(SCPSConfig().sample_sizes)
        assert res.metadata["telemetry"] == "option_scores"
        assert sum(c.num_tokens for c in res.completions) <= res.ledger["decode_tokens"]


def test_scps_is_deterministic(items):
    a = run_all(SCPSController(MockMCQBackbone(items), PAPER_BUDGET, SCPSConfig(), run_seed=5), items[:6])
    b = run_all(SCPSController(MockMCQBackbone(items), PAPER_BUDGET, SCPSConfig(), run_seed=5), items[:6])
    assert [r.answers for r in a] == [r.answers for r in b]
    assert [r.ledger for r in a] == [r.ledger for r in b]


@pytest.mark.parametrize("budget", [Budget(0, 192, 3072, 30000), Budget(16, 0, 3072, 30000), Budget(16, 192, 0, 30000)])
def test_zero_budget_gives_empty_portfolio(items, budget):
    for method in (SCPSController(MockMCQBackbone(items), budget), SelfConsistency(MockMCQBackbone(items), budget),
                   TreePrefix(MockMCQBackbone(items), budget)):
        res = method.run(items[0])
        assert res.completions == []
        assert res.ledger["rollouts"] == 0 and res.ledger["decode_tokens"] == 0
        assert res.metadata["violations"] == []


def test_total_limit_below_scoring_cost_falls_back(items):
    budget = Budget(16, 192, 3072, 0)
    res = SCPSController(MockMCQBackbone(items), budget).run(items[0])
    assert res.ledger["aux_tokens"] == 0 and res.completions == []


def test_minimum_budget_one_rollout(items):
    budget = Budget(1, 192, 192, 30000)
    for method in (SCPSController(MockMCQBackbone(items), budget), SelfConsistency(MockMCQBackbone(items), budget),
                   TreePrefix(MockMCQBackbone(items), budget)):
        res = method.run(items[1])
        assert res.ledger["rollouts"] <= 1 and len(res.completions) <= 1
        check_budget(res, budget)


def test_missing_option_scores_telemetry(items):
    res = SCPSController(NoScoreMockBackbone(items), PAPER_BUDGET).run(items[2])
    assert res.metadata["telemetry"] == "no_option_scores"
    assert res.ledger["aux_tokens"] == 0
    assert len(res.completions) > 0
    check_budget(res, PAPER_BUDGET)


def test_answer_class_quota_stops(items):
    cfg = SCPSConfig(answer_class_quota=1, sample_sizes=(1,))
    res = SCPSController(MockMCQBackbone(items), PAPER_BUDGET, cfg).run(items[3])
    assert res.unique_answers() <= 1
    assert res.stop_reason in {"answer_class_quota", "no_positive_action", "budget_exhausted"}


def test_horizon_creates_frontier_prefixes_and_respects_budget(items):
    cfg = SCPSConfig(horizon=8)
    ctrl = SCPSController(MockMCQBackbone(items), PAPER_BUDGET, cfg)
    results = run_all(ctrl, items[:8])
    assert any(a["source"].startswith("prefix:") for r in results for t in r.trace for a in t["actions"])
    for r in results:
        check_budget(r, PAPER_BUDGET)


def test_self_consistency_uses_full_budget(items):
    res = SelfConsistency(MockMCQBackbone(items), PAPER_BUDGET).run(items[4])
    assert res.ledger["rollouts"] == 16 and len(res.completions) == 16


@pytest.mark.parametrize("pruning", [False, True])
def test_tree_prefix_budget_and_names(items, pruning):
    cfg = TreePrefixConfig(horizon=8, semantic_pruning=pruning)
    m = TreePrefix(MockMCQBackbone(items), PAPER_BUDGET, cfg)
    assert m.method_name == ("semantic_pruning" if pruning else "tree_prefix")
    for res in run_all(m, items[:6]):
        check_budget(res, PAPER_BUDGET)


def test_cosine_overlap():
    assert cosine({"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 0.0}) == pytest.approx(1.0)
    assert cosine({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}) == pytest.approx(0.0)


def test_stopping_policy_contract():
    pol = SCPSStoppingPolicy(max_rounds=3, answer_class_quota=2)
    pol.reset()
    assert not pol.should_stop({"round": 0, "n_seen": 0, "best_score": 0.1}, {}, PAPER_BUDGET)
    assert pol.should_stop({"round": 0, "n_seen": 0, "best_score": 0.0}, {}, PAPER_BUDGET)
    assert pol.report()["stop_reason"] == "no_positive_action"
    assert pol.should_stop({"round": 3}, {}, PAPER_BUDGET) and pol.reason == "max_rounds"
    assert pol.should_stop({"round": 0, "n_seen": 2}, {}, PAPER_BUDGET) and pol.reason == "answer_class_quota"
    assert pol.should_stop({"round": 0, "feasible": False}, {}, PAPER_BUDGET) and pol.reason == "budget_exhausted"
    never = NeverStop()
    never.reset()
    assert not never.should_stop({}, {}, None)


def test_ledger_worst_case_feasibility():
    led = Ledger(Budget(4, 10, 25, 100))
    assert led.max_affordable_samples() == 2  # 3 * 10 > 25
    assert led.can_sample(1, 5, new_rollouts=0)
    assert not led.can_sample(0)


@pytest.mark.parametrize("bad", [dict(temperature=-1), dict(sample_sizes=()), dict(sample_sizes=(0,)),
                                 dict(posterior_mix=1.5), dict(horizon=0), dict(overlap_penalty=-0.1)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        SCPSConfig(**bad)


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SCPSConfig.from_dict({"temprature": 0.6})
    cfg = SCPSConfig.from_dict({"sample_sizes": [1, 2]})
    assert cfg.sample_sizes == (1, 2)
    assert dataclasses.replace(cfg, max_rounds=3).max_rounds == 3


def test_option_topk_control(items):
    from Adaptive_Stopping.baselines.option_topk import OptionTopK, OptionTopKConfig

    res = OptionTopK(MockMCQBackbone(items), PAPER_BUDGET, OptionTopKConfig(k=3)).run(items[0])
    probs = res.metadata["record_extra"]["root_probs"]
    assert len(res.completions) == 3 and res.ledger["rollouts"] == 0 and res.ledger["decode_tokens"] == 0
    top = sorted(probs, key=lambda a: -probs[a])[:3]
    assert res.answers == top
    none = OptionTopK(NoScoreMockBackbone(items), PAPER_BUDGET).run(items[0])
    assert none.completions == [] and none.stop_reason == "no_option_scores"


def test_run_script_separate_invocations_and_summary(tmp_path):
    import json

    from Adaptive_Stopping.experiments.scps_mmlupro import run

    cfg = "Adaptive_Stopping/configs/scps_synthetic_smoke.yaml"
    run.main(["--config", cfg, "--out-dir", str(tmp_path), "--limit", "6", "--methods", "scps,option_topk"])
    run.main(["--config", cfg, "--out-dir", str(tmp_path), "--limit", "6", "--methods", "self_consistency", "--resume"])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert set(s["methods"]) == {"scps", "option_topk", "self_consistency"}
    assert s["paired_vs_scps"]["self_consistency"]["n"] == 6
    assert s["option_topk_control"]["n"] == 6 and 0 <= s["option_topk_control"]["matched_k_pass"] <= 1
    assert (tmp_path / "run_meta_scps.json").exists()
    # resume keeps finished questions and does not duplicate them
    run.main(["--config", cfg, "--out-dir", str(tmp_path), "--limit", "6", "--methods", "scps", "--resume"])
    assert len((tmp_path / "per_question_scps.jsonl").read_text().splitlines()) == 6
    lines = (tmp_path / "results.csv").read_text().splitlines()
    assert len(lines) == 1 + 3


def test_report_renders_from_summary(tmp_path, capsys):
    from Adaptive_Stopping.experiments.scps_mmlupro import report, run

    cfg = "Adaptive_Stopping/configs/scps_synthetic_smoke.yaml"
    run.main(["--config", cfg, "--out-dir", str(tmp_path), "--limit", "5",
              "--methods", "scps,self_consistency,option_topk"])
    capsys.readouterr()
    report.main([str(tmp_path)])
    text = capsys.readouterr().out
    assert "| SCPS |" in text and "Paired bootstrap" in text and "Gold in top-K" in text


def test_reported_run_summary_matches_per_question_files():
    """The committed summary must be reproducible from the committed per-question files."""
    import json
    from pathlib import Path

    from Adaptive_Stopping.experiments.scps_mmlupro import run

    d = Path("Adaptive_Stopping/experiments/scps_mmlupro/reported_runs/qwen35_4b_bnb4_rtx3050")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    for m, v in s["methods"].items():
        recs = [json.loads(x) for x in (d / f"per_question_{m}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        assert run.method_summary(recs)["pass_count"] == v["pass_count"]
        assert len(recs) == 256 and v["budget_violations"] == 0
