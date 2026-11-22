"""Unit and integration tests for DIPA allocator, baselines, config, and StoppingPolicy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from Adaptive_Stopping.baselines.allocators import (
    Easy2HardAllocator,
    Hard2EasyAllocator,
    UniformAllocator,
)
from Adaptive_Stopping.methods.dipa.allocator import (
    BudgetAllocator,
    DIPABudgetAllocator,
)
from Adaptive_Stopping.methods.dipa.types import (
    AllocationConfig,
    ArmState,
    GenerationAttempt,
    ProblemInstance,
    StoppingPolicy,
)


class TestAllocationConfigAndYaml:
    """Validation tests for ``AllocationConfig`` and ``configs/dipa.yaml``."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"total_budget": -1},
            {"total_budget": 10, "lambda_mode": "invalid"},
            {"total_budget": 10, "lambda_value": 0},
            {"total_budget": 10, "lambda_value": -1.0},
            {"total_budget": 10, "lambda_value": float("nan")},
            {"total_budget": 10, "lambda_value": float("inf")},
            {"total_budget": 10, "min_difficulty": 0},
            {"total_budget": 10, "min_difficulty": -1e-6},
            {"total_budget": 10, "min_difficulty": float("nan")},
            {"total_budget": 10, "min_difficulty": float("inf")},
        ],
    )
    def test_invalid_allocation_config_raises(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            AllocationConfig(**kwargs)

    def test_allocation_config_is_frozen(self) -> None:
        cfg = AllocationConfig(total_budget=10)
        with pytest.raises(FrozenInstanceError):
            cfg.total_budget = 20  # type: ignore[misc]

    def test_dipa_yaml_schema_and_config_loading(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "dipa.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        for key in (
            "backbone_id",
            "dataset_id",
            "split",
            "prompt_template",
            "generation",
            "allocation",
        ):
            assert key in data

        assert isinstance(data["backbone_id"], str) and data["backbone_id"]
        assert isinstance(data["dataset_id"], str) and data["dataset_id"]
        assert isinstance(data["split"], str) and data["split"]
        assert "system" in data["prompt_template"]
        assert "user" in data["prompt_template"]
        assert data["generation"]["max_new_tokens"] > 0

        alloc = data["allocation"]
        for key in (
            "input_proxy",
            "gen_proxy",
            "lambda_mode",
            "lambda_value",
            "min_difficulty",
            "seeds",
            "budget_multipliers",
        ):
            assert key in alloc

        assert len(alloc["seeds"]) >= 1
        assert all(m > 0 for m in alloc["budget_multipliers"])

        loaded_cfg = AllocationConfig(
            total_budget=100,
            input_proxy=alloc["input_proxy"],
            gen_proxy=alloc["gen_proxy"],
            lambda_mode=alloc["lambda_mode"],
            lambda_value=float(alloc["lambda_value"]),
            min_difficulty=float(alloc["min_difficulty"]),
            seed=int(alloc["seeds"][0]),
        )
        assert loaded_cfg.input_proxy == "mql"
        assert loaded_cfg.gen_proxy == "mgl"
        assert loaded_cfg.lambda_mode == "dynamic"


class TestStoppingPolicyContract:
    """Tests for ``StoppingPolicy`` base class and ``BudgetAllocator`` policy contract."""

    def test_base_stopping_policy_abstract_methods(self) -> None:
        policy = StoppingPolicy()
        policy.reset()
        with pytest.raises(NotImplementedError):
            policy.should_stop([], {}, 10)
        with pytest.raises(NotImplementedError):
            policy.report()

    def test_budget_allocator_select_arm_not_implemented(self) -> None:
        alloc = BudgetAllocator(AllocationConfig(total_budget=5))
        with pytest.raises(NotImplementedError):
            alloc.select_arm([], np.random.default_rng(0))

    def test_should_stop_conditions_reset_and_report_metadata(self) -> None:
        alloc = DIPABudgetAllocator(AllocationConfig(total_budget=5))
        alloc.reset({"batch": "test"})

        assert alloc.should_stop([], {}, budget=0) is True
        assert alloc.should_stop([], {}, budget=-2) is True
        assert (
            alloc.should_stop([], {"budget_used": 5, "active_arms": 2}, budget=5)
            is True
        )
        assert (
            alloc.should_stop([], {"budget_used": 6, "active_arms": 2}, budget=5)
            is True
        )
        assert (
            alloc.should_stop([], {"budget_used": 2, "active_arms": 0}, budget=5)
            is True
        )
        assert alloc.should_stop({"active_arms": 0}, {"budget_used": 2}, budget=5) is True

        solved_arms = [
            ArmState(
                instance=ProblemInstance("p1", "1+1", "2"),
                difficulty=1.0,
                solved=True,
            )
        ]
        assert alloc.should_stop(solved_arms, {"budget_used": 1}, budget=5) is True

        unsolved_arms = [
            ArmState(
                instance=ProblemInstance("p2", "1+1", "2"),
                difficulty=1.0,
                solved=False,
            )
        ]
        assert alloc.should_stop(unsolved_arms, {"budget_used": 1}, budget=5) is False

        pre_report = alloc.report()
        assert pre_report["method"] == "dipa"
        assert pre_report["deterministic"] is False
        assert pre_report["total_budget"] == 5
        assert pre_report["budget_used"] == 0

        report = alloc.allocate(
            [ProblemInstance("p1", "1+1", "2")],
            lambda inst, idx, rng: GenerationAttempt(
                text=r"\boxed{2}", num_tokens=3, latency_ms=2.5
            ),
        )
        assert report.budget_used == 1
        post_report = alloc.report()
        for key in (
            "method",
            "deterministic",
            "total_budget",
            "budget_used",
            "num_instances",
            "num_solved",
            "coverage",
            "per_instance_allocations",
            "solved_ids",
            "unsolved_ids",
            "total_tokens",
            "total_latency_ms",
            "config",
        ):
            assert key in post_report
        assert post_report["deterministic"] is False

        alloc.reset()
        reset_report = alloc.report()
        assert reset_report["budget_used"] == 0
        assert "per_instance_allocations" not in reset_report

        for baseline_cls in (UniformAllocator, Easy2HardAllocator, Hard2EasyAllocator):
            baseline = baseline_cls(AllocationConfig(total_budget=3))
            assert baseline.report()["deterministic"] is True


class TestAllocatorEdgeCasesAndBehavior:
    """Budget edge cases, missing telemetry, probability stabilization, and determinism."""

    def test_empty_instances_and_zero_budget(self) -> None:
        calls = 0

        def gen(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            nonlocal calls
            calls += 1
            return GenerationAttempt(text=r"\boxed{1}", num_tokens=5)

        alloc = DIPABudgetAllocator(AllocationConfig(total_budget=10))
        empty_report = alloc.allocate([], gen)
        assert calls == 0
        assert empty_report.budget_used == 0
        assert empty_report.coverage == 0.0
        assert empty_report.num_instances == 0

        insts = [ProblemInstance("p1", "q", "1")]
        zero_alloc = DIPABudgetAllocator(AllocationConfig(total_budget=0))
        zero_report = zero_alloc.allocate(insts, gen)
        assert calls == 0
        assert zero_report.budget_used == 0
        assert zero_report.coverage == 0.0
        assert zero_report.per_instance_allocations == {"p1": 0}

    @pytest.mark.parametrize(
        ("num_instances", "total_budget"),
        [(5, 3), (4, 4), (3, 10)],
    )
    def test_budget_ratios_less_equal_greater_than_num_instances(
        self, num_instances: int, total_budget: int
    ) -> None:
        insts = [
            ProblemInstance(f"p{i}", f"word " * (i + 1), str(i))
            for i in range(num_instances)
        ]

        def gen_unsolved(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(text=r"\boxed{wrong}", num_tokens=4)

        report = DIPABudgetAllocator(AllocationConfig(total_budget=total_budget)).allocate(
            insts, gen_unsolved
        )
        assert report.budget_used == total_budget
        assert sum(report.per_instance_allocations.values()) == total_budget
        assert report.num_solved == 0
        if total_budget < num_instances:
            assert any(v == 0 for v in report.per_instance_allocations.values())

    def test_early_stopping_when_all_solved(self) -> None:
        def gen(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(
                text=rf"\boxed{{{inst.ground_truth}}}", num_tokens=4, latency_ms=1.5
            )

        insts = [ProblemInstance(f"p{i}", f"prob {i}", str(i)) for i in range(3)]
        alloc = DIPABudgetAllocator(AllocationConfig(total_budget=20))
        report = alloc.allocate(insts, gen)

        assert report.budget_used == 3
        assert report.budget_used < report.total_budget
        assert report.coverage == 1.0
        assert report.num_solved == 3
        assert report.unsolved_ids == []
        assert report.total_tokens == 12
        assert report.total_latency_ms == pytest.approx(4.5)

    @pytest.mark.parametrize("gen_proxy", ["mgl", "mgc"])
    def test_missing_telemetry_handling(self, gen_proxy: str) -> None:
        insts = [
            ProblemInstance("p0", "q0", "1"),
            ProblemInstance("p1", "q1 q1", "2"),
        ]

        def gen_empty(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(text="", extracted_answer=None, num_tokens=0)

        report = DIPABudgetAllocator(
            AllocationConfig(total_budget=6, gen_proxy=gen_proxy, seed=7)
        ).allocate(insts, gen_empty)
        assert report.budget_used == 6
        assert report.total_tokens == 0
        assert report.num_solved == 0
        assert sum(report.per_instance_allocations.values()) == 6

    def test_allocate_with_verifier_fn_none(self) -> None:
        def gen(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(
                text="raw text without box",
                extracted_answer="42",
                num_tokens=3,
                is_correct=(inst.instance_id == "p0"),
            )

        insts = [
            ProblemInstance("p0", "q0", "42"),
            ProblemInstance("p1", "q1", "99"),
        ]
        alloc = UniformAllocator(
            AllocationConfig(total_budget=4), eliminate_solved=True
        )
        report = alloc.allocate(insts, gen, verifier_fn=None)

        assert report.solved_ids == ["p0"]
        assert report.unsolved_ids == ["p1"]
        assert report.per_instance_allocations == {"p0": 1, "p1": 3}

    def test_compute_sampling_probabilities_modes_sharpening_and_500_arm_stability(
        self,
    ) -> None:
        inst = ProblemInstance("p", "q", "1")
        arms = [
            ArmState(instance=inst, difficulty=1.0),
            ArmState(instance=inst, difficulty=2.0),
        ]

        alloc_empty = DIPABudgetAllocator(AllocationConfig(total_budget=10))
        assert alloc_empty.compute_sampling_probabilities([]).size == 0

        p_low_lam = DIPABudgetAllocator(
            AllocationConfig(total_budget=10, lambda_mode="fixed", lambda_value=0.5)
        ).compute_sampling_probabilities(arms)
        p_mid_lam = DIPABudgetAllocator(
            AllocationConfig(total_budget=10, lambda_mode="fixed", lambda_value=1.0)
        ).compute_sampling_probabilities(arms)
        p_high_lam = DIPABudgetAllocator(
            AllocationConfig(total_budget=10, lambda_mode="fixed", lambda_value=4.0)
        ).compute_sampling_probabilities(arms)

        assert p_mid_lam.tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
        assert p_low_lam[0] < p_mid_lam[0] < p_high_lam[0]

        alloc_dyn = DIPABudgetAllocator(
            AllocationConfig(total_budget=10, lambda_mode="dynamic", lambda_value=1.0)
        )
        p_dyn = alloc_dyn.compute_sampling_probabilities(arms)
        assert p_dyn.tolist() == pytest.approx([0.8, 0.2])
        assert p_dyn[0] > p_mid_lam[0]

        # 500 arms with lambda_t=500 across [1e-6, 5e5] would overflow float64 without log-space shift.
        diffs_500 = np.geomspace(1e-6, 5e5, 500)
        many_arms = [
            ArmState(
                instance=ProblemInstance(f"p{i}", "q", "1"), difficulty=float(d)
            )
            for i, d in enumerate(diffs_500)
        ]
        p_many = alloc_dyn.compute_sampling_probabilities(many_arms)
        assert p_many.shape == (500,)
        assert np.all(np.isfinite(p_many))
        assert float(np.sum(p_many)) == pytest.approx(1.0)
        assert p_many[0] > p_many[-1]

        bad_arms = [
            ArmState(instance=inst, difficulty=float("nan")),
            ArmState(instance=inst, difficulty=0.0),
            ArmState(instance=inst, difficulty=float("inf")),
            ArmState(instance=inst, difficulty=float("-inf")),
        ]
        p_bad = alloc_dyn.compute_sampling_probabilities(bad_arms)
        assert np.all(np.isfinite(p_bad))
        assert float(np.sum(p_bad)) == pytest.approx(1.0)

    def test_dipa_determinism_by_seed(self) -> None:
        insts = [
            ProblemInstance(f"p{i}", " ".join(["w"] * (i + 1)), str(i))
            for i in range(6)
        ]

        def gen(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            ok = rng.random() < 0.25
            ans = inst.ground_truth if ok else "wrong"
            return GenerationAttempt(text=rf"\boxed{{{ans}}}", num_tokens=5)

        r1 = DIPABudgetAllocator(AllocationConfig(total_budget=12, seed=42)).allocate(
            insts, gen
        )
        r2 = DIPABudgetAllocator(AllocationConfig(total_budget=12, seed=42)).allocate(
            insts, gen
        )
        assert r1.per_instance_allocations == r2.per_instance_allocations
        assert r1.solved_ids == r2.solved_ids
        assert r1.unsolved_ids == r2.unsolved_ids
        assert r1.coverage == r2.coverage

        reports = [
            DIPABudgetAllocator(AllocationConfig(total_budget=12, seed=s)).allocate(
                insts, gen
            )
            for s in (42, 43, 44, 45)
        ]
        for rep in reports:
            assert rep.budget_used <= 12
            assert sum(rep.per_instance_allocations.values()) == rep.budget_used
        assert (
            len({tuple(sorted(r.per_instance_allocations.items())) for r in reports})
            > 1
        )


class TestBaselinesAndBenchmark:
    """Tests for Uniform, Easy2Hard, Hard2Easy, and DIPA synthetic benchmark."""

    def test_uniform_allocator_round_robin_and_elimination(self) -> None:
        insts = [ProblemInstance(f"p{i}", "q", str(i)) for i in range(4)]

        def gen_partial(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            ans = inst.ground_truth if inst.instance_id == "p0" else "wrong"
            return GenerationAttempt(text=rf"\boxed{{{ans}}}", num_tokens=2)

        def gen_all_solved(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(text=rf"\boxed{{{inst.ground_truth}}}", num_tokens=2)

        r_no_elim = UniformAllocator(
            AllocationConfig(total_budget=10), eliminate_solved=False
        ).allocate(insts, gen_partial)
        assert list(r_no_elim.per_instance_allocations.values()) == [3, 3, 2, 2]
        assert r_no_elim.budget_used == 10

        r_no_elim_all_solved = UniformAllocator(
            AllocationConfig(total_budget=10), eliminate_solved=False
        ).allocate(insts, gen_all_solved)
        assert list(r_no_elim_all_solved.per_instance_allocations.values()) == [
            3,
            3,
            2,
            2,
        ]
        assert r_no_elim_all_solved.budget_used == 10
        assert r_no_elim_all_solved.coverage == 1.0

        r_elim_partial = UniformAllocator(
            AllocationConfig(total_budget=10), eliminate_solved=True
        ).allocate(insts, gen_partial)
        assert r_elim_partial.per_instance_allocations == {
            "p0": 1,
            "p1": 3,
            "p2": 3,
            "p3": 3,
        }

        r_elim_all = UniformAllocator(
            AllocationConfig(total_budget=10), eliminate_solved=True
        ).allocate(insts, gen_all_solved)
        assert r_elim_all.budget_used == 4
        assert r_elim_all.per_instance_allocations == {
            "p0": 1,
            "p1": 1,
            "p2": 1,
            "p3": 1,
        }

    def test_easy2hard_and_hard2easy_selection_order(self) -> None:
        insts = [
            ProblemInstance("easy", "one", "1"),
            ProblemInstance("mid", "one two three", "2"),
            ProblemInstance("hard", "one two three four five", "3"),
        ]
        token_map = {"easy": 1, "mid": 3, "hard": 5}

        def gen_unsolved(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            return GenerationAttempt(
                text=r"\boxed{wrong}", num_tokens=token_map[inst.instance_id]
            )

        e2h = Easy2HardAllocator(
            AllocationConfig(total_budget=3, input_proxy="mql", gen_proxy="mgl")
        ).allocate(insts, gen_unsolved)
        assert e2h.per_instance_allocations == {"easy": 3, "mid": 0, "hard": 0}

        h2e = Hard2EasyAllocator(
            AllocationConfig(total_budget=3, input_proxy="mql", gen_proxy="mgl")
        ).allocate(insts, gen_unsolved)
        assert h2e.per_instance_allocations == {"easy": 0, "mid": 0, "hard": 3}

        def gen_solve_on_second(
            inst: ProblemInstance, attempt_idx: int, rng: np.random.Generator
        ) -> GenerationAttempt:
            ans = inst.ground_truth if attempt_idx >= 1 else "wrong"
            return GenerationAttempt(
                text=rf"\boxed{{{ans}}}", num_tokens=token_map[inst.instance_id]
            )

        e2h_seq = Easy2HardAllocator(
            AllocationConfig(total_budget=4, input_proxy="mql", gen_proxy="mgl")
        ).allocate(insts, gen_solve_on_second)
        assert e2h_seq.per_instance_allocations == {"easy": 2, "mid": 2, "hard": 0}
        assert e2h_seq.solved_ids == ["easy", "mid"]

        h2e_seq = Hard2EasyAllocator(
            AllocationConfig(total_budget=4, input_proxy="mql", gen_proxy="mgl")
        ).allocate(insts, gen_solve_on_second)
        assert h2e_seq.per_instance_allocations == {"easy": 0, "mid": 2, "hard": 2}
        assert h2e_seq.solved_ids == ["mid", "hard"]

    def test_end_to_end_synthetic_benchmark_dipa_beats_baselines(self) -> None:
        instances: list[ProblemInstance] = []
        for idx in range(40):
            if idx % 2 == 0:
                instances.append(ProblemInstance(f"easy_{idx}", "1 + 1", "2"))
            else:
                hard_prompt = " ".join(f"tok{j}" for j in range(30))
                instances.append(ProblemInstance(f"hard_{idx}", hard_prompt, "99"))

        def synthetic_gen(
            inst: ProblemInstance,
            attempt_idx: int,
            rng: np.random.Generator,
        ) -> GenerationAttempt:
            if inst.instance_id.startswith("easy_"):
                solved = attempt_idx >= 1 or (int(inst.instance_id.split("_")[1]) % 4 == 0)
                ans = inst.ground_truth if solved else "0"
                return GenerationAttempt(text=rf"\boxed{{{ans}}}", num_tokens=8)

            wrong_ans = str(1000 + attempt_idx * 7 + int(inst.instance_id.split("_")[1]))
            return GenerationAttempt(text=rf"\boxed{{{wrong_ans}}}", num_tokens=200)

        uniform_report = UniformAllocator(
            AllocationConfig(total_budget=20, seed=42), eliminate_solved=True
        ).allocate(instances, synthetic_gen)
        hard2easy_report = Hard2EasyAllocator(
            AllocationConfig(total_budget=20, input_proxy="mql", gen_proxy="mgl", seed=42)
        ).allocate(instances, synthetic_gen)
        dipa_mgl_report = DIPABudgetAllocator(
            AllocationConfig(total_budget=20, input_proxy="mql", gen_proxy="mgl", seed=42)
        ).allocate(instances, synthetic_gen)
        dipa_mgc_report = DIPABudgetAllocator(
            AllocationConfig(total_budget=20, input_proxy="mql", gen_proxy="mgc", seed=42)
        ).allocate(instances, synthetic_gen)

        assert dipa_mgl_report.coverage > uniform_report.coverage
        assert dipa_mgl_report.coverage > hard2easy_report.coverage
        assert dipa_mgc_report.coverage > uniform_report.coverage
        assert dipa_mgc_report.coverage > hard2easy_report.coverage
