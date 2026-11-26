"""Metrics (pass@k, pass@NFE, Best@k, Entropy@k), grading and diffu-GRPO tests."""

import math

import pytest
import torch

from tempered_remasking import SamplerConfig, build_policy
from tempered_remasking import metrics as M
from tempered_remasking import tasks as T
from tempered_remasking.grpo import (
    DiffuGRPOConfig,
    DiffuGRPOTrainer,
    completion_mask,
    diffu_grpo_loss,
    group_advantages,
    one_step_logprobs,
    random_prompt_mask,
)

MASK = 0


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_pass_at_k_values():
    assert math.isclose(M.pass_at_k(10, 3, 1), 0.3)
    assert M.pass_at_k(10, 0, 5) == 0.0
    assert M.pass_at_k(10, 10, 1) == 1.0
    assert M.pass_at_k(10, 3, 8) == 1.0  # n - c < k
    assert math.isclose(M.pass_at_k(5, 2, 2), 1 - math.comb(3, 2) / math.comb(5, 2))
    with pytest.raises(ValueError):
        M.pass_at_k(4, 5, 1)


def test_pass_at_k_and_nfe_curves():
    correct = [[True, False, False, False], [False, False, False, False]]
    nfes = [[10, 20, 30, 40], [5, 5, 5, 5]]
    curve = M.pass_at_k_curve(correct)
    assert list(curve) == [1, 2, 4] and math.isclose(curve[1], 0.125) and math.isclose(curve[4], 0.5)
    pn = M.pass_at_nfe_curve(correct, nfes)
    assert [p["k"] for p in pn] == [1, 2, 4]
    assert math.isclose(pn[0]["nfe"], (25 + 5) / 2) and math.isclose(pn[2]["nfe"], (100 + 20) / 2)
    assert [p["pass"] for p in pn] == [curve[k] for k in (1, 2, 4)]
    b = M.pass_within_budget(correct, nfes, [0, 1000], num_trials=50)
    assert b[0] == 0.0 and b[1000] == 0.5


def test_best_at_k_self_consistency_vs_orm():
    # problem: correct answer "7" appears once; "3" is the majority; ORM ranks "7" highest
    answers = [["3", "3", "7", "3"]]
    scores = [[0.1, 0.2, 0.9, 0.3]]
    assert M.best_at_k(answers, ["7"], 4, "self_consistency") == 0.0
    assert M.best_at_k(answers, ["7"], 4, "orm", scores=scores) == 1.0
    assert math.isclose(M.best_at_k(answers, ["7"], 1, "orm", scores=scores), 0.25)
    curve = M.best_at_nfe_curve(answers, ["7"], [[10, 10, 10, 10]], method="orm", scores=scores)
    assert curve[-1]["nfe"] == 40 and curve[-1]["best"] == 1.0
    assert M.majority_vote([None, None]) is None


def test_entropy_at_k_and_group_stats():
    assert M.answer_entropy(["a", "a"]) == 0
    assert math.isclose(M.answer_entropy(["a", "b"]), math.log(2))
    assert math.isclose(M.entropy_at_k([["a", "b", "c", "d"]], 4), math.log(4))
    assert M.entropy_at_k([["a"] * 8], 2) == 0
    st = M.group_reward_stats([[1, 1, 1], [0, 2, 0, 2]])
    assert st["frac_zero_std_groups"] == 0.5 and math.isclose(st["reward_std"], 0.5)


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def test_math_answer_extraction_and_grading():
    assert T.extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert T.extract_boxed(r"\boxed{a{b}c} and \boxed{42}") == "42"
    assert T.grade_gsm8k("... the answer is \\boxed{1,234}", T.extract_gsm8k_reference("#### 1234"))[1]
    assert T.grade_gsm8k("we get 18 apples, so 18.", "18")[1]
    assert not T.grade_gsm8k("no number here", "18")[1]
    assert T.grade_math(r"\boxed{\dfrac{1}{2}}", r"\frac12")[1]
    assert T.grade_math(r"\boxed{0.5}", r"\frac{1}{2}")[1]
    assert T.grade_math(r"\boxed{x = 5}", "5")[1]
    assert T.grade_math(r"\boxed{90^\circ}", "90")[1]
    assert not T.grade_math("no box", "5")[1]


def test_code_grading_runs_tests_in_subprocess():
    comp = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert T.grade_mbpp(comp, ["assert add(1, 2) == 3"])
    assert not T.grade_mbpp(comp, ["assert add(1, 2) == 4"])
    he_prompt = "def inc(x):\n"
    test = "def check(f):\n    assert f(1) == 2\n"
    assert T.grade_humaneval("```python\n    return x + 1\n```", he_prompt, test, "inc")
    assert not T.run_python_tests("while True: pass", timeout_s=1)


# --------------------------------------------------------------------------- #
# diffu-GRPO
# --------------------------------------------------------------------------- #
class TinyDLM(torch.nn.Module):
    def __init__(self, vocab=8, seq_len=12):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, 16)
        self.pos = torch.nn.Parameter(torch.randn(seq_len, 16) * 0.1)
        self.out = torch.nn.Linear(32, vocab)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        ctx = h.mean(1, keepdim=True).expand_as(h)
        logits = self.out(torch.cat([h, ctx], -1))
        logits = logits.clone()
        logits[..., MASK] = -1e4
        return logits


def test_grpo_building_blocks():
    model = TinyDLM()
    prompt = torch.tensor([[3, 4, 5, 6]] * 3)
    comp = torch.tensor([[1, 2, 7, 1, 1, 1, 1, 1]] * 3)
    m = random_prompt_mask(prompt, 0.5, torch.Generator().manual_seed(0))
    lp = one_step_logprobs(model, prompt, comp, MASK, m)
    assert lp.shape == (3, 8) and torch.all(lp <= 0)
    cm = completion_mask(comp, [7])
    assert cm[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert torch.allclose(group_advantages(torch.tensor([1.0, 1.0, 1.0])), torch.zeros(3))  # no signal
    adv = group_advantages(torch.tensor([0.0, 2.0, 1.0]))
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, 0.0]))
    loss, st = diffu_grpo_loss(lp, lp.detach(), lp.detach(), adv, cm)
    loss.backward()
    assert torch.isfinite(loss) and st["kl"] == 0.0
    assert model.out.weight.grad is not None and model.out.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("sampler", ["tct", "ct", "ar"])
def test_grpo_trainer_runs_with_each_rollout_sampler(sampler):
    model = TinyDLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    def reward(tokens, ex):
        return float(sum(t == ex["target"] for t in tokens))

    cfg = DiffuGRPOConfig(num_generations=4, num_iterations=2, max_steps=2, time_budget_s=60, rollout_batch_size=4)
    tr = DiffuGRPOTrainer(model, None, opt, build_policy(sampler), SamplerConfig(gen_length=8, block_length=4),
                          reward, MASK, eos_token_ids=[7], config=cfg, generator=torch.Generator().manual_seed(0))
    data = iter([{"prompt_ids": torch.tensor([3, 4, 5, 6]), "target": 2}] * 10)
    hist = tr.train(data)
    assert len(hist) == 2
    for rec in hist:
        assert math.isfinite(rec["loss"]) and rec["rollout_nfe_mean"] >= 1 and "reward_std" in rec
    assert tr.config_dict()["rollout_policy"]["name"] in ("tct", "confidence_threshold", "autoregressive")


def test_grpo_time_budget_zero_runs_no_steps():
    model = TinyDLM()
    tr = DiffuGRPOTrainer(model, None, torch.optim.SGD(model.parameters(), lr=0.1), build_policy("tct"),
                          SamplerConfig(gen_length=8, block_length=4), lambda t, e: 0.0, MASK,
                          config=DiffuGRPOConfig(time_budget_s=0.0))
    assert tr.train(iter([{"prompt_ids": torch.tensor([3, 4, 5, 6])}])) == []


# --------------------------------------------------------------------------- #
# model adapters and paper config
# --------------------------------------------------------------------------- #
def test_model_adapters_and_dream_logit_shift():
    from types import SimpleNamespace

    from tempered_remasking.models import DreamAdapter, LLaDAAdapter

    class Fake(torch.nn.Module):
        def forward(self, x, attention_mask=None):
            return SimpleNamespace(logits=torch.arange(x.shape[1], dtype=torch.float32).view(1, -1, 1).expand(x.shape[0], -1, 3))

    x = torch.zeros(2, 5, dtype=torch.long)
    assert LLaDAAdapter(Fake())(x)[0, :, 0].tolist() == [0, 1, 2, 3, 4]
    # Dream: logits at k predict k+1 -> shift right by one, duplicating the first
    assert DreamAdapter(Fake())(x)[0, :, 0].tolist() == [0, 0, 1, 2, 3]


def test_paper_config_presets_build():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "tempered_remasking"))
    pytest.importorskip("yaml")
    import common

    cfg = common.load_config()
    sc = common.sampler_config(cfg)
    assert (sc.gen_length, sc.block_length, sc.token_temperature) == (256, 32, 0.8)
    tlc = common.policy_from_preset(cfg, "tlc")
    tct = common.policy_from_preset(cfg, "tct")
    assert tlc.tpos == 1.0 and (tct.tpos, tct.threshold) == (0.1, 0.6)
    for preset in cfg["presets"]:
        sc.validate(common.policy_from_preset(cfg, preset))
