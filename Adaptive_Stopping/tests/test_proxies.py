"""Tests for pure-Python, torch-free, and PyTorch DIPA difficulty proxies."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from Adaptive_Stopping.methods.dipa.proxies import (
    compute_gen_difficulty,
    compute_input_difficulty,
    to_difficulty_score,
)
from Adaptive_Stopping.methods.dipa.types import GenerationAttempt, ProblemInstance


class _EncodeOnlyTokenizer:
    """Minimal tokenizer exposing only ``.encode()``."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [1 + (sum(ord(ch) for ch in word) % 10) for word in text.split()]
        if add_special_tokens and not tokens:
            return [1]
        return tokens


class _EmptyEncodeTokenizer:
    """Tokenizer returning zero tokens for empty strings."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1 for _ in text.split()]


class _DummyTokenizer:
    """Tokenizer supporting both ``__call__(return_tensors='pt')`` and ``.encode()``."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = vocab_size

    def _to_ids(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [
            1 + (ord(ch) % (self.vocab_size - 2))
            for ch in text.strip()
            if not ch.isspace()
        ]
        if add_special_tokens:
            return [0] + ids
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._to_ids(text, add_special_tokens=add_special_tokens)

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        import torch

        ids = self._to_ids(text, add_special_tokens=add_special_tokens)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _build_tiny_causal_lm(vocab_size: int = 16) -> Any:
    """Build a tiny 2-layer causal LM with ``nn.Embedding``, ``nn.Linear``, and ``get_input_embeddings``."""
    import torch
    import torch.nn as nn

    class TinyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 8)
            self.hidden = nn.Linear(8, 8)
            self.head = nn.Linear(8, vocab_size)
            with torch.no_grad():
                nn.init.normal_(self.embed.weight, mean=0.0, std=0.5)
                nn.init.normal_(self.hidden.weight, mean=0.0, std=0.5)
                nn.init.zeros_(self.hidden.bias)
                nn.init.normal_(self.head.weight, mean=0.0, std=0.5)
                nn.init.zeros_(self.head.bias)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embed

        def forward(
            self,
            input_ids: Any = None,
            inputs_embeds: Any = None,
        ) -> SimpleNamespace:
            x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
            h = torch.tanh(self.hidden(x))
            return SimpleNamespace(logits=self.head(h))

    torch.manual_seed(0)
    model = TinyCausalLM()
    model.eval()
    return model


def _build_contrast_causal_lm(vocab_size: int = 16) -> Any:
    """Build a causal LM where token IDs 0 and 1 yield sharp logits and others yield uniform logits."""
    import torch
    import torch.nn as nn

    class ContrastCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 8)
            self.head = nn.Linear(8, vocab_size, bias=False)
            with torch.no_grad():
                self.embed.weight.zero_()
                self.head.weight.zero_()
                self.embed.weight[0, 0] = 10.0
                self.embed.weight[1, 1] = 10.0
                self.head.weight[1, 0] = 5.0
                self.head.weight[1, 1] = 5.0

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embed

        def forward(
            self,
            input_ids: Any = None,
            inputs_embeds: Any = None,
        ) -> SimpleNamespace:
            hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
            return SimpleNamespace(logits=self.head(hidden))

    model = ContrastCausalLM()
    model.eval()
    return model


class TestToDifficultyScore:
    """Tests for ``to_difficulty_score``."""

    @pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_falls_back_to_unit_or_min_difficulty(self, raw: float) -> None:
        assert to_difficulty_score(raw, negatively_correlated=False) == 1.0
        assert to_difficulty_score(raw, negatively_correlated=True) == 1.0
        assert (
            to_difficulty_score(raw, negatively_correlated=False, min_difficulty=2.5)
            == 2.5
        )

    def test_zero_and_negative_clamped_to_min_difficulty(self) -> None:
        assert (
            to_difficulty_score(0.0, negatively_correlated=False, min_difficulty=1e-4)
            == 1e-4
        )
        assert (
            to_difficulty_score(-5.0, negatively_correlated=False, min_difficulty=1e-4)
            == 1e-4
        )

    def test_negatively_correlated_inverts_and_clamps(self) -> None:
        assert to_difficulty_score(2.0, negatively_correlated=True) == pytest.approx(0.5)
        assert to_difficulty_score(
            0.0, negatively_correlated=True, min_difficulty=1e-3
        ) == pytest.approx(1000.0)
        assert to_difficulty_score(
            1e9, negatively_correlated=True, min_difficulty=1e-3
        ) == pytest.approx(1e-3)


class TestPurePythonProxies:
    """Tests for model-free proxies, registry errors, and torch-free execution."""

    def test_const_proxies_return_one(self) -> None:
        inst = ProblemInstance("id1", "What is 2 + 2?", "4")
        assert compute_input_difficulty("const", inst) == 1.0
        assert compute_gen_difficulty("const", inst, []) == 1.0
        assert compute_gen_difficulty("const", inst, [GenerationAttempt("4")]) == 1.0

    def test_mql_whitespace_tokenizer_and_empty_clamp(self) -> None:
        inst = ProblemInstance("id1", "Solve x plus y equals ten", "5")
        assert compute_input_difficulty("mql", inst, tokenizer=None) == 6.0
        assert (
            compute_input_difficulty("mql", inst, tokenizer=_EncodeOnlyTokenizer())
            == 6.0
        )

        empty_inst = ProblemInstance("id2", "   ", "0")
        assert compute_input_difficulty("mql", empty_inst, tokenizer=None) == 1.0
        assert (
            compute_input_difficulty(
                "mql", empty_inst, tokenizer=_EmptyEncodeTokenizer()
            )
            == 1.0
        )

    def test_mgl_averages_tokens_with_whitespace_fallback_and_empty(self) -> None:
        inst = ProblemInstance("id1", "p", "1")
        assert compute_gen_difficulty("mgl", inst, []) == 1.0

        attempts = [
            GenerationAttempt(text="a b", num_tokens=10),
            GenerationAttempt(text="one two three four", num_tokens=0),
        ]
        assert compute_gen_difficulty("mgl", inst, attempts) == pytest.approx(7.0)

        blank_attempt = [GenerationAttempt(text="   ", num_tokens=0)]
        assert compute_gen_difficulty("mgl", inst, blank_attempt) == 1.0

    def test_mgc_agreement_normalization_fallback_and_inversion(self) -> None:
        inst = ProblemInstance("id1", "p", "2")
        assert compute_gen_difficulty("mgc", inst, []) == 1.0

        normalized_unanimous = [
            GenerationAttempt(text="x", extracted_answer=r"\dfrac{1}{2}"),
            GenerationAttempt(text="y", extracted_answer=r"$\frac{1}{2}$"),
        ]
        assert compute_gen_difficulty(
            "mgc", inst, normalized_unanimous
        ) == pytest.approx(1.0)

        split = [
            GenerationAttempt(text="a", extracted_answer="1,000"),
            GenerationAttempt(text="b", extracted_answer="1000"),
            GenerationAttempt(text="c", extracted_answer="$1000$"),
            GenerationAttempt(text="d", extracted_answer="999"),
        ]
        assert compute_gen_difficulty("mgc", inst, split) == pytest.approx(4.0 / 3.0)

        distinct = [
            GenerationAttempt(text="1", extracted_answer="1"),
            GenerationAttempt(text="2", extracted_answer="2"),
            GenerationAttempt(text="3", extracted_answer="3"),
            GenerationAttempt(text="4", extracted_answer=None),
        ]
        assert compute_gen_difficulty("mgc", inst, distinct) == pytest.approx(4.0)

        fallback_text = [
            GenerationAttempt(text="  alpha  ", extracted_answer=None),
            GenerationAttempt(text="alpha", extracted_answer=None),
            GenerationAttempt(text="beta", extracted_answer=None),
        ]
        assert compute_gen_difficulty("mgc", inst, fallback_text) == pytest.approx(1.5)

    def test_unknown_proxy_raises_value_error(self) -> None:
        inst = ProblemInstance("id1", "p", "1")
        with pytest.raises(ValueError, match="Unknown input proxy"):
            compute_input_difficulty("does_not_exist", inst)
        with pytest.raises(ValueError, match="Unknown generation proxy"):
            compute_gen_difficulty("does_not_exist", inst, [])

    @pytest.mark.parametrize("proxy_name", ["ment", "mgn", "mvog"])
    def test_missing_model_or_tokenizer_raises_value_error(
        self, proxy_name: str
    ) -> None:
        inst = ProblemInstance("id1", "p", "1")
        with pytest.raises(ValueError, match="requires both model and tokenizer"):
            compute_input_difficulty(proxy_name, inst, model=None, tokenizer=object())
        with pytest.raises(ValueError, match="requires both model and tokenizer"):
            compute_input_difficulty(proxy_name, inst, model=object(), tokenizer=None)
        with pytest.raises(ValueError, match="requires both model and tokenizer"):
            compute_gen_difficulty(
                proxy_name, inst, [], model=None, tokenizer=object()
            )
        with pytest.raises(ValueError, match="requires both model and tokenizer"):
            compute_gen_difficulty(
                proxy_name, inst, [], model=object(), tokenizer=None
            )

    def test_core_allocator_stays_torch_free_in_clean_subprocess(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
        )
        script = (
            "import sys\n"
            "from Adaptive_Stopping.methods.dipa.allocator import DIPABudgetAllocator\n"
            "from Adaptive_Stopping.methods.dipa.types import (\n"
            "    AllocationConfig, GenerationAttempt, ProblemInstance\n"
            ")\n"
            "insts = [ProblemInstance(f'p{i}', f'q {i}', str(i)) for i in range(4)]\n"
            "def gen(inst, idx, rng):\n"
            "    return GenerationAttempt(text=f'\\\\boxed{{{inst.ground_truth}}}', num_tokens=3)\n"
            "for gp in ('mgl', 'mgc'):\n"
            "    report = DIPABudgetAllocator(AllocationConfig(total_budget=4, gen_proxy=gp)).allocate(insts, gen)\n"
            "    assert report.budget_used > 0\n"
            "assert 'torch' not in sys.modules, 'torch must not be imported for zero-overhead proxies'\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            env=env,
            check=True,
        )


class TestTorchProxies:
    """Real PyTorch execution tests for ``ment``, ``mgn``, and ``mvog``."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch")

    def test_input_ment_mgn_mvog_finite_positive_and_informative(self) -> None:
        model = _build_tiny_causal_lm()
        tokenizer = _DummyTokenizer()
        inst_a = ProblemInstance("a", "abcdef", "1")
        inst_b = ProblemInstance("b", "zyxwvu", "2")

        for proxy in ("ment", "mgn", "mvog"):
            val_a = compute_input_difficulty(
                proxy, inst_a, model=model, tokenizer=tokenizer
            )
            val_b = compute_input_difficulty(
                proxy, inst_b, model=model, tokenizer=tokenizer
            )
            assert math.isfinite(val_a) and val_a > 0.0
            assert math.isfinite(val_b) and val_b > 0.0
            assert val_a != pytest.approx(val_b)

    def test_input_ment_distinguishes_low_vs_high_entropy(self) -> None:
        class _MappedTokenizer:
            def __init__(self) -> None:
                self._map = {"low": [0, 1, 1, 1], "high": [5, 6, 7, 8]}

            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return self._map[text]

            def __call__(
                self,
                text: str,
                return_tensors: str = "pt",
                add_special_tokens: bool = True,
            ) -> dict[str, Any]:
                import torch

                return {
                    "input_ids": torch.tensor([self._map[text]], dtype=torch.long)
                }

        model = _build_contrast_causal_lm()
        tok = _MappedTokenizer()
        low_ent = compute_input_difficulty(
            "ment", ProblemInstance("low", "low", "1"), model=model, tokenizer=tok
        )
        high_ent = compute_input_difficulty(
            "ment", ProblemInstance("high", "high", "1"), model=model, tokenizer=tok
        )
        assert low_ent < high_ent

    def test_gen_ment_mgn_mvog_with_and_without_token_ids(self) -> None:
        model = _build_tiny_causal_lm()
        tokenizer = _DummyTokenizer()
        inst = ProblemInstance("g1", "abc", "1")
        attempts_with_ids = [
            GenerationAttempt(text="def", token_ids=[11, 12, 13]),
            GenerationAttempt(text="ghi", token_ids=[14, 15, 2]),
        ]
        attempts_text_only = [
            GenerationAttempt(text="def", token_ids=None),
            GenerationAttempt(text="ghi", token_ids=None),
        ]

        for proxy in ("ment", "mgn", "mvog"):
            score_ids = compute_gen_difficulty(
                proxy, inst, attempts_with_ids, model=model, tokenizer=tokenizer
            )
            score_txt = compute_gen_difficulty(
                proxy, inst, attempts_text_only, model=model, tokenizer=tokenizer
            )
            assert math.isfinite(score_ids) and score_ids > 0.0
            assert math.isfinite(score_txt) and score_txt > 0.0
            assert score_ids != pytest.approx(score_txt)

    def test_torch_proxies_work_inside_no_grad(self) -> None:
        import torch

        model = _build_tiny_causal_lm()
        tokenizer = _DummyTokenizer()
        inst = ProblemInstance("ng", "abcdef", "1")
        attempts = [GenerationAttempt(text="ghijkl", token_ids=[3, 4, 5, 6])]

        for proxy in ("ment", "mgn", "mvog"):
            expected_in = compute_input_difficulty(
                proxy, inst, model=model, tokenizer=tokenizer
            )
            expected_gen = compute_gen_difficulty(
                proxy, inst, attempts, model=model, tokenizer=tokenizer
            )
            with torch.no_grad():
                in_score = compute_input_difficulty(
                    proxy, inst, model=model, tokenizer=tokenizer
                )
                gen_score = compute_gen_difficulty(
                    proxy, inst, attempts, model=model, tokenizer=tokenizer
                )
            assert math.isfinite(in_score) and in_score == pytest.approx(expected_in)
            assert math.isfinite(gen_score) and gen_score == pytest.approx(expected_gen)

    def test_encode_only_tokenizer_fallback_with_torch_proxies(self) -> None:
        model = _build_tiny_causal_lm()
        tok = _EncodeOnlyTokenizer()
        inst = ProblemInstance("enc", "one two three", "1")
        attempts = [GenerationAttempt(text="four five", token_ids=None)]

        for proxy in ("ment", "mgn", "mvog"):
            in_score = compute_input_difficulty(proxy, inst, model=model, tokenizer=tok)
            gen_score = compute_gen_difficulty(
                proxy, inst, attempts, model=model, tokenizer=tok
            )
            assert math.isfinite(in_score) and in_score > 0.0
            assert math.isfinite(gen_score) and gen_score > 0.0

    def test_single_token_edge_case_returns_one(self) -> None:
        class _SingleTokenTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1] if add_special_tokens else []

            def __call__(
                self,
                text: str,
                return_tensors: str = "pt",
                add_special_tokens: bool = True,
            ) -> dict[str, Any]:
                import torch

                ids = [1] if add_special_tokens else []
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        model = _build_tiny_causal_lm()
        tok = _SingleTokenTokenizer()
        inst = ProblemInstance("single", "x", "1")

        for proxy in ("ment", "mgn", "mvog"):
            assert (
                compute_input_difficulty(proxy, inst, model=model, tokenizer=tok) == 1.0
            )
            assert (
                compute_gen_difficulty(
                    proxy,
                    inst,
                    [GenerationAttempt(text="", token_ids=None)],
                    model=model,
                    tokenizer=tok,
                )
                == 1.0
            )

    def test_missing_get_input_embeddings_raises_value_error(self) -> None:
        import torch.nn as nn

        class NoEmbedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = nn.Parameter(nn.Linear(1, 1).weight)

            def forward(self, input_ids: Any = None, inputs_embeds: Any = None) -> Any:
                return None

        model = NoEmbedModel()
        tok = _DummyTokenizer()
        inst = ProblemInstance("no_embed", "abcd", "1")

        for proxy in ("mgn", "mvog"):
            with pytest.raises(ValueError, match="get_input_embeddings"):
                compute_input_difficulty(proxy, inst, model=model, tokenizer=tok)
            with pytest.raises(ValueError, match="get_input_embeddings"):
                compute_gen_difficulty(
                    proxy,
                    inst,
                    [GenerationAttempt(text="efgh", token_ids=[2, 3, 4])],
                    model=model,
                    tokenizer=tok,
                )
