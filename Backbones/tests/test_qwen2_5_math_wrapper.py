"""Unit and contract tests for Qwen2.5-Math-1.5B-Instruct backbone wrapper."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct import (
    BackboneConfig,
    Qwen25Math15BInstructBackbone,
    get_model_metadata,
)


class _DummyTokenizer:
    """Minimal tokenizer supporting chat templates, tokenization, and decoding."""

    def __init__(
        self,
        *,
        with_chat_template: bool = True,
        include_attention_mask: bool = True,
        return_torch_tensors: bool = True,
    ) -> None:
        self._include_attention_mask = include_attention_mask
        self._return_torch_tensors = return_torch_tensors
        self.pad_token_id: int | None = None
        self.eos_token_id: int = 2
        self.chat_template_calls: list[dict[str, Any]] = []
        self.decode_calls: list[dict[str, Any]] = []
        if with_chat_template:
            self.apply_chat_template = self._apply_chat_template

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        ids = [10 + (ord(ch) % 20) for ch in text[:8]] or [10]
        if self._return_torch_tensors:
            import torch

            input_ids: Any = torch.tensor([ids], dtype=torch.long)
            out: dict[str, Any] = {"input_ids": input_ids}
            if self._include_attention_mask:
                out["attention_mask"] = torch.ones_like(input_ids)
            return out

        out = {"input_ids": [ids]}
        if self._include_attention_mask:
            out["attention_mask"] = [[1] * len(ids)]
        return out

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        self.decode_calls.append(
            {"token_ids": list(token_ids), "skip_special_tokens": skip_special_tokens}
        )
        filtered = [tok for tok in token_ids if not (skip_special_tokens and tok == 2)]
        return " ".join(f"tok{tok}" for tok in filtered)


class _DummyCausalModel:
    """Minimal causal LM stub recording forward/generate calls and kwargs."""

    def __init__(self, *, use_torch: bool = True) -> None:
        self._use_torch = use_torch
        self.eval_called = False
        self.to_calls: list[Any] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: Any) -> _DummyCausalModel:
        self.to_calls.append(device)
        return self

    def eval(self) -> _DummyCausalModel:
        self.eval_called = True
        return self

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        self.forward_calls.append(inputs)
        return SimpleNamespace(logits="dummy_logits", inputs=inputs)

    def generate(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        num_seqs = int(kwargs["num_return_sequences"])
        do_sample = bool(kwargs["do_sample"])

        if self._use_torch:
            import torch

            prompt_row = input_ids[0].tolist()
            rows = []
            for seq_idx in range(num_seqs):
                if do_sample:
                    rand_tok = int(torch.randint(100, 999, (1,)).item())
                    gen_suffix = [rand_tok, 100 + seq_idx, 2]
                else:
                    gen_suffix = [50 + seq_idx, 51 + seq_idx, 2]
                rows.append(prompt_row + gen_suffix)
            return torch.tensor(rows, dtype=torch.long)

        prompt_row = list(input_ids[0])
        return [prompt_row + [70 + idx, 71 + idx, 2] for idx in range(num_seqs)]


class TestBackboneConfigAndMetadata:
    """Validation tests for ``BackboneConfig``, ``registry.yaml``, and metadata."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": -0.1},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"top_p": 0.0},
            {"top_p": -0.5},
            {"top_p": 1.01},
            {"top_p": float("nan")},
            {"repetition_penalty": 0.0},
            {"repetition_penalty": -1.0},
            {"repetition_penalty": float("nan")},
            {"repetition_penalty": float("inf")},
            {"max_new_tokens": 0},
            {"max_new_tokens": -8},
            {"max_new_tokens": True},
        ],
    )
    def test_invalid_config_raises_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            BackboneConfig(**kwargs)

    def test_valid_edge_configs_and_frozen_immutability(self) -> None:
        cfg = BackboneConfig(temperature=0.0, top_p=1.0, repetition_penalty=0.5, max_new_tokens=1)
        assert cfg.model_id == "Qwen/Qwen2.5-Math-1.5B-Instruct"
        assert cfg.revision == "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
        assert cfg.dtype == "float16"
        with pytest.raises(FrozenInstanceError):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_registry_yaml_matches_get_model_metadata(self) -> None:
        registry_path = Path(__file__).resolve().parents[1] / "registry.yaml"
        with registry_path.open("r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        assert isinstance(registry, dict)
        assert isinstance(registry.get("models"), list)

        matches = [
            entry
            for entry in registry["models"]
            if entry.get("id") == "llm.qwen2_5_math_1_5b_instruct.generative"
        ]
        assert len(matches) == 1
        entry = matches[0]

        expected_fields = {
            "id": "llm.qwen2_5_math_1_5b_instruct.generative",
            "name": "Qwen2.5-Math-1.5B-Instruct",
            "modality": "text",
            "architecture": "Qwen2ForCausalLM",
            "pretraining_regime": "mixed",
            "objective": "mathematical reasoning via chain-of-thought and tool-integrated reasoning",
            "weights_source": "huggingface",
            "weights_uri": "https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct",
            "revision": "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35",
            "license": "apache-2.0",
            "num_parameters": 1540000000,
            "context_length": 32768,
            "max_generation_tokens": 8192,
            "dtype": "float16",
            "parent_backbone": "Qwen/Qwen2.5-Math-1.5B",
            "fine_tuning_dataset": "unknown",
        }
        static_meta = get_model_metadata()
        for key, value in expected_fields.items():
            assert entry[key] == value
            assert static_meta[key] == value

        custom_cfg = BackboneConfig(dtype="bfloat16", temperature=0.2, max_new_tokens=256)
        wrapper = Qwen25Math15BInstructBackbone(config=custom_cfg)
        instance_meta = wrapper.get_model_metadata()
        for key, value in expected_fields.items():
            assert instance_meta[key] == value
        assert instance_meta["loaded"] is False
        assert instance_meta["config"]["dtype"] == "bfloat16"
        assert instance_meta["config"]["temperature"] == 0.2
        assert instance_meta["config"]["max_new_tokens"] == 256


class TestBackboneWrapperBehavior:
    """Tests for preprocess, forward, generate, load, save_adaptation_state, and errors."""

    def test_unloaded_component_errors(self) -> None:
        wrapper = Qwen25Math15BInstructBackbone()
        with pytest.raises(RuntimeError, match="Tokenizer is not loaded"):
            wrapper.preprocess("2 + 2 = ?")
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            wrapper.forward({"input_ids": [[1, 2]]})
        with pytest.raises(RuntimeError, match="Model and tokenizer must be loaded"):
            wrapper.generate("2 + 2 = ?")

        tok_only = Qwen25Math15BInstructBackbone(tokenizer=_DummyTokenizer(return_torch_tensors=False))
        with pytest.raises(RuntimeError, match="Model and tokenizer must be loaded"):
            tok_only.generate("2 + 2 = ?")

        model_only = Qwen25Math15BInstructBackbone(model=_DummyCausalModel(use_torch=False))
        with pytest.raises(RuntimeError, match="Model and tokenizer must be loaded"):
            model_only.generate("2 + 2 = ?")

    def test_preprocess_string_chat_template_fallback_and_mask(self) -> None:
        tok_chat = _DummyTokenizer(
            with_chat_template=True,
            include_attention_mask=False,
            return_torch_tensors=False,
        )
        wrapper = Qwen25Math15BInstructBackbone(tokenizer=tok_chat)

        str_out = wrapper.preprocess("Solve x = 1")
        assert "input_ids" in str_out and "attention_mask" in str_out
        assert len(str_out["attention_mask"][0]) == len(str_out["input_ids"][0])
        assert all(v == 1 for v in str_out["attention_mask"][0])

        messages = [
            {"role": "system", "content": "Reason step by step."},
            {"role": "user", "content": "What is 3 + 4?"},
        ]
        chat_out = wrapper.preprocess(messages)
        assert len(tok_chat.chat_template_calls) == 1
        assert tok_chat.chat_template_calls[0]["tokenize"] is False
        assert tok_chat.chat_template_calls[0]["add_generation_prompt"] is True
        assert len(chat_out["attention_mask"][0]) == len(chat_out["input_ids"][0])

        tok_plain = _DummyTokenizer(
            with_chat_template=False,
            include_attention_mask=False,
            return_torch_tensors=False,
        )
        plain_wrapper = Qwen25Math15BInstructBackbone(tokenizer=tok_plain)
        plain_out = plain_wrapper.preprocess(messages)
        assert len(plain_out["attention_mask"][0]) == len(plain_out["input_ids"][0])

        with pytest.raises(TypeError):
            plain_wrapper.preprocess(123)  # type: ignore[arg-type]

    def test_preprocess_torch_ones_mask_derivation(self) -> None:
        torch = pytest.importorskip("torch")
        tok = _DummyTokenizer(
            with_chat_template=True,
            include_attention_mask=False,
            return_torch_tensors=True,
        )
        wrapper = Qwen25Math15BInstructBackbone(tokenizer=tok)
        out = wrapper.preprocess("x^2 = 9")
        assert isinstance(out["input_ids"], torch.Tensor)
        assert isinstance(out["attention_mask"], torch.Tensor)
        assert out["attention_mask"].shape == out["input_ids"].shape
        assert torch.equal(out["attention_mask"], torch.ones_like(out["input_ids"]))

    def test_forward_passes_inputs_to_model(self) -> None:
        model = _DummyCausalModel(use_torch=False)
        wrapper = Qwen25Math15BInstructBackbone(model=model)
        inputs = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        out = wrapper.forward(inputs)
        assert out.logits == "dummy_logits"
        assert model.forward_calls == [inputs]

    def test_generate_prompt_slicing_num_return_sequences_and_sampling(self) -> None:
        pytest.importorskip("torch")
        tok = _DummyTokenizer(with_chat_template=True, return_torch_tensors=True)
        model = _DummyCausalModel(use_torch=True)
        wrapper = Qwen25Math15BInstructBackbone(
            config=BackboneConfig(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_new_tokens=64),
            model=model,
            tokenizer=tok,
        )

        sampled = wrapper.generate(
            [{"role": "user", "content": "Compute 6*7"}],
            num_return_sequences=3,
            seed=42,
        )
        assert len(sampled) == 3
        sampled_kwargs = model.generate_calls[-1]
        assert sampled_kwargs["do_sample"] is True
        assert sampled_kwargs["temperature"] == 0.7
        assert sampled_kwargs["top_p"] == 0.8
        assert sampled_kwargs["repetition_penalty"] == 1.05
        assert sampled_kwargs["max_new_tokens"] == 64
        assert sampled_kwargs["num_return_sequences"] == 3

        prompt_len = int(sampled_kwargs["input_ids"].shape[-1])
        assert prompt_len > 0
        for item in sampled:
            assert set(item.keys()) == {"text", "num_tokens", "token_ids", "latency_ms"}
            assert item["num_tokens"] == len(item["token_ids"]) == 3
            assert item["latency_ms"] >= 0.0
            assert item["token_ids"][-1] == 2
            assert "tok2" not in item["text"].split()
        assert all(c["skip_special_tokens"] is True for c in tok.decode_calls)

        sampled_repeat = wrapper.generate(
            [{"role": "user", "content": "Compute 6*7"}],
            num_return_sequences=3,
            seed=42,
        )
        assert [r["token_ids"] for r in sampled] == [r["token_ids"] for r in sampled_repeat]

        greedy = wrapper.generate("Compute 2+2", num_return_sequences=2, temperature=0.0)
        assert len(greedy) == 2
        greedy_kwargs = model.generate_calls[-1]
        assert greedy_kwargs["do_sample"] is False
        assert "temperature" not in greedy_kwargs
        assert "top_p" not in greedy_kwargs
        assert greedy[0]["token_ids"] == [50, 51, 2]
        assert greedy[1]["token_ids"] == [51, 52, 2]
        assert greedy[0]["text"] == "tok50 tok51"

        with pytest.raises(ValueError):
            wrapper.generate("Compute 2+2", temperature=-1.0)
        with pytest.raises(ValueError):
            wrapper.generate("Compute 2+2", num_return_sequences=0)

    def test_real_torch_module_forward_and_generate(self) -> None:
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        class _TinyLM(nn.Module):
            def __init__(self, vocab_size: int = 32) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 8)
                self.head = nn.Linear(8, vocab_size)

            def forward(
                self, input_ids: Any, attention_mask: Any = None
            ) -> SimpleNamespace:
                return SimpleNamespace(logits=self.head(self.embed(input_ids)))

            def generate(
                self,
                input_ids: Any,
                attention_mask: Any = None,
                num_return_sequences: int = 1,
                max_new_tokens: int = 4,
                **kwargs: Any,
            ) -> Any:
                seq = input_ids.repeat(num_return_sequences, 1)
                for _ in range(min(2, max_new_tokens)):
                    logits = self.forward(seq).logits[:, -1, :]
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                    seq = torch.cat([seq, next_tok], dim=1)
                return seq

        tok = _DummyTokenizer(with_chat_template=False, return_torch_tensors=True)
        model = _TinyLM()
        model.eval()
        wrapper = Qwen25Math15BInstructBackbone(
            config=BackboneConfig(temperature=0.0, max_new_tokens=4),
            model=model,
            tokenizer=tok,
        )
        prepared = wrapper.preprocess("1 + 1")
        fwd_out = wrapper.forward(prepared)
        assert fwd_out.logits.shape == (1, prepared["input_ids"].shape[1], 32)

        gen_out = wrapper.generate("1 + 1", num_return_sequences=2)
        assert len(gen_out) == 2
        assert all(item["num_tokens"] == 2 for item in gen_out)
        assert all(item["latency_ms"] >= 0.0 for item in gen_out)

    def test_save_adaptation_state_writes_json_only(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        tiny_linear = torch.nn.Linear(4, 4)
        wrapper = Qwen25Math15BInstructBackbone(
            config=BackboneConfig(max_new_tokens=256, dtype="bfloat16"),
            model=tiny_linear,
            tokenizer=_DummyTokenizer(),
        )
        target = tmp_path / "nested" / "adaptation_state.json"
        returned = wrapper.save_adaptation_state(target)
        assert returned == target
        assert target.exists()

        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["metadata"]["id"] == "llm.qwen2_5_math_1_5b_instruct.generative"
        assert payload["config"]["max_new_tokens"] == 256
        assert payload["config"]["dtype"] == "bfloat16"
        assert "weight" not in payload and "state_dict" not in payload

    def test_load_missing_transformers_and_mocked_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        monkeypatch.setitem(sys.modules, "transformers", None)
        wrapper = Qwen25Math15BInstructBackbone()
        with pytest.raises(RuntimeError, match="transformers is required"):
            wrapper.load()

        calls: dict[str, Any] = {}
        fake_tok = _DummyTokenizer()
        fake_model = _DummyCausalModel(use_torch=True)

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: Any) -> _DummyTokenizer:
                calls["tokenizer"] = (model_id, kwargs)
                return fake_tok

        class _FakeAutoModel:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: Any) -> _DummyCausalModel:
                calls["model"] = (model_id, kwargs)
                return fake_model

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=_FakeAutoTokenizer,
                AutoModelForCausalLM=_FakeAutoModel,
            ),
        )

        loaded = wrapper.load(device="cpu", dtype="float32")
        assert loaded is wrapper
        assert wrapper.model is fake_model
        assert wrapper.tokenizer is fake_tok
        assert fake_model.eval_called is True
        assert fake_model.to_calls == ["cpu"]
        assert fake_tok.pad_token_id == fake_tok.eos_token_id
        assert calls["tokenizer"][0] == "Qwen/Qwen2.5-Math-1.5B-Instruct"
        assert calls["tokenizer"][1]["revision"] == "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
        assert calls["model"][1]["revision"] == "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"

        with pytest.raises(ValueError, match="Unsupported dtype"):
            wrapper.load(dtype="int4")

    def test_import_isolation_in_clean_subprocess(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
        )
        script = (
            "import sys\n"
            "import Backbones\n"
            "import Backbones.Generative_Pretrained\n"
            "from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct import (\n"
            "    BackboneConfig,\n"
            "    Qwen25Math15BInstructBackbone,\n"
            "    get_model_metadata,\n"
            ")\n"
            "from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct import wrapper\n"
            "w = Qwen25Math15BInstructBackbone(BackboneConfig())\n"
            "assert w.get_model_metadata()['id'] == get_model_metadata()['id']\n"
            "for forbidden in ('transformers', 'torch', 'Dataloaders', 'Adaptive_Stopping'):\n"
            "    assert forbidden not in sys.modules, f'{forbidden} leaked at import time'\n"
            "    assert not any(m.startswith(forbidden + '.') for m in sys.modules), f'{forbidden}.* leaked'\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            env=env,
            check=True,
        )
