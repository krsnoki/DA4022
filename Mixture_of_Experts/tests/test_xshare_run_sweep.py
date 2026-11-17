"""CPU tests for the sweep runner: energy integration, the decode loop, config and CSV rows.

No GPU, ``nvidia-smi`` or model download is needed. The decode loop is run on
a scripted stand-in model that emits a fixed token schedule, so every count
and timing field of ``generate_batch`` can be checked exactly.
"""

from __future__ import annotations

import types

import pytest
import torch

from Mixture_of_Experts.execution_layouts.xshare_hf import XShareController
from Mixture_of_Experts.experiments.xshare.run_sweep import (
    CSV_FIELDS,
    PowerSampler,
    csv_row,
    generate_batch,
    method_config,
)
from Mixture_of_Experts.routing.xshare import XShareConfig

# ------------------------------------------------------------ PowerSampler


def make_sampler(samples):
    sampler = object.__new__(PowerSampler)
    sampler.samples = list(samples)
    sampler.interval_ms = 100
    sampler.field = "power.draw.instant"
    return sampler


def sampled(power, t_end=10.0):
    """(time, watts) every 0.1 s from 0 to ``t_end``; ``power`` maps time to watts."""
    return [(i / 10, power(i / 10)) for i in range(int(round(t_end * 10)) + 1)]


def test_energy_constant_power_window():
    """Would catch a wrong trapezoid sum or window clipping for constant power."""
    sampler = make_sampler(sampled(lambda t: 100.0))
    joules, mean_w = sampler.energy(2.0, 5.0)
    assert joules == pytest.approx(300.0, abs=1e-6)
    assert mean_w == pytest.approx(100.0, abs=1e-6)


def test_energy_linear_ramp_is_exact():
    """Would catch an off-by-one at the window ends (the trapezoid is exact for a ramp)."""
    sampler = make_sampler(sampled(lambda t: 10.0 * t))
    joules, mean_w = sampler.energy(0.0, 4.0)
    assert joules == pytest.approx(80.0, abs=1e-6)
    assert mean_w == pytest.approx(20.0, abs=1e-6)


def test_energy_window_edges_between_samples():
    """Would catch dropped edge segments when the window does not fall on a sample."""
    sampler = make_sampler(sampled(lambda t: 50.0))
    joules, mean_w = sampler.energy(2.05, 4.95)
    assert joules == pytest.approx(145.0, abs=1e-6)
    assert mean_w == pytest.approx(50.0, abs=1e-6)


def test_energy_no_samples_is_none():
    """Would catch an exception or a zero instead of None when nothing was sampled."""
    assert make_sampler([]).energy(0.0, 1.0) is None


def test_energy_single_sample_is_none():
    """Would catch an energy value computed from a single point."""
    assert make_sampler([(1.0, 50.0)]).energy(0.0, 2.0) is None


def test_energy_one_sample_inside_plus_one_before_is_usable():
    """Would catch the edge value before the window being ignored."""
    sampler = make_sampler([(0.5, 40.0), (1.0, 40.0)])
    joules, _ = sampler.energy(1.0, 2.0)
    assert joules is not None


def test_energy_ignores_samples_far_outside_window():
    """Would catch far-away samples leaking into the integral (only edge values may matter)."""
    base = sampled(lambda t: 100.0)
    clean = make_sampler(base).energy(2.0, 5.0)
    spiky = make_sampler([(-5.0, 1e6)] + base + [(20.0, 1e6)]).energy(2.0, 5.0)
    assert spiky == pytest.approx(clean)


# ------------------------------------------------------------ generate_batch


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=False,
                 add_special_tokens=False):
        ids = [[2 + (ord(c) % 50) for c in prompt] for prompt in prompts]
        longest = max(len(row) for row in ids)
        rows, masks = [], []
        for row in ids:
            fill = longest - len(row)
            if self.padding_side == "left":
                rows.append([0] * fill + row)
                masks.append([0] * fill + [1] * len(row))
            else:
                rows.append(row + [0] * fill)
                masks.append([1] * len(row) + [0] * fill)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [" ".join(str(t) for t in row.tolist() if t not in (0, 1)) for row in tokens]


class ScriptedLM(torch.nn.Module):
    """Emits ``schedule[:, call_number]`` as the greedy next token."""

    def __init__(self, schedule, vocab=64):
        super().__init__()
        self.schedule = schedule
        self.vocab = vocab
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        step = min(self.calls, self.schedule.shape[1] - 1)
        for b in range(batch):
            logits[b, -1, int(self.schedule[b, step])] = 1.0
        self.calls += 1
        return types.SimpleNamespace(logits=logits, past_key_values=None)


PROMPTS = ["ab", "abcd", "a"]
EOS_IDS = {1}


def run(schedule_rows, max_new_tokens):
    schedule = torch.tensor(schedule_rows, dtype=torch.long)
    lm = ScriptedLM(schedule)
    backbone = types.SimpleNamespace(model=lm, tokenizer=StubTokenizer(), device=torch.device("cpu"))
    controller = XShareController(None, n_layers=1)
    out = generate_batch(backbone, controller, PROMPTS, max_new_tokens, EOS_IDS)
    return out, lm, controller


MAIN_SCHEDULE = [[5, 6, 1, 7, 7], [5, 1, 9, 9, 9], [5, 6, 7, 8, 9]]


def test_generate_batch_counts_and_texts():
    """Would catch wrong per-row token counts, live-token totals or text after EOS."""
    out, _, _ = run(MAIN_SCHEDULE, 4)
    assert out["new_tokens"] == [2, 1, 4]
    assert out["texts"] == ["5 6", "5", "5 6 7 8"]
    assert len(out["step_ms"]) == 3
    assert out["live_tokens"] == 3 + 2 + 1
    assert out["prompt_tokens"] == 2 + 4 + 1


def test_generate_batch_timings_are_consistent():
    """Would catch latencies out of order with finish time, or negative durations."""
    out, _, _ = run(MAIN_SCHEDULE, 4)
    lat = out["latency_s"]
    assert len(lat) == 3
    assert all(x > 0 for x in lat)
    assert lat[1] <= lat[0] <= lat[2]
    assert out["prefill_s"] >= 0
    assert out["decode_s"] >= 0
    assert out["t_end"] >= out["t_start"]
    assert all(ms >= 0 for ms in out["step_ms"])


def test_generate_batch_clears_valid_rows():
    """Would catch the controller keeping stale valid_rows after the batch ends."""
    _, _, controller = run(MAIN_SCHEDULE, 4)
    assert controller.valid_rows is None


def test_generate_batch_no_wasted_forward_pass():
    """Would catch an extra model call after the last token is already known."""
    out, lm, _ = run(MAIN_SCHEDULE, 4)
    assert lm.calls == len(out["step_ms"]) + 1


def test_generate_batch_all_eos_at_prefill():
    """Would catch decode steps being run when every row already finished."""
    out, lm, _ = run([[1] * 5] * 3, 4)
    assert out["step_ms"] == []
    assert out["live_tokens"] == 0
    assert out["new_tokens"] == [0, 0, 0]
    assert out["texts"] == ["", "", ""]
    assert lm.calls == 1


def test_generate_batch_single_new_token_runs_no_decode():
    """Would catch a decode forward pass when max_new_tokens is 1."""
    out, lm, _ = run([[5, 6, 7]] * 3, 1)
    assert out["step_ms"] == []
    assert out["live_tokens"] == 0
    assert out["new_tokens"] == [1, 1, 1]
    assert out["texts"] == ["5", "5", "5"]
    assert lm.calls == len(out["step_ms"]) + 1 == 1


def test_generate_batch_uses_left_padding_and_restores_side():
    """Would catch right-padded prompts or a tokenizer left in the wrong padding mode."""
    schedule = torch.tensor(MAIN_SCHEDULE, dtype=torch.long)
    tok = StubTokenizer()
    backbone = types.SimpleNamespace(
        model=ScriptedLM(schedule), tokenizer=tok, device=torch.device("cpu")
    )
    generate_batch(backbone, XShareController(None, n_layers=1), PROMPTS, 2, EOS_IDS)
    assert tok.padding_side == "right"


# ------------------------------------------------------------ method_config


def test_method_config_baseline_is_none():
    """Would catch the baseline (no budget) being given a pruning configuration."""
    assert method_config({"name": "baseline"}, {}, 8, 64, "softmax_topk") is None


def test_method_config_budget_all_means_every_expert():
    """Would catch 'all' not being resolved to the number of experts."""
    cfg = method_config({"name": "keep_all", "budget": "all"}, {}, 8, 64, "softmax_topk")
    assert isinstance(cfg, XShareConfig)
    assert cfg.budget == 64
    assert cfg.top_k == 8


def test_method_config_passes_values_through():
    """Would catch warm-up, defaults or gate_norm being dropped or overridden."""
    defaults = {"budget_mode": "total", "score": "logits", "min_keep": 3}
    cfg = method_config(
        {"name": "x", "budget": "12", "warmup_k0": 2}, defaults, 8, 64, "softmax_topk_renorm"
    )
    assert cfg.budget == 12
    assert cfg.warmup_k0 == 2
    assert cfg.budget_mode == "total"
    assert cfg.score == "logits"
    assert cfg.min_keep == 3
    assert cfg.gate_norm == "softmax_topk_renorm"


def test_method_config_defaults_when_unspecified():
    """Would catch wrong fallback values for warm-up, budget mode, score and min_keep."""
    cfg = method_config({"name": "x", "budget": 5}, {}, 8, 64, "softmax_topk")
    assert cfg.warmup_k0 == 0
    assert cfg.budget_mode == "additional"
    assert cfg.score == "softmax"
    assert cfg.min_keep is None
    assert cfg.gate_norm == "softmax_topk"


# ------------------------------------------------------------------ csv_row

CFG = {
    "paper_id": "xshare-2025",
    "dataset": {"id": "text.gsm8k.test", "split": "test", "seed": 42},
    "generation": {"max_new_tokens": 256, "warmup_tokens": 8},
}


def make_result(energy):
    return {
        "experiment_id": "xshare-gsm8k-baseline-c4-s0",
        "method": "baseline",
        "commit": "abc123",
        "subset": 0,
        "batch_size": 4,
        "prompt_tokens": 800,
        "num_examples": 8,
        "accuracy": 0.375,
        "decode_step_median_ms": 12.3456,
        "decode_step_p95_ms": 15.0,
        "decode_tokens_per_s": 123.456,
        "peak_memory_allocated_mb": 4096.25,
        "request_latency_median_s": 2.5,
        "request_latency_p95_s": 4.5,
        "prefill_s_total": 1.5,
        "routing": {"mean_activated_experts": 20.0, "mean_kept_experts": 64.0},
        "energy": energy,
        "metadata": {
            "backbone_id": "allenai/OLMoE-1B-7B",
            "revision": "rev",
            "device": "gpu0",
            "driver": "550",
            "torch": "2.5",
            "cuda_runtime": "12.4",
            "transformers": "5.0",
            "attn_implementation": "sdpa",
            "experts_implementation": "eager",
            "dtype": "bfloat16",
            "n_experts": 64,
        },
    }


def make_energy(above):
    return {
        "source": "nvidia-smi power.draw.instant, every 100 ms, whole board",
        "joules": 1234.56,
        "idle_w": 60.0,
        "j_per_token": 0.5,
        "j_per_token_above_idle": above,
    }


def test_csv_row_keys_match_schema_without_energy():
    """Would catch a row whose keys drift from CSV_FIELDS (DictWriter would then fail)."""
    row = csv_row(make_result(None), CFG)
    assert list(row) == CSV_FIELDS
    assert set(row) == set(CSV_FIELDS)


def test_csv_row_without_energy():
    """Would catch an energy value or missing 'not_measured' note when energy is None."""
    row = csv_row(make_result(None), CFG)
    assert row["status"] == "partial"
    assert row["energy_j"] == ""
    assert "energy not_measured" in row["notes"]


def test_csv_row_with_energy():
    """Would catch a non-numeric energy_j or a missing J/token note."""
    row = csv_row(make_result(make_energy(0.2)), CFG)
    assert list(row) == CSV_FIELDS
    assert isinstance(row["energy_j"], (int, float))
    assert row["energy_j"] == pytest.approx(1234.6)
    assert "J/token" in row["notes"]
    assert "above idle" in row["notes"]
    assert "energy not_measured" not in row["notes"]


def test_csv_row_energy_without_above_idle_does_not_raise():
    """Would catch a crash when idle power was unavailable (above-idle value is None)."""
    row = csv_row(make_result(make_energy(None)), CFG)
    assert "J/token" in row["notes"]
    assert "above idle" not in row["notes"]
    assert row["energy_j"] == pytest.approx(1234.6)


def test_csv_row_fields_from_result_and_config():
    """Would catch swapped or mis-derived columns (input length, split, quality value)."""
    row = csv_row(make_result(None), CFG)
    assert row["paper_id"] == "xshare-2025"
    assert row["input_length"] == 100.0
    assert row["output_length"] == 256
    assert row["quality_value"] == 0.375
    assert row["split"] == "test (subset 0 of seed 42)"
    assert row["batch_size"] == 4
    assert row["latency_median_ms"] == 12.346
    assert row["throughput"] == 123.46
