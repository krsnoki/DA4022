"""Quality and efficiency measurements shared by synthetic and real adapters."""

from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import statistics
import time
from typing import Iterable

import torch
from torch import nn


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


@torch.no_grad()
def perplexity(model: nn.Module, batches: Iterable[torch.Tensor]) -> float:
    weighted_loss, token_count = 0.0, 0
    model.eval()
    for batch in batches:
        output = model(batch, labels=batch)
        tokens = batch.numel() - batch.shape[0]
        weighted_loss += float(output.loss) * tokens
        token_count += tokens
    if token_count == 0:
        raise ValueError("evaluation received no target tokens")
    return math.exp(min(weighted_loss / token_count, 80.0))


@torch.no_grad()
def benchmark_inference(model: nn.Module, sample: torch.Tensor, warmup: int = 3, repeats: int = 20) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        model(sample)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(sample)
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    return {
        "latency_ms_median": median * 1000,
        "throughput_tokens_per_second": sample.numel() / median,
        "repeats": repeats,
        **benchmark_peak_memory(model, sample, warmup, repeats),
    }


def benchmark_peak_memory(model, sample, warmup=3, repeats=20):
    """Measure OS-reported peak resident memory in a fresh CPU process."""
    from .synthetic import TinyMoELM, TinyDenseLM
    if not isinstance(model, (TinyMoELM, TinyDenseLM)) or sample.device.type != 'cpu':
        raise ValueError('isolated CPU memory benchmark supports tiny CPU models only')
    dense = isinstance(model, TinyDenseLM)
    info = dict(vocab_size=model.vocab_size, hidden_size=model.hidden_size,
                intermediate_size=(model.ffn.gate_proj.out_features if dense else model.ffn.experts[0].gate_proj.out_features),
                num_experts=1 if dense else model.num_experts, top_k=1 if dense else model.ffn.top_k)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'memory.pt'
        torch.save(dict(state=model.state_dict(), sample=sample, info=info, dense=dense, threads=torch.get_num_threads()), path)
        result = subprocess.run([sys.executable, '-m', 'Mixture_of_Experts.moe_to_dense.scripts.memory_worker', str(path), str(warmup), str(repeats)], check=True, capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)
