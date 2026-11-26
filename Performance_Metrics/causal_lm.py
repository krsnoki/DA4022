import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from math import exp
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from Dataloaders.causal_lm import TokenBatch

REPORT_FIELDS = (
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision",
    "dataset_id", "split", "seed", "device", "framework_version", "dtype", "batch_size",
    "input_length", "output_length", "quality_metric", "quality_value", "latency_median_ms",
    "latency_p95_ms", "throughput", "peak_memory_mb", "energy_j", "status", "notes",
)


@dataclass(frozen=True)
class ZeroShotTask:
    task_id: str
    dataset_id: str
    revision: str
    split: str
    metric: str = "acc,none"
    data_files: Mapping[str, str] | None = None


@dataclass(frozen=True)
class EvalSets:
    perplexity: Mapping[str, Iterable[TokenBatch]]
    tasks: Sequence[ZeroShotTask] = ()
    harness_version: str = "0.4.8"
    max_length: int = 2048
    task_batch_size: int = 1
    seed: int = 0


@dataclass(frozen=True)
class QualityScore:
    value: float
    examples: int | None
    scored_tokens: int | None = None


@dataclass(frozen=True)
class QualityResult:
    perplexity: dict[str, QualityScore]
    zero_shot: dict[str, QualityScore]

    @property
    def mean_perplexity(self) -> float | None:
        return mean(s.value for s in self.perplexity.values()) if self.perplexity else None

    @property
    def mean_zero_shot(self) -> float | None:
        return mean(s.value for s in self.zero_shot.values()) if self.zero_shot else None


class _HarnessModel:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.config = getattr(model, "config", None) or model.model.config

    def __call__(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False)


def _zero_shot(model, tokenizer, sets: EvalSets) -> dict[str, QualityScore]:
    if tokenizer is None:
        raise ValueError("zero-shot evaluation requires a tokenizer")
    if version("lm_eval") != sets.harness_version:
        raise ValueError(f"zero-shot evaluation requires lm_eval=={sets.harness_version}")

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=_HarnessModel(model), tokenizer=tokenizer, backend="causal",
              max_length=sets.max_length, batch_size=sets.task_batch_size, add_bos_token=False)
    tasks = []
    for task in sets.tasks:
        dataset_kwargs = {"revision": task.revision, "trust_remote_code": False}
        if task.data_files is not None:
            dataset_kwargs["data_files"] = dict(task.data_files)
        tasks.append({"task": task.task_id, "dataset_path": task.dataset_id, "dataset_kwargs": dataset_kwargs})
    result = simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=0, log_samples=False, bootstrap_iters=0,
        random_seed=sets.seed, numpy_random_seed=sets.seed, torch_random_seed=sets.seed,
        fewshot_random_seed=sets.seed,
    )
    scores = {}
    samples = result["n-samples"]
    for task in sets.tasks:
        task_id = task.task_id
        metrics = result.get("groups", {}).get(task_id) or result["results"][task_id]
        count = samples.get(task_id, {}).get("effective")
        if count is None:
            members = result.get("group_subtasks", {}).get(task_id, ())
            counts = [samples.get(member, {}).get("effective") for member in members]
            if counts and all(n is not None for n in counts):
                count = sum(counts)
        scores[task_id] = QualityScore(float(metrics[task.metric]) * 100, count)
    return scores


@torch.inference_mode()
def evaluate_quality(model, tokenizer, eval_sets: EvalSets) -> QualityResult:
    model.eval()
    device = next(model.parameters()).device
    perplexity = {}
    for dataset, batches in eval_sets.perplexity.items():
        nll = 0.0
        examples = tokens = 0
        for batch in batches:
            batch = batch.to(device)
            x, mask = batch.input_ids, batch.attention_mask
            valid = mask[:, :-1] & mask[:, 1:]
            labels = x[:, 1:].masked_fill(~valid, -100)
            logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
            losses = cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction="none")
            nll += losses.sum(dtype=torch.float64).item()
            tokens += valid.sum().item()
            examples += x.shape[0]
        if not tokens:
            raise ValueError(f"{dataset} has no scored next-token positions")
        perplexity[dataset] = QualityScore(exp(nll / tokens), examples, tokens)

    zero_shot = _zero_shot(model, tokenizer, eval_sets) if eval_sets.tasks else {}
    return QualityResult(perplexity, zero_shot)


@dataclass(frozen=True)
class GenerationSettings:
    prompt_length: int = 2048
    decode_tokens: int = 128
    warmups: int = 5
    iterations: int = 30


@dataclass(frozen=True)
class LatencyResult:
    samples_ms: list[float]
    throughput: float
    peak_memory_mb: float | None

    @property
    def median_ms(self) -> float:
        return median(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        if len(self.samples_ms) == 1:
            return self.samples_ms[0]
        return quantiles(self.samples_ms, n=100, method="inclusive")[94]


@dataclass(frozen=True)
class RuntimeResult:
    prefill: LatencyResult
    decode: LatencyResult


@torch.inference_mode()
def benchmark_generation(model, token_batches: Sequence[TokenBatch], settings: GenerationSettings) -> RuntimeResult:
    cfg = settings
    if min(cfg.prompt_length, cfg.decode_tokens, cfg.iterations) < 1 or cfg.warmups < 0 or not token_batches:
        raise ValueError("generation requires positive lengths and iterations, nonnegative warmups and token batches")
    model.eval()
    device = next(model.parameters()).device
    batches = [b.to(device) for b in token_batches]
    batch_size = batches[0].input_ids.shape[0]
    length = cfg.prompt_length + cfg.decode_tokens
    if not batch_size or any(b.input_ids.shape != (batch_size, length) or not b.attention_mask.all() for b in batches):
        raise ValueError("generation requires full unpadded batches of prompt plus fixed continuation tokens")

    prefill_ms, decode_ms = [], []
    prefill_peak = decode_peak = None
    cuda = device.type == "cuda"
    for i in range(cfg.warmups + cfg.iterations):
        x, mask = batches[i % len(batches)].input_ids, batches[i % len(batches)].attention_mask
        end = cfg.prompt_length
        measure = i >= cfg.warmups
        if cuda and measure:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        out = model(input_ids=x[:, :end], attention_mask=mask[:, :end], use_cache=True, num_logits_to_keep=1)
        if cuda and measure:
            torch.cuda.synchronize(device)
        if measure:
            prefill_ms.append((perf_counter() - start) * 1000)
            if cuda:
                prefill_peak = max(prefill_peak or 0, torch.cuda.max_memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)

        cache = out.past_key_values
        del out
        start = perf_counter()
        for pos in range(end, length):
            out = model(input_ids=x[:, pos:pos + 1], attention_mask=mask[:, :pos + 1],
                        past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = out.past_key_values
            del out
        if cuda and measure:
            torch.cuda.synchronize(device)
        if measure:
            decode_ms.append((perf_counter() - start) * 1000)
            if cuda:
                decode_peak = max(decode_peak or 0, torch.cuda.max_memory_allocated(device))
        del cache

    return RuntimeResult(
        LatencyResult(prefill_ms, 1000 * batch_size * cfg.prompt_length / mean(prefill_ms),
                      None if prefill_peak is None else prefill_peak / 1_000_000),
        LatencyResult(decode_ms, 1000 * batch_size * cfg.decode_tokens / mean(decode_ms),
                      None if decode_peak is None else decode_peak / 1_000_000),
    )


def write_json_report(path: str | Path, rows: Iterable[Mapping]) -> None:
    records = []
    for row in rows:
        missing = set(REPORT_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"missing report fields: {', '.join(sorted(missing))}")
        records.append(dict(row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
