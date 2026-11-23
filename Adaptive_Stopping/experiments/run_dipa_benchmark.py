"""Checkpointed DIPA benchmark runner for MATH-500 evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
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
    GenerationAttempt,
    ProblemInstance,
)
from Adaptive_Stopping.methods.dipa.verifier import verify_math_answer
from Dataloaders.evaluation.math500 import (
    DEFAULT_SYSTEM_PROMPT,
    Math500LoaderConfig,
    load_math500,
)
from Performance_Metrics import (
    VALID_STATUSES,
    MetricsRow,
    append_metrics_rows,
    summarize_latencies_and_throughput,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "Adaptive_Stopping" / "configs" / "dipa.yaml"
DEFAULT_OUTPUT_CSV = Path("Performance_Metrics/results/dipa_math500.csv")
DEFAULT_METHODS: tuple[str, ...] = (
    "uniform",
    "easy2hard",
    "hard2easy",
    "dipa_mgl",
    "dipa_mgc",
)
VALID_METHODS: frozenset[str] = frozenset(DEFAULT_METHODS)


def _seed_from_key(*parts: object) -> int:
    """Derive a cross-process deterministic integer seed from arbitrary key parts."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _detect_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else "unknown"


def _detect_framework_version(mode: str) -> str:
    parts = [
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        f"numpy={np.__version__}",
    ]
    if mode == "hf":
        try:
            import torch

            parts.append(f"torch={torch.__version__}")
        except ImportError:
            pass
        try:
            import transformers

            parts.append(f"transformers={transformers.__version__}")
        except ImportError:
            pass
    return ";".join(parts)


def _resolve_device(device: str | None, mode: str) -> str:
    if device is not None and device != "auto":
        return device
    if mode != "hf":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            try:
                return f"cuda:{torch.cuda.get_device_name(0)}"
            except Exception:
                return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file() and not path.is_absolute():
        candidate = REPO_ROOT / path
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config {path}, got {type(data).__name__}")
    return data


def load_pool_jsonl(pool_path: Path) -> dict[str, dict[str, Any]]:
    """Load checkpointed per-instance candidate pools keyed by ``instance_id``."""
    path = Path(pool_path)
    if not path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict) or "instance_id" not in parsed:
                raise ValueError(
                    f"Malformed pool entry at line {line_no} of {path}"
                )
            iid = str(parsed["instance_id"])
            prev = entries.get(iid)
            # Later lines from resumed runs supersede incomplete entries for the same ID.
            if prev is None or len(parsed.get("attempts", [])) >= len(
                prev.get("attempts", [])
            ):
                entries[iid] = parsed
    return entries


def append_pool_entry(pool_path: Path, entry: dict[str, Any]) -> None:
    """Append one instance pool entry as JSONL and flush immediately for resume safety."""
    path = Path(pool_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()


def _generate_mock_attempts(
    record: Mapping[str, Any],
    start_idx: int,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Synthesize deterministic attempts correlated with MATH level / prompt length."""
    iid = str(record["instance_id"])
    prompt = str(record["prompt"])
    ground_truth = str(record["ground_truth"])
    metadata = record.get("metadata", {})
    raw_level = metadata.get("level") if isinstance(metadata, Mapping) else None
    if raw_level is not None:
        level = float(raw_level)
    else:
        level = min(5.0, max(1.0, len(prompt.split()) / 6.0))

    hard_frac = min(1.0, max(0.0, (level - 1.0) / 4.0))
    prompt_len_bias = min(0.08, len(prompt.split()) * 0.002)
    p_solve = max(0.08, 0.78 - 0.62 * hard_frac - prompt_len_bias)
    base_tokens = int(40 + 220 * hard_frac + min(40, len(prompt.split())))
    num_distractors = 2 if level <= 2.0 else 6

    rng = np.random.default_rng(_seed_from_key("mock_pool", seed, iid))
    attempts: list[dict[str, Any]] = []
    for attempt_idx in range(target_count):
        solve_roll = float(rng.random())
        tok_jitter = int(rng.integers(-10, 16))
        distractor_idx = int(rng.integers(0, num_distractors))
        lat_factor = 0.45 + 0.15 * float(rng.random())
        if attempt_idx < start_idx:
            continue

        num_tokens = max(10, base_tokens + tok_jitter)
        latency_ms = round(float(num_tokens * lat_factor), 3)
        if solve_roll < p_solve:
            ans = ground_truth
        else:
            ans = f"incorrect_{distractor_idx}"
        text = f"Mock reasoning for {iid} attempt {attempt_idx}. \\boxed{{{ans}}}"
        is_correct, extracted = verify_math_answer(text, ground_truth)
        attempts.append(
            {
                "text": text,
                "num_tokens": num_tokens,
                "latency_ms": latency_ms,
                "extracted_answer": extracted,
                "is_correct": bool(is_correct),
            }
        )
    return attempts


def _load_default_backbone(
    config_dict: Mapping[str, Any],
    device: str | None = None,
    dtype: str | None = None,
    backbone_revision: str = "main",
) -> Any:
    """Lazy-import and load the Qwen2.5-Math-1.5B-Instruct backbone wrapper."""
    from Backbones.Generative_Pretrained.qwen2_5_math_1_5b_instruct import (
        BackboneConfig,
        Qwen25Math15BInstructBackbone,
    )

    gen_cfg = config_dict.get("generation", {})
    cfg = BackboneConfig(
        model_id=str(config_dict.get("backbone_id", "Qwen/Qwen2.5-Math-1.5B-Instruct")),
        revision=backbone_revision,
        device=device or "auto",
        dtype=dtype or "float16",
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.8)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
    )
    return Qwen25Math15BInstructBackbone(cfg).load()


def build_response_pool(
    records: Sequence[Mapping[str, Any]],
    mode: str,
    pool_size: int,
    pool_path: str | Path | None = None,
    backbone: Any = None,
    gen_batch_size: int = 4,
    seed: int = 42,
) -> tuple[dict[str, dict[str, Any]], float | None]:
    """Generate or resume a checkpointed response pool across ``records``."""
    if mode not in {"hf", "mock"}:
        raise ValueError(f"mode must be 'hf' or 'mock', got {mode!r}")
    if isinstance(pool_size, bool) or pool_size < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size!r}")
    if isinstance(gen_batch_size, bool) or gen_batch_size <= 0:
        raise ValueError(f"gen_batch_size must be > 0, got {gen_batch_size!r}")

    resolved_pool_path = Path(pool_path) if pool_path is not None else None
    existing = (
        load_pool_jsonl(resolved_pool_path)
        if resolved_pool_path is not None and resolved_pool_path.exists()
        else {}
    )

    missing_indices = [
        idx
        for idx, rec in enumerate(records)
        if len(existing.get(str(rec["instance_id"]), {}).get("attempts", []))
        < pool_size
    ]

    torch_mod: Any = None
    cuda_available = False
    if missing_indices and mode == "hf":
        if backbone is None:
            backbone = _load_default_backbone({})
        elif (
            getattr(backbone, "model", "SENTINEL") is None
            and hasattr(backbone, "load")
        ):
            backbone.load()
        try:
            import torch as torch_mod

            cuda_available = bool(torch_mod.cuda.is_available())
        except ImportError:
            torch_mod = None
        if cuda_available and torch_mod is not None:
            torch_mod.cuda.reset_peak_memory_stats()

    pool_entries: dict[str, dict[str, Any]] = {}
    for rec_idx, record in enumerate(records):
        iid = str(record["instance_id"])
        prev = existing.get(iid)
        prev_attempts = list(prev.get("attempts", [])) if prev is not None else []
        if prev is not None and len(prev_attempts) >= pool_size:
            pool_entries[iid] = prev
            continue

        attempts = list(prev_attempts)
        if mode == "mock":
            attempts.extend(
                _generate_mock_attempts(
                    record,
                    start_idx=len(attempts),
                    target_count=pool_size,
                    seed=seed,
                )
            )
        else:
            prompt_input = record.get("messages", record["prompt"])
            ground_truth = str(record["ground_truth"])
            chunk_idx = 0
            while len(attempts) < pool_size:
                chunk = min(gen_batch_size, pool_size - len(attempts))
                chunk_seed = seed + rec_idx * 1000 + chunk_idx
                raw_outputs = backbone.generate(
                    prompt_input,
                    num_return_sequences=chunk,
                    seed=chunk_seed,
                )
                for out in raw_outputs:
                    text = str(out.get("text", ""))
                    is_correct, extracted = verify_math_answer(text, ground_truth)
                    attempts.append(
                        {
                            "text": text,
                            "num_tokens": int(out.get("num_tokens", 0)),
                            "latency_ms": float(out.get("latency_ms", 0.0)),
                            "extracted_answer": extracted,
                            "is_correct": bool(is_correct),
                        }
                    )
                chunk_idx += 1

        entry: dict[str, Any] = {
            "instance_id": iid,
            "prompt": str(record["prompt"]),
            "ground_truth": str(record["ground_truth"]),
            "metadata": dict(record.get("metadata", {})),
            "attempts": attempts,
        }
        if resolved_pool_path is not None:
            append_pool_entry(resolved_pool_path, entry)
        pool_entries[iid] = entry

    peak_memory_mb: float | None = None
    if missing_indices and mode == "hf" and cuda_available and torch_mod is not None:
        peak_memory_mb = float(torch_mod.cuda.max_memory_allocated() / (1024**2))

    return pool_entries, peak_memory_mb


def _build_allocator(method: str, cfg: AllocationConfig) -> BudgetAllocator:
    if method == "uniform":
        return UniformAllocator(cfg, eliminate_solved=False)
    if method == "easy2hard":
        return Easy2HardAllocator(cfg)
    if method == "hard2easy":
        return Hard2EasyAllocator(cfg)
    if method == "dipa_mgl":
        return DIPABudgetAllocator(replace(cfg, gen_proxy="mgl"))
    if method == "dipa_mgc":
        # MGC has no input-based counterpart in Table 1; 1/MGC >= 1.0 needs a unit prior to avoid mql scale lock-in.
        return DIPABudgetAllocator(replace(cfg, input_proxy="const", gen_proxy="mgc"))
    available = ", ".join(sorted(VALID_METHODS))
    raise ValueError(f"Unknown method {method!r}. Expected one of: {available}")


def evaluate_pool(
    pool_entries: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str],
    budget_multipliers: Sequence[float],
    seeds: Sequence[int],
    config_dict: Mapping[str, Any],
    status: str = "partial",
    commit: str = "unknown",
    device: str = "cpu",
    framework_version: str = "unknown",
    dtype: str = "unknown",
    backbone_revision: str = "main",
    peak_memory_mb: float | None = None,
    gen_batch_size: int = 1,
    reverify: bool = False,
) -> list[MetricsRow]:
    """Evaluate all requested methods across shared instances, budgets, and seeds."""
    for method in methods:
        if method not in VALID_METHODS:
            available = ", ".join(sorted(VALID_METHODS))
            raise ValueError(f"Unknown method {method!r}. Expected one of: {available}")

    if reverify:
        reverified_entries: dict[str, dict[str, Any]] = {}
        for iid, entry in pool_entries.items():
            gt = str(entry["ground_truth"])
            new_attempts: list[dict[str, Any]] = []
            for attempt in entry.get("attempts", []):
                text = str(attempt.get("text", ""))
                is_correct, extracted = verify_math_answer(text, gt)
                updated = dict(attempt)
                updated["is_correct"] = bool(is_correct)
                updated["extracted_answer"] = extracted
                new_attempts.append(updated)
            copied_entry = dict(entry)
            copied_entry["attempts"] = new_attempts
            reverified_entries[iid] = copied_entry
        active_pool: Mapping[str, Mapping[str, Any]] = reverified_entries
    else:
        active_pool = pool_entries

    instances = [
        ProblemInstance(
            instance_id=str(entry["instance_id"]),
            prompt=str(entry["prompt"]),
            ground_truth=str(entry["ground_truth"]),
            metadata=dict(entry.get("metadata", {})),
        )
        for entry in active_pool.values()
    ]

    alloc_cfg = config_dict.get("allocation", {})
    gen_cfg = config_dict.get("generation", {})
    backbone_id = str(config_dict.get("backbone_id", "Qwen/Qwen2.5-Math-1.5B-Instruct"))
    dataset_id = str(config_dict.get("dataset_id", "HuggingFaceH4/MATH-500"))
    split = str(config_dict.get("split", "test"))
    input_proxy = str(alloc_cfg.get("input_proxy", "mql"))
    default_gen_proxy = str(alloc_cfg.get("gen_proxy", "mgl"))
    lambda_mode = str(alloc_cfg.get("lambda_mode", "dynamic"))
    lambda_value = float(alloc_cfg.get("lambda_value", 1.0))
    min_difficulty = float(alloc_cfg.get("min_difficulty", 1e-6))
    output_length = (
        int(gen_cfg["max_new_tokens"]) if "max_new_tokens" in gen_cfg else None
    )

    pool_sizes = [len(active_pool[inst.instance_id].get("attempts", [])) for inst in instances]
    if not pool_sizes:
        pool_size_desc = "0"
    elif len(set(pool_sizes)) == 1:
        pool_size_desc = str(pool_sizes[0])
    else:
        pool_size_desc = f"{min(pool_sizes)}-{max(pool_sizes)}"

    rows: list[MetricsRow] = []
    for mult in budget_multipliers:
        mult_f = float(mult)
        total_budget = 0 if not instances else max(1, round(mult_f * len(instances)))

        for seed in seeds:
            seed_i = int(seed)
            permutations: dict[str, np.ndarray] = {}
            for inst in instances:
                n_attempts = len(active_pool[inst.instance_id].get("attempts", []))
                if total_budget > 0 and n_attempts == 0:
                    raise ValueError(
                        f"Instance {inst.instance_id!r} has 0 pooled attempts"
                    )
                if n_attempts > 0:
                    perm_rng = np.random.default_rng(
                        _seed_from_key("eval_perm", seed_i, inst.instance_id)
                    )
                    permutations[inst.instance_id] = perm_rng.permutation(n_attempts)

            for method in methods:
                cfg = AllocationConfig(
                    total_budget=total_budget,
                    input_proxy=input_proxy,
                    gen_proxy=default_gen_proxy,
                    lambda_mode=lambda_mode,
                    lambda_value=lambda_value,
                    min_difficulty=min_difficulty,
                    seed=seed_i,
                )
                allocator = _build_allocator(method, cfg)
                drawn_attempts: list[GenerationAttempt] = []
                wrapped_count = 0

                def _pool_generate(
                    inst: ProblemInstance,
                    attempt_idx: int,
                    rng: np.random.Generator,
                    _permutations: dict[str, np.ndarray] = permutations,
                ) -> GenerationAttempt:
                    nonlocal wrapped_count
                    attempts = active_pool[inst.instance_id]["attempts"]
                    perm = _permutations[inst.instance_id]
                    if attempt_idx >= len(attempts):
                        wrapped_count += 1
                    chosen = attempts[int(perm[attempt_idx % len(attempts)])]
                    text = str(chosen.get("text", ""))
                    if "is_correct" in chosen:
                        is_correct = bool(chosen["is_correct"])
                        extracted = chosen.get("extracted_answer")
                    else:
                        is_correct, extracted = verify_math_answer(
                            text, inst.ground_truth
                        )
                    attempt = GenerationAttempt(
                        text=text,
                        extracted_answer=extracted,
                        num_tokens=int(chosen.get("num_tokens", 0)),
                        is_correct=is_correct,
                        latency_ms=float(chosen.get("latency_ms", 0.0)),
                    )
                    drawn_attempts.append(attempt)
                    return attempt

                report = allocator.allocate(instances, _pool_generate, verifier_fn=None)
                if wrapped_count > 0:
                    logger.warning(
                        "Method %s (mult=%g, seed=%d) wrapped pool %d time(s)",
                        method,
                        mult_f,
                        seed_i,
                        wrapped_count,
                    )

                med_ms, p95_ms, throughput = summarize_latencies_and_throughput(
                    [attempt.latency_ms for attempt in drawn_attempts],
                    report.total_tokens,
                )
                note_parts = [
                    f"budget_multiplier={mult_f:g}",
                    f"budget_used={report.budget_used}/{report.total_budget}",
                    f"pool_size={pool_size_desc}",
                    f"num_instances={report.num_instances}",
                    f"num_solved={report.num_solved}",
                    f"total_tokens={report.total_tokens}",
                    f"lambda_mode={lambda_mode}",
                    f"lambda_value={lambda_value:g}",
                ]
                if wrapped_count > 0:
                    note_parts.append(f"pool_wrapped={wrapped_count}")
                if reverify:
                    note_parts.append("reverified=true")

                rows.append(
                    MetricsRow(
                        experiment_id=f"dipa_{method}_b{mult_f:g}_s{seed_i}",
                        paper_id="DIPA",
                        method=method,
                        commit=commit,
                        backbone_id=backbone_id,
                        backbone_revision=backbone_revision,
                        dataset_id=dataset_id,
                        split=split,
                        seed=seed_i,
                        device=device,
                        framework_version=framework_version,
                        dtype=dtype,
                        batch_size=int(gen_batch_size),
                        input_length=None,
                        output_length=output_length,
                        quality_metric="coverage",
                        quality_value=float(report.coverage),
                        latency_median_ms=med_ms,
                        latency_p95_ms=p95_ms,
                        throughput=throughput,
                        peak_memory_mb=peak_memory_mb,
                        energy_j="not_measured",
                        status=status,
                        notes="; ".join(note_parts),
                    )
                )
    return rows


def _extract_budget_multiplier(notes: str) -> str:
    for part in notes.split(";"):
        stripped = part.strip()
        if stripped.startswith("budget_multiplier="):
            return stripped.split("=", 1)[1]
    return "?"


def format_summary_table(rows: Sequence[MetricsRow]) -> str:
    """Format mean coverage by method and budget multiplier across seeds."""
    if not rows:
        return "No benchmark rows generated."

    methods_order: list[str] = []
    mults_order: list[str] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.method not in methods_order:
            methods_order.append(row.method)
        mult_key = _extract_budget_multiplier(row.notes)
        if mult_key not in mults_order:
            mults_order.append(mult_key)
        if row.quality_value is not None:
            grouped.setdefault((row.method, mult_key), []).append(float(row.quality_value))

    method_w = max(len("Method"), *(len(m) for m in methods_order))
    col_headers = [f"b={m}" for m in mults_order]
    col_w = max(7, *(len(h) for h in col_headers))

    header = f"{'Method':<{method_w}} | " + " | ".join(
        f"{h:>{col_w}}" for h in col_headers
    )
    sep = "-" * method_w + "-+-" + "-+-".join("-" * col_w for _ in col_headers)
    lines = [header, sep]
    for method in methods_order:
        cells: list[str] = []
        for mult_key in mults_order:
            vals = grouped.get((method, mult_key), [])
            cell = f"{sum(vals) / len(vals):.3f}" if vals else "n/a"
            cells.append(f"{cell:>{col_w}}")
        lines.append(f"{method:<{method_w}} | " + " | ".join(cells))
    return "\n".join(lines)


def run_benchmark(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    mode: str = "hf",
    local_jsonl: str | Path | None = None,
    num_samples: int | None = None,
    pool_size: int = 8,
    gen_batch_size: int = 4,
    pool_path: str | Path | None = None,
    eval_only: bool = False,
    reverify: bool = False,
    methods: Sequence[str] = DEFAULT_METHODS,
    budget_multipliers: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    lambda_mode: str | None = None,
    lambda_value: float | None = None,
    output_csv: str | Path | None = DEFAULT_OUTPUT_CSV,
    overwrite_csv: bool = False,
    status: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    backbone_revision: str = "main",
    backbone: Any = None,
    print_summary: bool = True,
) -> list[MetricsRow]:
    """Run DIPA pool construction (or resume/eval-only) and emit CSV metrics."""
    if mode not in {"hf", "mock"}:
        raise ValueError(f"mode must be 'hf' or 'mock', got {mode!r}")
    for method in methods:
        if method not in VALID_METHODS:
            available = ", ".join(sorted(VALID_METHODS))
            raise ValueError(f"Unknown method {method!r}. Expected one of: {available}")

    resolved_status = (
        status
        if status is not None
        else ("smoke_test" if mode == "mock" else "partial")
    )
    if resolved_status not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise ValueError(
            f"Invalid status {resolved_status!r}. Expected one of: {allowed}"
        )
    if isinstance(lambda_value, bool):
        raise ValueError(f"lambda_value must be > 0, got {lambda_value!r}")
    if lambda_mode is not None or lambda_value is not None:
        AllocationConfig(
            total_budget=0,
            lambda_mode=lambda_mode if lambda_mode is not None else "dynamic",
            lambda_value=float(lambda_value) if lambda_value is not None else 1.0,
        )

    config_dict = dict(_load_yaml_config(config_path))
    raw_alloc = config_dict.get("allocation")
    alloc_cfg = dict(raw_alloc) if isinstance(raw_alloc, Mapping) else {}
    if lambda_mode is not None:
        alloc_cfg["lambda_mode"] = lambda_mode
    if lambda_value is not None:
        alloc_cfg["lambda_value"] = float(lambda_value)
    config_dict["allocation"] = alloc_cfg
    prompt_cfg = config_dict.get("prompt_template", {})

    resolved_multipliers = (
        [float(m) for m in budget_multipliers]
        if budget_multipliers is not None
        else [float(m) for m in alloc_cfg.get("budget_multipliers", [0.5, 1, 2, 4, 8])]
    )
    resolved_seeds = (
        [int(s) for s in seeds]
        if seeds is not None
        else [int(s) for s in alloc_cfg.get("seeds", [42, 43, 44])]
    )

    if eval_only:
        if pool_path is None:
            raise ValueError("pool_path is required when eval_only=True")
        resolved_pool_path = Path(pool_path)
        if not resolved_pool_path.is_file():
            raise FileNotFoundError(
                f"Pool checkpoint not found for eval_only: {resolved_pool_path}"
            )
        pool_entries = load_pool_jsonl(resolved_pool_path)
        if num_samples is not None:
            pool_entries = dict(list(pool_entries.items())[:num_samples])
        peak_memory_mb: float | None = None
    else:
        subsample_seed = resolved_seeds[0] if resolved_seeds else 42
        loader_cfg = Math500LoaderConfig(
            dataset_id=str(config_dict.get("dataset_id", "HuggingFaceH4/MATH-500")),
            split=str(config_dict.get("split", "test")),
            num_samples=num_samples,
            seed=subsample_seed,
            system_prompt=str(prompt_cfg.get("system", DEFAULT_SYSTEM_PROMPT)),
            local_jsonl_path=local_jsonl,
        )
        records = load_math500(loader_cfg)

        active_backbone = backbone
        if mode == "hf" and active_backbone is None:
            resolved_pool_file = Path(pool_path) if pool_path is not None else None
            existing = (
                load_pool_jsonl(resolved_pool_file)
                if resolved_pool_file is not None and resolved_pool_file.exists()
                else {}
            )
            needs_gen = any(
                len(existing.get(str(r["instance_id"]), {}).get("attempts", []))
                < pool_size
                for r in records
            )
            if needs_gen:
                active_backbone = _load_default_backbone(
                    config_dict,
                    device=device,
                    dtype=dtype,
                    backbone_revision=backbone_revision,
                )

        pool_entries, peak_memory_mb = build_response_pool(
            records=records,
            mode=mode,
            pool_size=pool_size,
            pool_path=pool_path,
            backbone=active_backbone,
            gen_batch_size=gen_batch_size,
            seed=subsample_seed,
        )

    effective_mode = "eval_only" if eval_only else mode
    resolved_device = _resolve_device(device, effective_mode)
    if dtype is not None:
        resolved_dtype = dtype
    elif eval_only:
        resolved_dtype = "unknown"
    elif mode == "hf":
        resolved_dtype = "float16"
    else:
        resolved_dtype = "none"
    rows = evaluate_pool(
        pool_entries=pool_entries,
        methods=methods,
        budget_multipliers=resolved_multipliers,
        seeds=resolved_seeds,
        config_dict=config_dict,
        status=resolved_status,
        commit=_detect_git_commit(),
        device=resolved_device,
        framework_version=_detect_framework_version(effective_mode),
        dtype=resolved_dtype,
        backbone_revision=backbone_revision,
        peak_memory_mb=peak_memory_mb,
        gen_batch_size=gen_batch_size,
        reverify=reverify,
    )

    if output_csv is not None:
        append_metrics_rows(output_csv, rows, overwrite=overwrite_csv)
    if print_summary:
        print(format_summary_table(rows))
    return rows


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DIPA checkpointed MATH-500 benchmark and emit CSV metrics."
    )
    parser.add_argument(
        "--config",
        default="Adaptive_Stopping/configs/dipa.yaml",
        help="Path to DIPA YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=("hf", "mock"),
        default="hf",
        help="Generation backend mode ('hf' for Qwen backbone, 'mock' for CPU).",
    )
    parser.add_argument(
        "--local-jsonl",
        default=None,
        help="Optional local MATH-500 JSONL path instead of Hugging Face Datasets.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional number of subsampled MATH-500 problems.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=8,
        help="Number of candidate responses per problem in the shared pool.",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=4,
        help="Number of sequences requested per backbone.generate() call.",
    )
    parser.add_argument(
        "--pool-path",
        default=None,
        help="Path to JSONL checkpoint pool file for incremental resume/eval.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip model loading and generation; evaluate an existing --pool-path.",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="Re-run verify_math_answer on pooled attempts before evaluation.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated list of allocator methods to compare.",
    )
    parser.add_argument(
        "--budget-multipliers",
        default=None,
        help="Comma-separated budget multipliers overriding config.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated integer seeds overriding config.",
    )
    parser.add_argument(
        "--lambda-mode",
        choices=("fixed", "dynamic"),
        default=None,
        help="Override allocation.lambda_mode ('fixed' or 'dynamic').",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=None,
        help="Override allocation.lambda_value (must be > 0).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Destination CSV path conforming to Performance_Metrics schema.",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Overwrite --output-csv instead of appending.",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Override MetricsRow status (reproduced, partial, approximate, not_reproduced, smoke_test).",
    )
    parser.add_argument("--device", default=None, help="Device override (e.g. cuda, cpu).")
    parser.add_argument("--dtype", default=None, help="Dtype override (e.g. float16, bfloat16).")
    parser.add_argument(
        "--backbone-revision",
        default="main",
        help="Pinned Hugging Face model revision SHA or branch.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for DIPA benchmark execution."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    methods = _parse_csv_list(args.methods)
    budget_multipliers = (
        [float(x) for x in _parse_csv_list(args.budget_multipliers)]
        if args.budget_multipliers is not None
        else None
    )
    seeds = (
        [int(x) for x in _parse_csv_list(args.seeds)]
        if args.seeds is not None
        else None
    )
    run_benchmark(
        config_path=args.config,
        mode=args.mode,
        local_jsonl=args.local_jsonl,
        num_samples=args.num_samples,
        pool_size=args.pool_size,
        gen_batch_size=args.gen_batch_size,
        pool_path=args.pool_path,
        eval_only=args.eval_only,
        reverify=args.reverify,
        methods=methods,
        budget_multipliers=budget_multipliers,
        seeds=seeds,
        lambda_mode=args.lambda_mode,
        lambda_value=args.lambda_value,
        output_csv=args.output_csv,
        overwrite_csv=args.overwrite_csv,
        status=args.status,
        device=args.device,
        dtype=args.dtype,
        backbone_revision=args.backbone_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
