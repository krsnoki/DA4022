"""
RULER benchmark for KV-cache eviction presses (KVgrad paper, AdaptFM / ICML 2026 workshop).

Runs several kvpress presses on the *same* fixed, per-task-stratified RULER sample and records,
for every (press, compression ratio) configuration:

  * quality:  RULER string-match per task and macro-averaged over tasks
  * latency:  per-example wall time, split into "prefill + compression" and "decode"
  * memory:   peak GPU memory allocated during each example, plus the KV-cache size that the
              kept entries would occupy (kvpress masks evicted entries instead of freeing them,
              so measured peak memory does NOT shrink with the compression ratio)
  * metadata: software / hardware versions, model revision, sample ids, git commit

Outputs (under --output_dir):

  env.json                                 software / hardware metadata of the session
  samples_<N>pertask_seed<S>.csv           the exact rows used (re-used by every later run)
  <press>_cr<ratio>/predictions.csv        one row per example, with timings and memory
  <press>_cr<ratio>/metrics.json           per-task scores + latency / memory summary
  summary.csv                              one row per configuration, schema of
                                           Performance_Metrics/README_metrics.md

A configuration whose metrics.json already exists is skipped, and a configuration that was
interrupted resumes from its partial predictions.csv, so a disconnected Colab session can simply
re-run the same command.

Example (Colab, T4):
    python run_ruler.py --config configs/ruler4k_qwen3_0.6b.yaml \
        --output_dir /content/drive/MyDrive/kvgrad_results/phase2
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ATTENTION_DIR = SCRIPT_DIR.parents[1]  # makes Attention/methods importable

DEFAULTS: dict[str, Any] = {
    "model": "Qwen/Qwen3-0.6B",
    "model_revision": None,  # pin a Hugging Face commit; None = latest
    "dtype": "float32",  # T4 has no bf16; fp32 avoids fp16 overflow
    "attn_implementation": "sdpa",  # KVzip/KVgrad refuse "eager"; FlashAttention-2 does not run on T4
    "device": "cuda:0",
    "dataset_id": "simonjegou/ruler",
    "data_dir": "4096",
    "split": "test",
    "data_file": None,  # optional local .parquet/.jsonl/.csv instead of the Hugging Face dataset
    "per_task": 20,
    "seed": 42,
    "presses": ["no_press", "kvzip", "kvzip_plus", "kvgrad"],
    "compression_ratios": [0.5, 0.75, 0.875],
    "warmup": 1,
    "max_new_tokens": None,  # None = use each task's default from the dataset
    "save_every": 10,
    "share_scores": True,  # KVgrad variants: replay each context once and reuse the scores of all variants
    "output_dir": "results",
    "status": "partial",
    "notes": "",
}

SUMMARY_COLUMNS = [
    "experiment_id",
    "paper_id",
    "method",
    "commit",
    "backbone_id",
    "backbone_revision",
    "dataset_id",
    "split",
    "seed",
    "device",
    "framework_version",
    "dtype",
    "batch_size",
    "input_length",
    "output_length",
    "quality_metric",
    "quality_value",
    "latency_median_ms",
    "latency_p95_ms",
    "throughput",
    "peak_memory_mb",
    "energy_j",
    "status",
    "notes",
]


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML file with any of the settings below")
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_revision", type=str)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--attn_implementation", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--per_task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--presses", type=str, nargs="+")
    parser.add_argument("--compression_ratios", type=float, nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--notes", type=str)
    args = parser.parse_args(argv)

    config = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            file_config = yaml.safe_load(f) or {}
        unknown = set(file_config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown keys in {args.config}: {sorted(unknown)}")
        config.update(file_config)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    for press_name in config["presses"]:
        if press_name not in PRESS_BUILDERS:
            raise ValueError(f"Unknown press '{press_name}'. Choose from {sorted(PRESS_BUILDERS)}")
    for ratio in config["compression_ratios"]:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"compression ratios must be in (0, 1), got {ratio}")
    return config


# --------------------------------------------------------------------------------------------
# Presses
# --------------------------------------------------------------------------------------------
def _kvzip(ratio: float):
    from kvpress import KVzipPress

    return KVzipPress(compression_ratio=ratio)


def _kvzip_plus(ratio: float):
    from kvpress import KVzipPress

    return KVzipPress(compression_ratio=ratio, kvzip_plus_normalization=True)


def _kvgrad(ratio: float):
    from kvpress import KVgradPress

    return KVgradPress(compression_ratio=ratio)


# Scores of our KVgrad variants, shared by every configuration of one run (see KVgradVariantsPress).
SCORE_CACHE: dict = {}


def _kvgrad_variant(variant: str):
    def build(ratio: float, share_scores: bool = True):
        if str(ATTENTION_DIR) not in sys.path:
            sys.path.insert(0, str(ATTENTION_DIR))
        from methods.context_compression.kvgrad import KVgradVariantsPress

        cache = SCORE_CACHE if share_scores else None
        return KVgradVariantsPress(compression_ratio=ratio, variant=variant, score_cache=cache)

    return build


# Baselines: same definitions as PRESS_REGISTRY in kvpress/evaluation/evaluate_registry.py (v0.5.5).
# kvgrad_*: our implementation of the paper's score variants (Attention/methods/context_compression/kvgrad).
PRESS_BUILDERS = {
    "no_press": None,
    "kvzip": _kvzip,
    "kvzip_plus": _kvzip_plus,
    "kvgrad": _kvgrad,
    "kvgrad_bound_max": _kvgrad_variant("bound_max"),  # Eq. (8), KVgrad
    "kvgrad_bound_sum": _kvgrad_variant("bound_sum"),  # Eq. (7) summed over probes (w/o PSP)
    "kvgrad_taylor_max": _kvgrad_variant("taylor_max"),  # signed Taylor terms, max over probes (w/o MAS)
    "kvgrad_taylor_sum": _kvgrad_variant("taylor_sum"),  # Eq. (6), first-order Taylor (w/o both)
}


def build_press(name: str, ratio: float, share_scores: bool = True):
    builder = PRESS_BUILDERS[name]
    if builder is None:
        return None
    if name.startswith("kvgrad_"):
        return builder(ratio, share_scores)
    return builder(ratio)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load_ruler(config: dict[str, Any]) -> pd.DataFrame:
    if config["data_file"]:
        path = Path(config["data_file"])
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in {".jsonl", ".json"}:
            df = pd.read_json(path, lines=path.suffix == ".jsonl")
        else:
            df = pd.read_csv(path)
    else:
        from datasets import load_dataset

        df = load_dataset(config["dataset_id"], data_dir=str(config["data_dir"]), split=config["split"]).to_pandas()
    required = {"context", "question", "answer_prefix", "answer", "task", "max_new_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns {sorted(missing)}")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


def stratified_sample(df: pd.DataFrame, per_task: int, seed: int, sample_file: Path) -> pd.DataFrame:
    """Same rows for every press and every session: sample once, then always re-use the saved ids."""
    if sample_file.exists():
        ids = pd.read_csv(sample_file)["row_id"].tolist()
        print(f"Re-using {len(ids)} sampled rows from {sample_file}")
    else:
        parts = [
            group.sample(n=min(per_task, len(group)), random_state=seed) for _, group in df.groupby("task", sort=True)
        ]
        sampled = pd.concat(parts)
        sampled[["row_id", "task"]].to_csv(sample_file, index=False)
        ids = sampled["row_id"].tolist()
        print(f"Sampled {len(ids)} rows ({per_task} per task, seed {seed}) -> {sample_file}")
    return df.set_index("row_id").loc[ids].reset_index()


# --------------------------------------------------------------------------------------------
# Scoring: identical to kvpress/evaluation/benchmarks/ruler/calculate_metrics.py (v0.5.5, Apache-2.0)
# --------------------------------------------------------------------------------------------
_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def _clean(prediction: str) -> str:
    return _CONTROL_CHARS.sub("", str(prediction).strip()).strip()


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref) for p, ref in zip(preds, refs))
    return round(score / len(preds) * 100, 2)


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref) for p, ref in zip(preds, refs))
    return round(score / len(preds) * 100, 2)


def ruler_scores(df: pd.DataFrame) -> dict[str, float]:
    scores = {}
    for task, group in df.groupby("task", sort=True):
        preds = [_clean(p) for p in group["predicted_answer"]]
        refs = [list(a) for a in group["answer"]]
        metric = string_match_part if task.split("_")[0] == "qa" else string_match_all
        scores[task] = metric(preds, refs)
    return scores


# --------------------------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------------------------
def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class DecodeTimer:
    """Wraps pipeline.generate_answer so the decode phase is timed separately from prefill + compression."""

    def __init__(self, pipe, device: str):
        self.seconds = 0.0
        self._device = device
        self._original = pipe.generate_answer
        pipe.generate_answer = self._timed

    def _timed(self, *args, **kwargs):
        _sync(self._device)
        start = time.perf_counter()
        result = self._original(*args, **kwargs)
        _sync(self._device)
        self.seconds += time.perf_counter() - start
        return result


def run_example(pipe, timer: DecodeTimer, row: pd.Series, press, config: dict[str, Any]) -> dict[str, Any]:
    device = config["device"]
    max_new_tokens = int(config["max_new_tokens"] or row["max_new_tokens"])
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    timer.seconds = 0.0
    _sync(device)
    start = time.perf_counter()
    output = pipe(
        row["context"],
        question=row["question"],
        answer_prefix=row["answer_prefix"],
        press=press,
        max_new_tokens=max_new_tokens,
    )
    _sync(device)
    total = time.perf_counter() - start
    answer = output["answer"]
    return {
        "predicted_answer": answer,
        "total_ms": total * 1000,
        "prefill_compress_ms": (total - timer.seconds) * 1000,
        "decode_ms": timer.seconds * 1000,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else np.nan,
        "output_tokens": len(pipe.tokenizer(answer, add_special_tokens=False).input_ids),
        # True when a KVgrad variant reused cached scores (no replay), so this latency excludes scoring
        "scoring_cached": getattr(press, "last_scoring_cached", None),
    }


# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------
def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=SCRIPT_DIR, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=SCRIPT_DIR, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment(config: dict[str, Any]) -> dict[str, Any]:
    device = config["device"]
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _pkg_version("transformers"),
        "kvpress": _pkg_version("kvpress"),
        "datasets": _pkg_version("datasets"),
        "device": gpu,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if gpu != "cpu" else None,
        "git_commit": git_commit(),
    }


# --------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------
def run_dir_name(press_name: str, ratio: float) -> str:
    return f"{press_name}_cr{ratio:.3f}"


def kv_bytes_per_token(model_config, dtype: str) -> int:
    """Bytes of keys + values stored per context token, summed over all layers."""
    head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // model_config.num_attention_heads
    element_size = torch.tensor([], dtype=getattr(torch, dtype)).element_size()
    return 2 * model_config.num_hidden_layers * model_config.num_key_value_heads * head_dim * element_size


def summarise(predictions: pd.DataFrame, ratio: float, bytes_per_token: int) -> dict[str, Any]:
    scores = ruler_scores(predictions)
    total = predictions["total_ms"]
    peak = predictions["peak_memory_mb"].dropna()
    return {
        "compression_ratio": ratio,
        "n_examples": len(predictions),
        "n_tasks": len(scores),
        "string_match_per_task": scores,
        "string_match_macro": round(float(np.mean(list(scores.values()))), 2),
        "latency_median_ms": round(float(total.median()), 1),
        "latency_p95_ms": round(float(total.quantile(0.95)), 1),
        "prefill_compress_median_ms": round(float(predictions["prefill_compress_ms"].median()), 1),
        "decode_median_ms": round(float(predictions["decode_ms"].median()), 1),
        "peak_memory_mb_max": round(float(peak.max()), 0) if len(peak) else None,
        "peak_memory_mb_median": round(float(peak.median()), 0) if len(peak) else None,
        "input_tokens_median": int(predictions["input_tokens"].median()),
        "kv_cache_mb_full_median": round(float(predictions["input_tokens"].median()) * bytes_per_token / 2**20, 1),
        "kv_cache_mb_kept_median": round(
            float(predictions["input_tokens"].median()) * (1 - ratio) * bytes_per_token / 2**20, 1
        ),
        "output_tokens_median": int(predictions["output_tokens"].median()),
        "examples_per_s": round(float(len(predictions) / (total.sum() / 1000)), 4),
        "scoring_cached_fraction": round(float(_cached_fraction(predictions)), 3),
    }


def _cached_fraction(predictions: pd.DataFrame) -> float:
    if "scoring_cached" not in predictions:
        return 0.0
    return float(predictions["scoring_cached"].map(lambda v: str(v).lower() == "true").mean())


def run_configuration(
    pipe, sample: pd.DataFrame, press_name: str, ratio: float, config: dict, out_dir: Path, bytes_per_token: int
) -> None:
    run_dir = out_dir / run_dir_name(press_name, ratio)
    metrics_file = run_dir / "metrics.json"
    predictions_file = run_dir / "predictions.csv"
    if metrics_file.exists():
        print(f"[skip] {run_dir.name}: metrics.json already exists")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    done = pd.read_csv(predictions_file) if predictions_file.exists() else pd.DataFrame()
    done_ids = set(done["row_id"]) if len(done) else set()
    todo = sample[~sample["row_id"].isin(done_ids)]
    print(f"[run ] {run_dir.name}: {len(done_ids)} done, {len(todo)} to go")

    press = build_press(press_name, ratio, config["share_scores"])
    timer = DecodeTimer(pipe, config["device"])
    warmup_press = build_press(press_name, ratio, share_scores=False)  # must not fill the score cache
    for _ in range(config["warmup"] if len(todo) else 0):  # untimed warm-up, result discarded
        run_example(pipe, timer, sample.iloc[0], warmup_press, config)

    records = done.to_dict("records") if len(done) else []
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        result = run_example(pipe, timer, row, press, config)
        records.append({"row_id": row["row_id"], "task": row["task"], "input_tokens": row["input_tokens"], **result})
        if i % config["save_every"] == 0 or i == len(todo):
            pd.DataFrame(records).to_csv(predictions_file, index=False)
            print(f"       {len(records)}/{len(sample)}  last: {result['total_ms'] / 1000:.2f} s")

    predictions = pd.DataFrame(records).merge(sample[["row_id", "answer"]], on="row_id")
    metrics = summarise(predictions, ratio, bytes_per_token)
    metrics.update({"press": press_name, "config": {k: config[k] for k in sorted(config)}})
    metrics_file.write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"[done] {run_dir.name}: macro string-match {metrics['string_match_macro']:.1f}, "
        f"median {metrics['latency_median_ms'] / 1000:.2f} s, peak {metrics['peak_memory_mb_max']} MB"
    )


def write_summary(out_dir: Path, config: dict[str, Any], env: dict[str, Any], model_revision: str) -> Path:
    rows = []
    for metrics_file in sorted(out_dir.glob("*_cr*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        cfg = m["config"]
        rows.append(
            {
                "experiment_id": f"kvgrad-ruler{cfg['data_dir']}-{m['press']}-cr{m['compression_ratio']:.3f}",
                "paper_id": "kvgrad-icml2026-adaptfm",
                "method": m["press"],
                "commit": env["git_commit"],
                "backbone_id": cfg["model"],
                "backbone_revision": model_revision,
                "dataset_id": cfg["data_file"] or f"{cfg['dataset_id']}/{cfg['data_dir']}",
                "split": f"{cfg['split']} ({cfg['per_task']}/task, {m['n_tasks']} tasks)",
                "seed": cfg["seed"],
                "device": env["device"],
                "framework_version": f"torch {env['torch']}; transformers {env['transformers']}; kvpress {env['kvpress']}",
                "dtype": cfg["dtype"],
                "batch_size": 1,
                "input_length": m["input_tokens_median"],
                "output_length": m["output_tokens_median"],
                "quality_metric": "ruler_string_match_macro",
                "quality_value": m["string_match_macro"],
                "latency_median_ms": m["latency_median_ms"],
                "latency_p95_ms": m["latency_p95_ms"],
                "throughput": m["examples_per_s"],
                "peak_memory_mb": m["peak_memory_mb_max"],
                "energy_j": None,
                "status": cfg["status"],
                "notes": (
                    f"compression_ratio={m['compression_ratio']}; n={m['n_examples']}; "
                    f"KV cache kept {m['kv_cache_mb_kept_median']} of {m['kv_cache_mb_full_median']} MB (median context, "
                    f"theoretical: kvpress masks evicted entries, so peak_memory_mb does not shrink); "
                    f"prefill+compress median {m['prefill_compress_median_ms']} ms; decode median {m['decode_median_ms']} ms; "
                    f"throughput in examples/s; latency = full pipeline call incl. tokenisation; "
                    f"attn={cfg['attn_implementation']}; energy not measured"
                    + (
                        f"; scores reused without replay for {m['scoring_cached_fraction']:.0%} of examples, "
                        "so latency excludes scoring for those (quality unaffected)"
                        if m.get("scoring_cached_fraction", 0) > 0
                        else ""
                    )
                ),
            }
        )
    summary_file = out_dir / "summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(summary_file, index=False)
    return summary_file


def main(argv: list[str] | None = None) -> Path:
    config = load_config(argv)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    import kvpress  # noqa: F401  (registers the "kv-press-text-generation" pipeline)
    from transformers import pipeline

    env = environment(config)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    pipeline_kwargs: dict[str, Any] = {
        "model": config["model"],
        "device": config["device"],
        "model_kwargs": {
            "attn_implementation": config["attn_implementation"],
            "dtype": getattr(torch, config["dtype"]),
        },
    }
    if config["model_revision"]:
        pipeline_kwargs["revision"] = config["model_revision"]
    pipe = pipeline("kv-press-text-generation", **pipeline_kwargs)
    pipe.model.eval()
    model_revision = config["model_revision"] or getattr(pipe.model.config, "_commit_hash", None) or "unknown"
    bytes_per_token = kv_bytes_per_token(pipe.model.config, config["dtype"])
    print(f"KV cache: {bytes_per_token / 2**10:.1f} KiB per context token ({config['dtype']})")

    df = load_ruler(config)
    sample_file = out_dir / f"samples_{config['per_task']}pertask_seed{config['seed']}.csv"
    sample = stratified_sample(df, config["per_task"], config["seed"], sample_file)
    sample["input_tokens"] = [len(pipe.tokenizer(c, add_special_tokens=False).input_ids) for c in sample["context"]]
    print(sample.groupby("task").size().to_string())

    for press_name in config["presses"]:
        ratios = [0.0] if press_name == "no_press" else config["compression_ratios"]
        for ratio in ratios:
            run_configuration(pipe, sample, press_name, ratio, config, out_dir, bytes_per_token)

    summary_file = write_summary(out_dir, config, env, model_revision)
    print(f"\nSummary written to {summary_file}")
    table = pd.read_csv(summary_file)
    table["ratio"] = table["experiment_id"].str.extract(r"cr([0-9.]+)$", expand=False)
    print(table[["method", "ratio", "quality_value", "latency_median_ms", "peak_memory_mb"]].to_string(index=False))
    return summary_file


if __name__ == "__main__":
    main()
