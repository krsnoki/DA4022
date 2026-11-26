import argparse
import platform
import shutil
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
import transformers
import yaml

from Backbones.Generative_Pretrained.llama import LLAMA_REGISTRY, TINY_LLAMA_ID, TINYLLAMA_1B_ID, load_llama
from Dataloaders.causal_lm import corpus_spec
from Dataloaders.evaluation.causal_lm import ZERO_SHOT_TASKS, make_evaluation_batches
from Dataloaders.synthetic.causal_lm import make_synthetic_batches
from Performance_Metrics.causal_lm import (
    REPORT_FIELDS,
    EvalSets,
    GenerationSettings,
    ZeroShotTask,
    benchmark_generation,
    evaluate_quality,
    write_json_report,
)
from Quantization_and_Pruning.quantization.ptq.coat.llama import fuse_llama
from Quantization_and_Pruning.quantization.ptq.coat.pipeline import CoatConfig, load_coat, run_coat
from Quantization_and_Pruning.quantization.ptq.coat.rotations import (
    HADAMARD_44_SHA256,
    HADAMARD_172_SHA256,
    load_hadamard44,
    load_hadamard172,
)
from Quantization_and_Pruning.quantization.ptq.coat.runtime import PackedLinear, RuntimeConfig
from Quantization_and_Pruning.quantization.ptq.gptq import GPTQConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "coat_llama2_7b.yaml"
PAPER_ID = "openreview:6JGbm48VoG"


def _profile(args, cache_dir):
    if args.dry_run:
        torch.set_num_threads(1)
        dataset = "synthetic_causal_lm"
        return (CoatConfig(backbone_id=TINY_LLAMA_ID, dtype=torch.float32),
                {"datasets": [dataset], "seq_len": 32, "batch_size": 1, "seed": 0},
                GenerationSettings(prompt_length=16, decode_tokens=4, warmups=1, iterations=3), dataset, ())

    spec = yaml.safe_load((args.config or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    coat, revisions, evaluation = spec["coat"], spec["revisions"], spec["evaluation"]
    backbone = LLAMA_REGISTRY[coat["backbone_id"]]
    if (backbone.revision, backbone.tokenizer_revision) != (revisions["backbone"], revisions["tokenizer"]):
        raise ValueError("config and registry backbone/tokenizer revisions differ")
    generation = dict(spec["generation"])
    generation_dataset = generation.pop("dataset")
    sources = [(name, "evaluation") for name in {*evaluation["datasets"], generation_dataset}]
    sources.append((coat["calibration_dataset"], "calibration"))
    for name, role in sources:
        if corpus_spec(name, role)["revision"] != revisions["datasets"][name]:
            raise ValueError(f"config and registry revisions differ for {name}")
    tinyllama = backbone.id == TINYLLAMA_1B_ID
    task_specs = evaluation.get("tasks", {})
    if (not tinyllama or task_specs) and set(task_specs) != set(ZERO_SHOT_TASKS):
        raise ValueError("expected the nine COAT zero-shot tasks")
    if tinyllama:
        order, checksum, factor, load_factor = 44, HADAMARD_44_SHA256, args.hadamard_44, load_hadamard44
    else:
        order, checksum, factor, load_factor = 172, HADAMARD_172_SHA256, args.hadamard_172, load_hadamard172
    if revisions[f"hadamard_{order}_sha256"] != checksum:
        raise ValueError("config and implementation Hadamard checksums differ")
    device = torch.device(coat["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the pretrained profile requires CUDA. Use --dry-run for the CPU fixture.")
    if factor is None:
        raise ValueError(f"{backbone.id} requires --hadamard-{order} PATH")
    load_factor(factor)
    if task_specs and version("lm_eval") != evaluation["harness_version"]:
        raise ValueError(f"install lm_eval=={evaluation['harness_version']} for zero-shot evaluation")
    coat = {**coat, "dtype": getattr(torch, coat["dtype"]), "gptq": GPTQConfig(**coat["gptq"]),
            "runtime": RuntimeConfig(**coat["runtime"]), "cache_dir": cache_dir, "factors": {order: factor}}
    tasks = tuple(ZeroShotTask(name, **task_specs[name]) for name in ZERO_SHOT_TASKS if name in task_specs)
    return CoatConfig(**coat), evaluation, GenerationSettings(**generation), generation_dataset, tasks


def _command(*args):
    if not shutil.which(args[0]):
        return None
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _hardware(device):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "cpu_threads": torch.get_num_threads(), "cuda_runtime": torch.version.cuda,
            "accelerator_memory_bytes": None, "driver": None}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        drivers = _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        info.update(name=props.name, accelerator_memory_bytes=props.total_memory,
                    driver=drivers.splitlines()[0] if drivers else None)
    else:
        info["name"] = platform.processor() or platform.machine()
    return info


def _model_size(model):
    stores = {}
    for tensor in model.state_dict().values():
        storage = tensor.untyped_storage()
        stores[storage.data_ptr()] = storage.nbytes()
    parameters = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, PackedLinear):
            parameters += module.in_features * module.out_features
            if module.bias is not None:
                parameters += module.bias.numel()
    return parameters, sum(stores.values())


def _checkpoint_bytes(meta, cache_dir):
    if meta.tokenizer_revision is None:
        return None
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(meta.id, revision=meta.revision, local_files_only=True, cache_dir=cache_dir))
    sizes = [p.stat().st_size for p in path.glob("*.safetensors")]
    return sum(sizes) if sizes else None


def _run(args):
    output = args.output_dir
    cache_dir = args.cache_dir or output / "hub"
    cfg, ev, gen, generation_dataset, tasks = _profile(args, cache_dir)
    profile = "tinyllama_1b_cuda" if cfg.backbone_id == TINYLLAMA_1B_ID else "llama2_7b_cuda"
    torch.set_float32_matmul_precision("highest")
    device = torch.device(cfg.device)
    meta = LLAMA_REGISTRY[cfg.backbone_id]
    hw = _hardware(device)
    common = dict.fromkeys(REPORT_FIELDS)
    common.update(
        paper_id=PAPER_ID, commit=_command("git", "rev-parse", "HEAD"),
        source_dirty=bool(_command("git", "status", "--porcelain")), backbone_id=meta.id,
        backbone_revision=meta.revision, tokenizer_revision=meta.tokenizer_revision,
        seed=ev["seed"], device=str(device), hardware=hw, framework_version=hw["torch"],
        dtype=str(cfg.dtype).removeprefix("torch."), backend="pytorch_eager", status="partial",
        energy_status="not_measured", cost=None, cost_status="not_measured",
        profile="synthetic-smoke" if args.dry_run else profile,
    )
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    run_id = uuid4().hex
    rows = []
    baseline_scores = {}

    def row(variant, workload, **values):
        return {**common, "experiment_id": f"{run_id}/{variant}/{workload}", "method": variant,
                "workload": workload, **values}

    try:
        report = run_coat(cfg, output)
    except Exception as exc:
        write_json_report(report_path, [row("coat_w4a4kv4", "preparation", status="failed",
                                            notes=f"{type(exc).__name__}: {exc}")])
        raise

    load_args = dict(device=device, dtype=cfg.dtype, seed=cfg.model_seed, cache_dir=cfg.cache_dir)
    backbone = load_llama(cfg.backbone_id, **load_args)
    tokenizer = backbone.tokenizer
    gen_length = gen.prompt_length + gen.decode_tokens
    if args.dry_run:
        batches = make_synthetic_batches(split="evaluation", count=4, seq_len=ev["seq_len"], seed=ev["seed"])
        ppl = {generation_dataset: batches}
        generation_batches = make_synthetic_batches(split="evaluation", count=1, seq_len=gen_length, seed=ev["seed"] + 1)
    else:
        ppl = {name: tuple(make_evaluation_batches(name, tokenizer, seq_len=ev["seq_len"],
                                                  batch_size=ev["batch_size"], cache_dir=cache_dir))
               for name in ev["datasets"]}
        generation_batches = (next(make_evaluation_batches(generation_dataset, tokenizer, seq_len=gen_length,
                                                           cache_dir=cache_dir)),)
    torch.save([{"input_ids": b.input_ids, "attention_mask": b.attention_mask} for b in generation_batches],
               output / "generation-inputs")
    eval_sets = EvalSets(ppl, tasks, ev.get("harness_version", "0.4.8"), ev["seq_len"], ev["batch_size"], ev["seed"])
    specs = {name: corpus_spec(name, "evaluation") for name in ev["datasets"]}
    timing_source = corpus_spec(generation_dataset, "evaluation")

    def record(variant, model, checkpoint_bytes, storage):
        packed = variant == "coat_w4a4kv4"
        original = variant == "floating_point"
        stages = {} if original else {k: v for k, v in report.timings_s.items() if packed or k != "gptq"}
        parameter_count, tensor_bytes = _model_size(model)
        extra = {"parameter_count": parameter_count, "tensor_storage_bytes": tensor_bytes,
                 "checkpoint_bytes": checkpoint_bytes,
                 "storage_format": storage, "preparation_wall_s": stages,
                 "method_settings": {} if original else report.provenance}
        extra["notes"] = {
            "floating_point": "Untouched floating-point backbone.",
            "coat_rotation_only": "COAT rotations with floating-point inference.",
            "coat_w4a4kv4": "Packed INT4 storage with floating-point compute.",
        }[variant]
        try:
            quality = evaluate_quality(model, tokenizer, eval_sets)
            runtime = benchmark_generation(model, generation_batches, gen)
        except Exception as exc:
            rows.append(row(variant, "evaluation", **{**extra, "status": "failed", "notes": f"{type(exc).__name__}: {exc}"}))
            write_json_report(report_path, rows)
            raise

        variant_rows = []
        for name, score in quality.perplexity.items():
            spec = specs[name]
            variant_rows.append(row(variant, f"perplexity/{name}", **extra, dataset_id=name, split=spec["split"],
                                    dataset_revision=spec["revision"], batch_size=ev["batch_size"], input_length=ev["seq_len"],
                                    quality_metric="perplexity", quality_value=score.value, examples=score.examples,
                                    scored_tokens=score.scored_tokens, preprocessing="nonoverlapping_complete_windows_v1"))
        variant_rows.append(row(variant, "perplexity/mean", **extra, dataset_id="perplexity_suite", split="aggregate",
                                dataset_revision={n: s["revision"] for n, s in specs.items()}, batch_size=ev["batch_size"],
                                input_length=ev["seq_len"], quality_metric="mean_perplexity", quality_value=quality.mean_perplexity,
                                examples=sum(s.examples for s in quality.perplexity.values())))
        for task in tasks:
            score = quality.zero_shot[task.task_id]
            variant_rows.append(row(variant, f"zero_shot/{task.task_id}", **extra, dataset_id=task.dataset_id,
                                    split=task.split, dataset_revision=task.revision, task_id=task.task_id,
                                    metric_key=task.metric, batch_size=ev["batch_size"], input_length=ev["seq_len"],
                                    input_length_kind="maximum", quality_metric="accuracy_percent", quality_value=score.value,
                                    examples=score.examples, evaluator=f"lm_eval=={eval_sets.harness_version}", num_fewshot=0))
        if tasks:
            variant_rows.append(row(variant, "zero_shot/mean", **extra, dataset_id="zero_shot_suite", split="aggregate",
                                    dataset_revision={t.task_id: t.revision for t in tasks}, batch_size=ev["batch_size"],
                                    input_length=ev["seq_len"], input_length_kind="maximum", quality_metric="mean_accuracy_percent",
                                    quality_value=quality.mean_zero_shot, evaluator=f"lm_eval=={eval_sets.harness_version}", num_fewshot=0))

        for phase, timing in (("prefill", runtime.prefill), ("decode", runtime.decode)):
            variant_rows.append(row(
                variant, f"generation/{phase}", **extra, dataset_id=generation_dataset, split=timing_source["split"],
                dataset_revision=timing_source["revision"], batch_size=generation_batches[0].input_ids.shape[0],
                input_length=gen.prompt_length, output_length=0 if phase == "prefill" else gen.decode_tokens,
                latency_median_ms=timing.median_ms, latency_p95_ms=timing.p95_ms, throughput=timing.throughput,
                peak_memory_mb=timing.peak_memory_mb, latency_samples_ms=timing.samples_ms, throughput_unit="tokens/s",
                memory_measurement="torch.cuda.max_memory_allocated" if device.type == "cuda" else "not_measured",
                generation=asdict(gen), timing_scope="cached forward, synchronized on CUDA, inputs already on device",
            ))
        for item in variant_rows:
            value = item["quality_value"]
            if value is not None:
                if original:
                    baseline_scores[item["workload"]] = value
                item["quality_delta"] = value - baseline_scores[item["workload"]]
        rows.extend(variant_rows)
        write_json_report(report_path, rows)
        print(f"{variant}: {len(variant_rows)} rows", flush=True)

    bits = 16 if cfg.dtype == torch.float16 else 32
    floating_storage = f"W{bits}A{bits}KV{bits}"
    record("floating_point", backbone.model, _checkpoint_bytes(meta, cfg.cache_dir), floating_storage)
    del backbone
    quantized = load_coat(report.artifact_path, device=device, cache_dir=cfg.cache_dir)
    record("coat_w4a4kv4", quantized, report.artifact_path.stat().st_size, "W4A4KV4")
    rotations = quantized.rotations
    del quantized
    target = load_llama(cfg.backbone_id, **load_args)
    record("coat_rotation_only", fuse_llama(target.adapter, rotations), None, floating_storage)
    print(f"COAT report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the COAT benchmark or its offline CPU fixture.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="run the synthetic CPU fixture")
    mode.add_argument("--config", type=Path, help=f"GPU benchmark settings (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for generated artifacts and the report")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hadamard-172", type=Path, help="local Sloane factor for the 7B profile")
    parser.add_argument("--hadamard-44", type=Path, help="local Sloane factor for the TinyLlama profile")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
