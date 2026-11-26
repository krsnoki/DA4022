from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch
import transformers
from torch import Tensor, nn

from Backbones.Generative_Pretrained.llama import LLAMA_REGISTRY, TINY_LLAMA_ID, load_llama
from Dataloaders.calibration.causal_lm import make_calibration_batches
from Dataloaders.causal_lm import TokenBatch
from Dataloaders.synthetic.causal_lm import make_synthetic_batches

from ..gptq import GPTQConfig, PackedWeight, collect_input_grams, gptq_quantize
from .calibration import collect_covariances
from .llama import RotatedLlama, fuse_llama
from .rotations import (
    HADAMARD_44_SHA256,
    HADAMARD_172_SHA256,
    Hadamard,
    RotationDimensions,
    RotationPlan,
    SignedHadamard,
    build_rotations,
)
from .runtime import QuantizedLlama, RuntimeConfig, install_quantized_runtime

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class CoatConfig:
    backbone_id: str = TINY_LLAMA_ID
    device: str = "cpu"
    dtype: torch.dtype | None = None
    model_seed: int = 0
    calibration_dataset: str | None = None
    calibration_seed: int = 0
    calibration_count: int | None = None
    calibration_seq_len: int | None = None
    batch_size: int = 1
    rotation_seed: int = 0
    factors: dict[int, str | Path] = field(default_factory=dict)
    cache_dir: str | Path | None = None
    logit_atol: float | None = None
    logit_rtol: float | None = None
    gptq: GPTQConfig = GPTQConfig()
    runtime: RuntimeConfig = RuntimeConfig()


@dataclass(frozen=True)
class RunReport:
    artifact_path: Path
    provenance: dict
    timings_s: dict[str, float]
    max_abs_logit_error: float
    mean_abs_logit_error: float


@contextmanager
def _timed(timings: dict[str, float], name: str, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings[name] = perf_counter() - start


@torch.inference_mode()
def _probe_logits(model, batch: TokenBatch) -> Tensor:
    end = batch.input_ids.shape[1] - 4
    out = model(input_ids=batch.input_ids[:, :end], attention_mask=batch.attention_mask[:, :end], use_cache=True)
    logits = [out.logits.cpu()]
    for i in range(end, end + 4):
        out = model(input_ids=batch.input_ids[:, i:i + 1], attention_mask=batch.attention_mask[:, :i + 1],
                    past_key_values=out.past_key_values, use_cache=True)
        logits.append(out.logits.cpu())
    return torch.cat(logits, dim=1)


def save_coat(model: QuantizedLlama, path: str | Path, *, provenance: dict, calibration: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save({
            "version": ARTIFACT_VERSION,
            "provenance": provenance,
            "calibration": calibration,
            "state": model.state_dict(),
        }, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_coat(path: str | Path, *, device: str | torch.device = "cpu",
              cache_dir: str | Path | None = None) -> QuantizedLlama:
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if artifact["version"] != ARTIFACT_VERSION:
        raise ValueError(f"unsupported COAT artifact version {artifact['version']}")
    meta, state = artifact["provenance"], artifact["state"]
    backbone_id = meta["backbone"]["id"]
    registered = LLAMA_REGISTRY[backbone_id]
    if (registered.revision != meta["backbone"]["revision"]
            or registered.tokenizer_revision != meta["backbone"]["tokenizer_revision"]):
        raise ValueError("artifact backbone or tokenizer revision differs from the registered revision")

    dtype = getattr(torch, meta["dtype"])
    backbone = load_llama(backbone_id, device=device, dtype=dtype, seed=meta["model_seed"], cache_dir=cache_dir)
    adapter = backbone.adapter
    dims = RotationDimensions.from_config(backbone.model.config)
    head = adapter.lm_head
    if not meta["tied_embeddings"] and head.weight is adapter.embeddings.weight:
        head.weight = nn.Parameter(torch.empty_like(head.weight), requires_grad=head.weight.requires_grad)
        backbone.model.config.tie_word_embeddings = False

    def online(name: str, width: int) -> list[SignedHadamard]:
        factor = state.get(f"rotations.{name}.0.hadamard.factor")
        h = Hadamard(width if factor is None else width // factor.shape[0], factor)
        return [SignedHadamard(h, state[f"rotations.{name}.{i}.signs"]) for i in range(dims.num_layers)]

    plan = RotationPlan(
        state["rotations.r1"], state["rotations.r2"], online("r3", dims.head_dim),
        online("r4", dims.intermediate_size), seed=meta["rotation_seed"],
    ).to(device=device)
    weights = {}
    for i, layer in enumerate(adapter.layers):
        attn = layer.self_attn
        weights[i] = {}
        for name, proj in adapter.projections(i).items():
            owner = "self_attn" if hasattr(attn, name) else "mlp"
            prefix = f"model.model.layers.{i}.{owner}.{name}"
            weights[i][name] = PackedWeight(state[f"{prefix}.codes"], state[f"{prefix}.scale"], proj.in_features)

    backbone.model._coat_fused = True
    model = install_quantized_runtime(RotatedLlama(adapter, plan), weights, RuntimeConfig(**meta["runtime"]))
    model.load_state_dict(state)
    return model


def run_coat(config: CoatConfig, output_dir: str | Path) -> RunReport:
    device = torch.device(config.device)
    dtype = config.dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    tiny = config.backbone_id == TINY_LLAMA_ID
    dataset = config.calibration_dataset or ("synthetic_causal_lm" if tiny else "wikitext2")
    count = config.calibration_count if config.calibration_count is not None else (8 if tiny else 128)
    seq_len = config.calibration_seq_len if config.calibration_seq_len is not None else (32 if tiny else 2048)
    stats_dtype = torch.float64 if device.type == "cpu" else torch.float32
    model_args = dict(device=device, dtype=dtype, seed=config.model_seed, cache_dir=config.cache_dir)
    baseline = load_llama(config.backbone_id, **model_args)
    batches_args = dict(seed=config.calibration_seed, count=count, seq_len=seq_len, batch_size=config.batch_size)
    vocab_size = baseline.model.config.vocab_size
    if dataset == "synthetic_causal_lm":
        batches = make_synthetic_batches(vocab_size=vocab_size, **batches_args)
        manifest = {"dataset_id": dataset, "revision": "v1", "split": "calibration", **batches_args}
    else:
        if baseline.tokenizer is None:
            raise ValueError("text calibration requires a registered tokenizer")
        cal = make_calibration_batches(dataset, baseline.tokenizer, tokenizer_revision=baseline.metadata.tokenizer_revision,
                                       cache_dir=config.cache_dir, **batches_args)
        manifest = asdict(cal.manifest)
        batches = tuple(cal)

    timings: dict[str, float] = {}
    with _timed(timings, "covariance", device):
        cov = collect_covariances(baseline.adapter, batches, stats_dtype=stats_dtype)
    dims = RotationDimensions.from_config(baseline.model.config)
    with _timed(timings, "rotations", device):
        plan = build_rotations(cov, dims, seed=config.rotation_seed, factors=config.factors)
    del cov
    probe = make_synthetic_batches(split="evaluation", seed=config.calibration_seed, count=2,
                                   seq_len=36, batch_size=2, vocab_size=vocab_size)[0].to(device)
    expected = _probe_logits(baseline.model, probe)
    backbone_meta = asdict(baseline.metadata)
    del baseline

    target = load_llama(config.backbone_id, **model_args)
    with _timed(timings, "fusion", device):
        rotated = fuse_llama(target.adapter, plan)
    actual = _probe_logits(rotated, probe)
    tol = 1e-4 if dtype in (torch.float32, torch.float64) else 5e-3
    atol = config.logit_atol if config.logit_atol is not None else tol
    rtol = config.logit_rtol if config.logit_rtol is not None else tol
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg="COAT logit equivalence failed before GPTQ")
    error = (actual.float() - expected.float()).abs()
    max_error, mean_error = error.max().item(), error.mean().item()

    weights = {}
    with _timed(timings, "gptq", device):
        for i, grams in collect_input_grams(rotated, batches):
            weights[i] = {}
            for name, proj in rotated.adapter.projections(i).items():
                packed = gptq_quantize(proj.weight, grams[name], config.gptq)
                weights[i][name] = PackedWeight(packed.codes.cpu(), packed.scale.cpu(), packed.in_features)
            del grams
    model = install_quantized_runtime(rotated, weights, config.runtime)

    factor = plan.r4[0].hadamard.factor
    factor_order = factor.shape[0] if factor is not None else None
    provenance = {
        "backbone": backbone_meta,
        "model_seed": config.model_seed,
        "rotation_seed": config.rotation_seed,
        "hadamard_44_sha256": HADAMARD_44_SHA256 if factor_order == 44 else None,
        "hadamard_172_sha256": HADAMARD_172_SHA256 if factor_order == 172 else None,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "stats_dtype": str(stats_dtype).removeprefix("torch."),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "tied_embeddings": target.adapter.embeddings.weight is target.adapter.lm_head.weight,
        "gptq": {**asdict(config.gptq), "range": [-7, 7], "group_size": -1,
                 "act_order": False, "clipping_search": False, "activation_quantization_during_collection": False},
        "runtime": asdict(config.runtime),
        "calibration": manifest,
        "calibration_batch_size": config.batch_size,
        "equivalence": {"atol": atol, "rtol": rtol, "max_abs_error": max_error, "mean_abs_error": mean_error},
        "timings_s": timings,
    }
    calibration = {"input_ids": torch.cat([batch.input_ids for batch in batches]),
                   "attention_mask": torch.cat([batch.attention_mask for batch in batches])}
    path = Path(output_dir) / "coat-artifact"
    save_coat(model, path, provenance=provenance, calibration=calibration)
    return RunReport(path, provenance, timings, max_error, mean_error)
