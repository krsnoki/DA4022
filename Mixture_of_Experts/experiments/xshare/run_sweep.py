"""Run the XShare GSM8K sweep described by a YAML config.

    python -m Mixture_of_Experts.experiments.xshare.run_sweep \
        --config Mixture_of_Experts/configs/xshare/gsm8k_sweep.yaml \
        --out-dir Mixture_of_Experts/experiments/xshare/results/gsm8k

One *cell* is one (batch size, question subset, method). For every cell the
script writes ``xshare-gsm8k-{method}-c{batch}-s{subset}.json`` with every
generation, all timings and all metadata. At the end it rewrites
``results.csv`` with one row per cell in the schema of
``Performance_Metrics/README_metrics.md``. Cells whose JSON already exists are
skipped, so the same command resumes an interrupted sweep. ``--methods``,
``--batch-sizes`` and ``--subsets`` run part of the sweep.

The model is loaded once per batch size and the routing configuration is
switched between cells, so every cell runs on the same loaded weights.

What is measured, and where the clock starts and stops
------------------------------------------------------
* Each batch is one prefill pass over the prompts, then greedy decode steps
  with a KV cache. The GPU is synchronised at every boundary.
* ``decode_tokens_per_s``: tokens produced by decode steps for sequences that
  were still generating, divided by total decode time. Prefill, tokenisation
  and answer scoring are excluded.
* ``decode_step_ms``: wall time of every decode step (the whole batch), kept as
  raw samples; the JSON also has the median and p95.
* Request latency: for each question, the time from the start of its batch's
  prefill to the moment its final token was produced. With static batching all
  questions of a batch start together, so this includes no queueing.
* ``load_time_s``: reading the cached weights and moving them to the GPU.
* Peak memory: ``torch.cuda.max_memory_allocated`` over the cell.
* Energy: GPU board power sampled with ``nvidia-smi`` every ``interval_ms``,
  integrated over every batch (prefill + decode). Idle power is measured once
  per batch size with the model loaded and nothing running, and is reported
  so energy above idle can be computed. This is whole-board power: other
  processes on the same GPU would be included, which is why every cell first
  checks that no other process is using the GPU.

Batching is static: a finished sequence stays in the batch as padding and
does not take part in the expert selection. A serving engine with continuous
batching (as in the paper) would refill its slot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from Backbones.Generative_Pretrained.hf_causal_lm import HFCausalLMBackbone
from Dataloaders.evaluation import gsm8k
from Mixture_of_Experts.execution_layouts.xshare_hf import XShareController, install_xshare
from Mixture_of_Experts.routing.xshare import XShareConfig

CSV_FIELDS = [
    "experiment_id", "paper_id", "method", "commit", "backbone_id", "backbone_revision",
    "dataset_id", "split", "seed", "device", "framework_version", "dtype", "batch_size",
    "input_length", "output_length", "quality_metric", "quality_value", "latency_median_ms",
    "latency_p95_ms", "throughput", "peak_memory_mb", "energy_j", "status", "notes",
]  # fmt: skip


class GPUBusyError(RuntimeError):
    """Another process is using the GPU; measurements would be unreliable."""


# --------------------------------------------------------------------------
# Environment checks and power sampling
# --------------------------------------------------------------------------


def other_gpu_processes() -> list[int]:
    """PIDs of compute processes on the GPU other than this one."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return [p for p in pids if p != os.getpid()]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return os.environ.get("XSHARE_COMMIT", "unknown")


class PowerSampler:
    """Samples GPU board power (watts) with ``nvidia-smi`` in a background thread."""

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.field = self._pick_field()
        self.samples: list[tuple[float, float]] = []
        self._proc = None
        self._thread = None

    def _pick_field(self) -> str:
        # power.draw.instant is a single reading; plain power.draw may be a
        # one-second average on recent drivers. Prefer the instant value.
        for field in ("power.draw.instant", "power.draw"):
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index), f"--query-gpu={field}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )  # fmt: skip
            try:
                float(out.stdout.strip())
                return field
            except ValueError:
                continue
        raise RuntimeError("nvidia-smi reports no usable power reading")

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["nvidia-smi", "-i", str(self.gpu_index), f"--query-gpu={self.field}",
             "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )  # fmt: skip
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def energy(self, t0: float, t1: float) -> Optional[tuple[float, float]]:
        """(joules, mean watts) between two ``perf_counter`` times, or None.

        Trapezoidal integration of the samples inside the window, with the
        power at the window edges taken from the nearest sample.
        """
        inside = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        before = [(t, w) for t, w in self.samples if t < t0]
        after = [(t, w) for t, w in self.samples if t > t1]
        points = ([(t0, before[-1][1])] if before else []) + inside
        points += [(t1, after[0][1])] if after else []
        if len(points) < 2:
            return None
        joules = sum((tb - ta) * (wa + wb) / 2 for (ta, wa), (tb, wb) in zip(points, points[1:]))
        span = points[-1][0] - points[0][0]
        return joules, (joules / span if span > 0 else points[0][1])

    def idle_power(self, seconds: float) -> Optional[float]:
        t0 = time.perf_counter()
        time.sleep(seconds)
        result = self.energy(t0, time.perf_counter())
        return None if result is None else result[1]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _synchronize(device) -> None:
    """Wait for queued GPU work before reading the clock; a no-op on the CPU."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_batch(backbone, controller: XShareController, prompts, max_new_tokens, eos_ids):
    """Greedy decoding of one batch with timings per step and per request."""
    model, tokenizer = backbone.model, backbone.tokenizer
    device = backbone.device
    enc = {k: v.to(device) for k, v in gsm8k.collate_prompts(prompts, tokenizer).items()}
    batch = enc["input_ids"].shape[0]
    eos = torch.tensor(sorted(eos_ids), device=device)
    pad = tokenizer.pad_token_id

    controller.valid_rows = None
    _synchronize(device)
    t_start = time.perf_counter()
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
    next_token = out.logits[:, -1].argmax(dim=-1)
    _synchronize(device)
    t_prefill_end = time.perf_counter()

    past = out.past_key_values
    attention_mask = enc["attention_mask"]
    ones = torch.ones(batch, 1, dtype=attention_mask.dtype, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    finish_idx = torch.full((batch,), -1, dtype=torch.long, device=device)
    generated, token_time, step_ms = [], [], []
    live_tokens = 0
    produced_at = t_prefill_end  # when ``next_token`` became available

    for j in range(max_new_tokens):
        is_eos = torch.isin(next_token, eos)
        finish_idx = torch.where(is_eos & ~finished, j, finish_idx)
        finished = finished | is_eos
        generated.append(next_token.masked_fill(finished, pad))
        token_time.append(produced_at)
        live = int((~finished).sum())  # also synchronises before the step timer
        if live == 0 or j == max_new_tokens - 1:
            break
        live_tokens += live
        t_step = time.perf_counter()
        controller.valid_rows = ~finished
        attention_mask = torch.cat([attention_mask, ones], dim=1)
        out = model(
            input_ids=generated[-1].unsqueeze(1),
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1].argmax(dim=-1)
        _synchronize(device)
        produced_at = time.perf_counter()
        step_ms.append((produced_at - t_step) * 1000.0)
    t_end = time.perf_counter()
    controller.valid_rows = None

    tokens = torch.stack(generated, dim=1)
    finish = finish_idx.tolist()
    last = len(generated) - 1
    latencies = [token_time[f if f >= 0 else last] - t_start for f in finish]
    new_tokens = [f if f >= 0 else len(generated) for f in finish]
    return {
        "texts": tokenizer.batch_decode(tokens, skip_special_tokens=True),
        "prompt_tokens": int(enc["attention_mask"].sum()),
        "live_tokens": live_tokens,
        "prefill_s": t_prefill_end - t_start,
        "decode_s": t_end - t_prefill_end,
        "step_ms": step_ms,
        "latency_s": latencies,
        "new_tokens": new_tokens,
        "t_start": t_start,
        "t_end": t_end,
    }


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------


def method_config(method: dict, defaults: dict, top_k: int, n_experts: int, gate_norm: str):
    """XShareConfig for a method entry, or None for the unpruned baseline."""
    if "budget" not in method:
        return None
    budget = n_experts if method["budget"] == "all" else int(method["budget"])
    return XShareConfig(
        top_k=top_k,
        warmup_k0=int(method.get("warmup_k0", 0)),
        budget=budget,
        budget_mode=defaults.get("budget_mode", "additional"),
        score=defaults.get("score", "softmax"),
        gate_norm=gate_norm,
        min_keep=defaults.get("min_keep"),
    )


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def run_cell(backbone, controller, sampler, idle_w, examples, prompts, batch_size, config, gen):
    controller.config = config
    batches = list(zip(gsm8k.make_batches(examples, batch_size), gsm8k.make_batches(prompts, batch_size)))
    eos_ids = backbone.eos_token_ids()
    generate_batch(backbone, controller, batches[0][1], gen["warmup_tokens"], eos_ids)  # warm-up
    controller.reset_stats()
    torch.cuda.reset_peak_memory_stats()

    records, step_ms = [], []
    prefill_s = decode_s = 0.0
    live_tokens = prompt_tokens = 0
    joules = window_s = 0.0
    energy_ok = sampler is not None
    for batch_examples, batch_prompts in batches:
        out = generate_batch(backbone, controller, batch_prompts, gen["max_new_tokens"], eos_ids)
        prefill_s += out["prefill_s"]
        decode_s += out["decode_s"]
        live_tokens += out["live_tokens"]
        prompt_tokens += out["prompt_tokens"]
        step_ms.extend(out["step_ms"])
        if energy_ok:
            measured = sampler.energy(out["t_start"], out["t_end"])
            if measured is None:
                energy_ok = False
            else:
                joules += measured[0]
                window_s += out["t_end"] - out["t_start"]
        for example, text, latency, n_new in zip(
            batch_examples, out["texts"], out["latency_s"], out["new_tokens"]
        ):
            records.append(
                {
                    "index": example["index"],
                    "gold": example["answer"],
                    "prediction": gsm8k.extract_answer(text),
                    "correct": gsm8k.is_correct(text, example["answer"]),
                    "latency_s": round(latency, 4),
                    "new_tokens": n_new,
                    "generation": text,
                }
            )

    latencies = [r["latency_s"] for r in records]
    generated = live_tokens + len(records)  # decode tokens + the first token from prefill
    energy = None
    if energy_ok and window_s > 0:
        above = joules - idle_w * window_s if idle_w is not None else None
        energy = {
            "source": f"nvidia-smi {sampler.field}, every {sampler.interval_ms} ms, whole board",
            "window_s": window_s,
            "mean_w": joules / window_s,
            "idle_w": idle_w,
            "joules": joules,
            "joules_above_idle": above,
            "tokens": generated,
            "j_per_token": joules / generated,
            "j_per_token_above_idle": None if above is None else above / generated,
        }
    return {
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "num_examples": len(records),
        "decode_tokens_per_s": live_tokens / decode_s if decode_s else None,
        "decode_step_median_ms": statistics.median(step_ms) if step_ms else None,
        "decode_step_p95_ms": percentile(step_ms, 0.95),
        "decode_step_ms": [round(x, 3) for x in step_ms],
        "request_latency_median_s": statistics.median(latencies),
        "request_latency_p95_s": percentile(latencies, 0.95),
        "prefill_s_total": prefill_s,
        "decode_s_total": decode_s,
        "generated_tokens_timed": live_tokens,
        "prompt_tokens": prompt_tokens,
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "energy": energy,
        "routing": controller.stats(),
        "records": records,
    }


def save_capture(backbone, controller, prompts, cfg, eos_ids, path: Path) -> None:
    """Store decode-step router logits of one batch: ``[steps, layers, batch, experts]``."""
    controller.config = None
    controller.capture = []
    generate_batch(backbone, controller, prompts, cfg["generation"]["max_new_tokens"], eos_ids)
    capture, controller.capture = controller.capture, None
    n_layers = len(backbone.model.model.layers)
    steps = min(cfg["capture_router_logits"]["steps"], len(capture) // n_layers)
    logits = torch.stack(
        [torch.stack([capture[s * n_layers + layer][1] for layer in range(n_layers)]) for s in range(steps)]
    )
    valid = torch.stack([capture[s * n_layers][2] for s in range(steps)])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"router_logits": logits, "valid": valid}, path)


# --------------------------------------------------------------------------
# Result rows
# --------------------------------------------------------------------------


def csv_row(result: dict, cfg: dict) -> dict:
    meta = result["metadata"]
    stats = result["routing"]
    energy = result.get("energy")
    if energy is None:
        energy_note = "energy not_measured"
    else:
        energy_note = (
            f"energy=whole-board {energy['source'].split(',')[0]}, prefill+decode, "
            f"{energy['j_per_token']:.3f} J/token"
        )
        if energy.get("j_per_token_above_idle") is not None:
            energy_note += (
                f", idle {energy['idle_w']:.1f} W, above idle {energy['j_per_token_above_idle']:.3f} J/token"
            )
    return {
        "experiment_id": result["experiment_id"],
        "paper_id": cfg["paper_id"],
        "method": result["method"],
        "commit": result["commit"],
        "backbone_id": meta["backbone_id"],
        "backbone_revision": meta["revision"],
        "dataset_id": cfg["dataset"]["id"],
        "split": f"{cfg['dataset']['split']} (subset {result['subset']} of seed {cfg['dataset']['seed']})",
        "seed": cfg["dataset"]["seed"],
        "device": f"{meta['device']} (driver {meta['driver']})",
        "framework_version": (
            f"torch {meta['torch']} (CUDA {meta['cuda_runtime']}); transformers {meta['transformers']}; "
            f"attention {meta['attn_implementation']}; experts {meta['experts_implementation']}"
        ),
        "dtype": meta["dtype"],
        "batch_size": result["batch_size"],
        "input_length": round(result["prompt_tokens"] / result["num_examples"], 1),
        "output_length": cfg["generation"]["max_new_tokens"],
        "quality_metric": "gsm8k_exact_match_flexible_extract",
        "quality_value": round(result["accuracy"], 4),
        "latency_median_ms": round(result["decode_step_median_ms"], 3),
        "latency_p95_ms": round(result["decode_step_p95_ms"], 3),
        "throughput": round(result["decode_tokens_per_s"], 2),
        "peak_memory_mb": round(result["peak_memory_allocated_mb"], 1),
        "energy_j": "" if energy is None else round(energy["joules"], 1),
        "status": "partial",
        "notes": (
            f"n={result['num_examples']}; latency=one decode step of the whole batch; "
            f"request latency median {result['request_latency_median_s']:.2f} s "
            f"p95 {result['request_latency_p95_s']:.2f} s; throughput=decode tokens/s; "
            f"prefill {result['prefill_s_total']:.1f} s total; memory=torch max_memory_allocated; "
            f"{energy_note}; warm-up 1 pass of {cfg['generation']['warmup_tokens']} tokens; "
            f"activated experts/layer {stats.get('mean_activated_experts', 0):.2f} of {meta['n_experts']}, "
            f"kept {stats.get('mean_kept_experts', 0):.2f}; dropped tokens 0; "
            f"static batching, Hugging Face transformers; not comparable to the paper's vLLM numbers"
        ),
    }


def write_csv(out_dir: Path, cfg: dict) -> Path:
    rows = []
    for path in sorted(out_dir.glob("xshare-gsm8k-*.json")):
        rows.append(csv_row(json.loads(path.read_text()), cfg))
    out = out_dir / "results.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    import yaml

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=None, help="run only these method names")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--subsets", type=int, nargs="+", default=None)
    p.add_argument("--no-capture", action="store_true", help="skip saving router logits")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("calibration_dataset") is not None:
        raise ValueError("XShare is calibration-free; calibration_dataset must be null")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = cfg["dataset"]
    methods = [m for m in cfg["methods"] if args.methods is None or m["name"] in args.methods]
    subsets = args.subsets or data["subsets"]
    commit = git_commit()

    for batch_size in args.batch_sizes or cfg["batch_sizes"]:
        todo = [
            (s, m)
            for s in subsets
            for m in methods
            if not (args.out_dir / f"xshare-gsm8k-{m['name']}-c{batch_size}-s{s}.json").exists()
        ]
        capture_path = args.out_dir / f"router_logits_c{batch_size}.pt"
        need_capture = not args.no_capture and not capture_path.exists()
        if not todo and not need_capture:
            print(f"batch {batch_size}: all cells done", flush=True)
            continue
        if other_gpu_processes():
            raise GPUBusyError(f"other GPU processes: {other_gpu_processes()}")

        backbone = HFCausalLMBackbone.load(cfg["backbone_id"], device="cuda")
        meta = backbone.get_model_metadata()
        gate_norm = "softmax_topk_renorm" if meta["norm_topk_prob"] else "softmax_topk"
        controller = install_xshare(backbone.model, None)
        sampler, idle_w = None, None
        if cfg["energy"]["enabled"]:
            sampler = PowerSampler(cfg["energy"]["interval_ms"])
            sampler.start()
            idle_w = sampler.idle_power(cfg["energy"]["idle_seconds"])
        print(f"batch {batch_size}: model loaded in {meta['load_time_s']:.1f} s, idle {idle_w} W", flush=True)

        prompts_by_subset = {}
        for s in subsets:
            examples = gsm8k.load_gsm8k_subset(data["num_samples"], data["seed"], data["split"], s)
            prompts = backbone.preprocess([gsm8k.build_prompt(e["question"]) for e in examples])
            prompts_by_subset[s] = (examples, prompts)

        try:
            for s, method in todo:
                if other_gpu_processes():
                    raise GPUBusyError(f"other GPU processes: {other_gpu_processes()}")
                config = method_config(method, cfg.get("xshare", {}), meta["top_k"], meta["n_experts"], gate_norm)
                examples, prompts = prompts_by_subset[s]
                t0 = time.time()
                result = run_cell(
                    backbone, controller, sampler, idle_w, examples, prompts, batch_size, config, cfg["generation"]
                )
                result.update(
                    {
                        "experiment_id": f"xshare-gsm8k-{method['name']}-c{batch_size}-s{s}",
                        "method": method["name"],
                        "xshare_config": None if config is None else config.__dict__,
                        "batch_size": batch_size,
                        "subset": s,
                        "seed": data["seed"],
                        "max_new_tokens": cfg["generation"]["max_new_tokens"],
                        "warmup_tokens": cfg["generation"]["warmup_tokens"],
                        "dataset_id": data["id"],
                        "dataset_revision": gsm8k.HF_REVISION,
                        "commit": commit,
                        "n_experts": meta["n_experts"],
                        "top_k": meta["top_k"],
                        "dropped_tokens": 0,
                        "metadata": meta,
                        "wall_time_s": time.time() - t0,
                    }
                )
                path = args.out_dir / f"{result['experiment_id']}.json"
                path.write_text(json.dumps(result, indent=1, default=str))
                stats = result["routing"]
                print(
                    f"{result['experiment_id']}: acc {result['accuracy']:.3f}, "
                    f"{result['decode_tokens_per_s']:.1f} tok/s, activated "
                    f"{stats.get('mean_activated_experts', 0):.1f}, {result['wall_time_s']:.0f} s",
                    flush=True,
                )
            if need_capture:
                examples, prompts = prompts_by_subset[subsets[0]]
                save_capture(backbone, controller, prompts[:batch_size], cfg, backbone.eos_token_ids(), capture_path)
                print(f"saved {capture_path}", flush=True)
        finally:
            if sampler is not None:
                sampler.stop()
            del backbone, controller
            torch.cuda.empty_cache()

    print(f"wrote {write_csv(args.out_dir, cfg)}", flush=True)


if __name__ == "__main__":
    main()
