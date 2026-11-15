#!/usr/bin/env python
"""BASTION reproduction harness (AR vs DFlash vs BASTION, T=0, batch size 1).

Derived from ``bastion/benchmark.py::_run_transformers`` of the official BASTION
code release (https://github.com/kaist-ai-osi-lab/BASTION, MIT licence,
Copyright (c) 2026 Z Lab), pinned as the git submodule ``./upstream`` at commit 1989b80.
See ``upstream/LICENSE``.

What is kept identical to upstream ``_run_transformers``:
  * model loading (``AutoModelForCausalLM`` / ``DFlashDraftModel`` in bf16, one
    target copy shared by all modes), attention-backend selection
    (``flash_attention_2`` for AR/DFlash if flash-attn is importable, else
    ``sdpa``; ``sdpa`` always for BASTION verification),
  * the calibration / profiling cache logic and file paths under
    ``upstream/cache`` (copied below, distributed branches removed),
  * prompt construction (chat template, ``enable_thinking=False``), stop token,
    the order AR -> DFlash -> BASTION per turn, multi-turn history taken from
    the AR output,
  * the generation functions themselves, which are imported, not copied
    (``dflash.model.dflash_generate``, ``bastion.tree_draft.bastion_generate``).

Additions (see README "How to run"): several datasets per process, per-dataset
sample caps, ``--modes``, fixed verification budgets, per-step tree-size
logging, incremental JSONL output with resume, run metadata.
Not supported: torchrun / multi-GPU sharding, temperature > 0 was not the
target of this harness (it is passed through but untested).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import repro_lib as rl  # puts ./upstream on sys.path

from bastion import benchmark as ub  # noqa: E402  (upstream module, unmodified)
from bastion import cost_model, tree_draft  # noqa: E402

CACHE_DIR = ub.CACHE_DIR  # == upstream/cache, the same files upstream reads/writes


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _hf_cached_revision(repo_id: str) -> dict:
    """Resolve the locally cached snapshot of a model repo without network access."""
    info: dict[str, Any] = {"repo_id": repo_id}
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    except Exception:
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    info["refs_main"] = ref.read_text().strip() if ref.exists() else None
    snaps = repo_dir / "snapshots"
    info["snapshots"] = sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []
    return info


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_env_metadata(torch, device) -> dict:
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ["torch", "transformers", "datasets", "numpy", "scipy", "accelerate", "flash-attn", "bastion", "huggingface-hub"]
        },
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        meta["gpu_name"] = torch.cuda.get_device_name(device)
        meta["gpu_total_memory_mib"] = round(props.total_memory / 2**20)
        meta["gpu_capability"] = f"{props.major}.{props.minor}"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        meta["nvidia_smi"] = out
    except Exception:
        meta["nvidia_smi"] = None
    return meta


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - metadata only
        return {"error": repr(exc)}


class RunMeta:
    """``<run>.meta.json``: one entry per process invocation ("session")."""

    def __init__(self, path: Path, run_name: str) -> None:
        self.path = path
        self.data = _read_json(path) if path.exists() else {"run_name": run_name, "sessions": []}
        if "sessions" not in self.data:
            self.data = {"run_name": run_name, "sessions": []}
        self.session: dict[str, Any] = {"start": _now(), "status": "starting"}
        self.data["sessions"].append(self.session)

    def update(self, **kwargs) -> None:
        self.session.update(rl.to_jsonable(kwargs))
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# calibration / profiling (copied from upstream _run_transformers, single process)
# ---------------------------------------------------------------------------

def ensure_calibration_and_profile(*, args, target, draft_model, tokenizer, device, block_size,
                                   model_key, gpu_key, probe_turn: str, bastion_attn_impl: str) -> dict:
    """Upstream lines 429-619 (commit 1989b80) with the torch.distributed branches removed.

    Cache paths, cache-validity rules, probe collection, fit, and profiling
    are unchanged, so caches written by ``python -m bastion.benchmark`` (or
    ``tools/bastion_local.py``) are reused and vice versa.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    calib_cache = CACHE_DIR / f"calibration_{gpu_key}_{model_key}.json"
    profile_cache = CACHE_DIR / f"calibration_{gpu_key}_{model_key}.profile.json"

    sigma_scale = float(os.environ.get("BASTION_SIGMA_SCALE", "30.0"))
    fit_version = "sigma_sequence_v1"
    probe_path = CACHE_DIR / f"calib_probe_{gpu_key}_{model_key}.{fit_version}.jsonl"
    created = {"calibration_probe": False, "calibration_fit": False, "profile": False}

    def _probe_cache_attn_impl(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        return json.loads(line).get("attn_impl")
        except Exception:
            return None
        return None

    def _calibration_cache_current(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        metadata_current = (
            isinstance(cached.get("fit_parameters"), dict)
            and cached.get("fit_version") == fit_version
            and float(cached.get("sigma_scale", -1.0)) == sigma_scale
        )
        if not metadata_current:
            return False
        cached_attn_impl = cached.get("attn_impl")
        if cached_attn_impl is None:
            cached_attn_impl = _probe_cache_attn_impl(probe_path)
            if cached_attn_impl == bastion_attn_impl:
                cached["attn_impl"] = cached_attn_impl
                path.write_text(json.dumps(cached), encoding="utf-8")
        return cached_attn_impl == bastion_attn_impl

    ub._set_attn_impl(target, bastion_attn_impl)

    if not _calibration_cache_current(calib_cache):
        if not probe_path.exists():
            print(f"[calibration] collecting probe data -> {probe_path} (several minutes)")
            # inference_mode: upstream's untimed context prefill otherwise keeps
            # autograd activations alive and OOMs Qwen3-8B on 32 GB (timed
            # forwards are already in inference mode upstream). See tools/bastion_local.py.
            import torch

            with torch.inference_mode():
                cost_model.collect_calibration_data(
                    target=target,
                    tokenizer=tokenizer,
                    device=device,
                    model_name_or_path=args.model,
                    gpu_type=gpu_key,
                    output_path=probe_path,
                    attn_implementation=bastion_attn_impl,
                )
            created["calibration_probe"] = True
        params = cost_model.fit_linear_calibration(probe_path, sigma_scale=sigma_scale)
        calib_cache.write_text(
            json.dumps({"fit_parameters": params, "fit_version": fit_version, "sigma_scale": sigma_scale}),
            encoding="utf-8",
        )
        created["calibration_fit"] = True

    if not _calibration_cache_current(calib_cache):
        raise RuntimeError(f"Calibration cache was not created or is stale: {calib_cache}")
    os.environ["CALIBRATION_JSON_PATH"] = str(calib_cache)

    if not profile_cache.exists():
        print(f"[profile] profiling draft/non-verify latency -> {profile_cache}")
        probe_messages = [{"role": "user", "content": probe_turn}]
        probe_text = tokenizer.apply_chat_template(
            probe_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        probe_input_ids = tokenizer.encode(probe_text, return_tensors="pt").to(device)
        dl_s, nv_s = ub.profile_tree_adaptive_constants(
            draft_model=draft_model,
            target=target,
            input_ids=probe_input_ids,
            block_size=block_size,
            temperature=args.temperature,
            model_name=model_key,
            gpu_type=gpu_key,
            num_trials=3,
        )
        profile_cache.write_text(
            json.dumps({"draft_latency_s": dl_s, "non_verify_latency_s": nv_s, "attn_impl": bastion_attn_impl}),
            encoding="utf-8",
        )
        created["profile"] = True

    prof = json.loads(profile_cache.read_text(encoding="utf-8"))
    if "draft_latency_s" not in prof or "non_verify_latency_s" not in prof:
        raise ValueError(f"Invalid profile cache: {profile_cache}")
    if prof.get("attn_impl") is None:
        prof_attn_impl = _probe_cache_attn_impl(probe_path)
        if prof_attn_impl == bastion_attn_impl:
            prof["attn_impl"] = prof_attn_impl
            profile_cache.write_text(json.dumps(prof), encoding="utf-8")
    if prof.get("attn_impl") != bastion_attn_impl:
        raise ValueError(
            f"Invalid profile cache attention backend in {profile_cache}: "
            f"expected {bastion_attn_impl}, got {prof.get('attn_impl')}"
        )
    draft_latency_s = float(prof["draft_latency_s"])
    non_verify_latency_s = float(prof["non_verify_latency_s"])
    if draft_latency_s <= 0 or non_verify_latency_s <= 0:
        raise ValueError(f"Invalid profiling constants in {profile_cache}")

    probe_lines = None
    if probe_path.exists():
        with probe_path.open(encoding="utf-8") as fh:
            probe_lines = sum(1 for line in fh if line.strip())
    return {
        "draft_latency_s": draft_latency_s,
        "non_verify_latency_s": non_verify_latency_s,
        "calibration_cache": str(calib_cache),
        "calibration_cache_contents": _read_json(calib_cache),
        "profile_cache": str(profile_cache),
        "profile_cache_contents": prof,
        "probe_path": str(probe_path),
        "probe_sha256": rl.sha256_file(probe_path),
        "probe_num_records": probe_lines,
        "sigma_scale": sigma_scale,
        "created_this_session": created,
    }


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def load_selected_datasets(names: list[str], default_cap: int | None, caps: dict[str, int]) -> dict[str, dict]:
    out = {}
    for name in names:
        if name in ub.DATASET_GROUPS:
            raise ValueError(f"group alias {name!r} not allowed here; list the datasets explicitly")
        full = ub.load_and_process_dataset(name)  # upstream loader + upstream/cache/<name>.jsonl
        cap = caps.get(name, default_cap)
        idx = rl.upstream_selection_indices(len(full), cap)
        out[name] = {
            "rows": [full[i] for i in idx],
            "orig_indices": idx,
            "dataset_size": len(full),
            "cap": cap,
            "jsonl": str(CACHE_DIR / f"{name}.jsonl"),
            "jsonl_sha256": rl.sha256_file(CACHE_DIR / f"{name}.jsonl"),
            "hf_source": {
                "load_args": list(ub.DATASETS[name]["load_args"]),
                "load_kwargs": {k: v for k, v in ub.DATASETS[name]["load_kwargs"].items()},
            },
        }
    return out


# ---------------------------------------------------------------------------
# summary (same table as upstream _print_decode_summary)
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], block_size: int, datasets: list[str]) -> None:
    def ns(rec):
        return SimpleNamespace(
            time_per_output_token=rec["time_per_output_token"],
            acceptance_lengths=rec.get("acceptance_lengths") or [],
        )

    by_key: dict[tuple, dict] = {}
    for rec in records:
        key = (rec["dataset"], rec["sample_idx"], rec["turn"])
        slot = by_key.setdefault(key, {})
        if rec["mode"] == "ar":
            slot[1] = ns(rec)
        elif rec["mode"] == "dflash":
            slot[block_size] = ns(rec)
        elif rec["mode"] == "bastion" and rec.get("fixed_budget") is None:
            slot[block_size * 10] = ns(rec)

    groups = [(ds, [k for k in by_key if k[0] == ds]) for ds in datasets]
    if len(datasets) > 1:
        groups.append(("ALL (pooled over records, as upstream's group alias would)", list(by_key)))
    for label, keys in groups:
        responses = [by_key[k] for k in keys if {1, block_size, block_size * 10} <= set(by_key[k])]
        print(f"\n### {label}: {len(responses)} responses with AR, DFlash and adaptive BASTION")
        if responses:
            ub._print_decode_summary(responses, block_size)

    # Fixed-budget runs (not part of the upstream table).
    fixed = sorted({r["fixed_budget"] for r in records if r["mode"] == "bastion" and r.get("fixed_budget") is not None})
    if fixed:
        print("\nFixed-budget BASTION (pooled over this run's records; speedup needs AR on the same keys):")
        ar = {(r["dataset"], r["sample_idx"], r["turn"]): r for r in records if r["mode"] == "ar"}
        for budget in fixed:
            recs = [r for r in records if r["mode"] == "bastion" and r.get("fixed_budget") == budget]
            tpot = np.mean([r["time_per_output_token"] for r in recs])
            aal = np.mean([a for r in recs for a in r["acceptance_lengths"]])
            sizes = [s for r in recs for s in (r.get("tree_sizes") or [])]
            matched = [r for r in recs if (r["dataset"], r["sample_idx"], r["turn"]) in ar]
            sp = "-"
            if matched:
                sp = "%.2fx" % (
                    np.mean([ar[(r["dataset"], r["sample_idx"], r["turn"])]["time_per_output_token"] for r in matched])
                    / np.mean([r["time_per_output_token"] for r in matched])
                )
            print(f"  N={budget:<5d} n={len(recs):<4d} {1 / tpot:8.2f} tok/s  speedup={sp:>7}  AAL={aal:.2f}  "
                  f"mean tree size={np.mean(sizes) if sizes else float('nan'):.1f}")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def _gen_stats(resp, num_input_tokens: int) -> dict:
    out_ids = resp.output_ids[0, num_input_tokens:].tolist()
    stats = {
        "num_input_tokens": int(resp.num_input_tokens),
        "num_output_tokens": int(resp.num_output_tokens),
        "time_per_output_token": float(resp.time_per_output_token),
        "time_to_first_token": None if getattr(resp, "time_to_first_token", None) is None else float(resp.time_to_first_token),
        "acceptance_lengths": [int(a) for a in resp.acceptance_lengths],
        "num_steps": len(resp.acceptance_lengths),
        "output_ids": [int(t) for t in out_ids],
    }
    for name in ["draft_latencies", "verification_latencies", "tree_build_latencies",
                 "tree_select_latencies", "tree_kv_reorder_latencies", "tree_extract_latencies"]:
        values = getattr(resp, name, None)
        if values:
            stats[name.replace("latencies", "latency") + "_mean_s"] = float(np.mean(values))
    if getattr(resp, "verification_latencies", None):
        stats["verification_latencies"] = rl.round_list(resp.verification_latencies)
    return stats


def run_loop(*, datasets, selected, plan, generate, tokenizer, device, writer, recorder, history_records,
             base_fields, enable_thinking, max_new_tokens, eos, block_size, mem_reset, mem_peak_mib, attn_impl) -> int:
    """Per-turn generation loop: AR -> DFlash -> BASTION budgets, one JSONL record each.

    ``generate(mode, input_ids, budget, max_new_tokens)`` returns upstream's
    stats namespace. Separated from ``run`` so it can be tested on CPU.
    """
    n_new = 0
    for ds_name in datasets:
        info = selected[ds_name]
        rows = info["rows"]
        print(f"\n=== {ds_name}: {len(rows)} samples (cap={info['cap']}, dataset size={info['dataset_size']})")
        for sample_idx, instance in enumerate(rows):
            messages: list[dict] = []
            history_sources: list[str] = []
            for turn, user_content in enumerate(instance["turns"]):
                messages.append({"role": "user", "content": user_content})
                input_text = ub._apply_chat_template(tokenizer, messages, enable_thinking)
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                ihash = rl.input_hash(input_ids[0].tolist())
                outputs: dict[tuple, list[int]] = {}
                for mode, budget in plan:
                    key = (ds_name, sample_idx, turn, mode, budget)
                    existing = writer.get(key)
                    if existing is not None:
                        # Resume keys on sample_idx (position in the selection). A different cap can map the
                        # same sample_idx to another problem (e.g. aime25 is unshuffled at cap>=30, shuffled
                        # below), which would silently mix problems into this run: refuse.
                        if (existing.get("orig_idx") is not None
                                and int(existing["orig_idx"]) != int(info["orig_indices"][sample_idx])):
                            raise RuntimeError(
                                f"resumed record {key} is for orig_idx={existing['orig_idx']} but the current "
                                f"selection has orig_idx={info['orig_indices'][sample_idx]} at this position "
                                "(cap/selection changed); use a new --run-name")
                        if existing.get("input_hash") != ihash:
                            print(f"[warn] resumed record {key} has a different input hash "
                                  f"({existing.get('input_hash')} vs {ihash})", file=sys.stderr)
                        outputs[(mode, budget)] = existing["output_ids"]
                        continue
                    mem_reset()
                    t0 = time.perf_counter()
                    if mode == "bastion":
                        with recorder.recording():
                            resp = generate(mode, input_ids, budget, max_new_tokens)
                        sizes, contexts = list(recorder.sizes), list(recorder.contexts)
                    else:
                        resp = generate(mode, input_ids, budget, max_new_tokens)
                        sizes, contexts = None, None
                    wall = time.perf_counter() - t0
                    stats = _gen_stats(resp, int(input_ids.shape[1]))
                    rec = {
                        **base_fields,
                        "dataset": ds_name,
                        "sample_idx": sample_idx,
                        "orig_idx": info["orig_indices"][sample_idx],
                        "turn": turn,
                        "num_turns": len(instance["turns"]),
                        "mode": mode,
                        "fixed_budget": budget,
                        "block_size": 1 if mode == "ar" else block_size,
                        "target_attn_impl": attn_impl(),
                        "input_hash": ihash,
                        "history_sources": list(history_sources),
                        **stats,
                        "hit_eos": bool(stats["output_ids"]) and stats["output_ids"][-1] == eos,
                        "wall_time_s": wall,
                        "peak_mem_allocated_mib": mem_peak_mib(),
                        "timestamp": _now(),
                    }
                    if mode == "bastion":
                        rec["tree_sizes"] = sizes
                        rec["tree_contexts"] = contexts
                        rec["mean_tree_size"] = float(np.mean(sizes)) if sizes else None
                        rec["tree_size_mismatch_steps"] = (
                            None if budget is None else int(sum(1 for x in sizes if x != budget))
                        )
                        if budget is not None and rec["tree_size_mismatch_steps"]:
                            print(f"[warn] {key}: {rec['tree_size_mismatch_steps']} steps with tree size != {budget}",
                                  file=sys.stderr)
                    writer.write(rec)
                    n_new += 1
                    outputs[(mode, budget)] = stats["output_ids"]
                    tag = mode if budget is None else f"{mode}[N={budget}]"
                    print(f"  {ds_name} #{sample_idx} t{turn} {tag:<16} out={stats['num_output_tokens']:5d} "
                          f"tpot={stats['time_per_output_token'] * 1e3:7.2f}ms "
                          f"aal={np.mean(stats['acceptance_lengths']):5.2f}", flush=True)

                if turn + 1 >= len(instance["turns"]):
                    continue
                # Multi-turn history: AR output (upstream); else external AR run; else first mode in plan.
                if ("ar", None) in outputs:
                    hist_ids, hist_src = outputs[("ar", None)], "ar"
                elif (ds_name, info["orig_indices"][sample_idx], turn) in history_records:
                    hrec = history_records[(ds_name, info["orig_indices"][sample_idx], turn)]
                    if hrec.get("input_hash") != ihash:
                        print(f"[warn] history AR record for {(ds_name, sample_idx, turn)} has a different "
                              "input hash; its output is still used as history", file=sys.stderr)
                    hist_ids, hist_src = hrec["output_ids"], f"external-ar:{hrec.get('run_name')}"
                else:
                    m, b = plan[0]
                    hist_ids, hist_src = outputs[(m, b)], (m if b is None else f"{m}[N={b}]")
                output_text = tokenizer.decode(hist_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})
                history_sources.append(hist_src)
    return n_new


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash.model import DFlashDraftModel, dflash_generate

    env_values = rl.check_method_env(args.allow_env_overrides)
    ub._check_transformers_model(args.model)
    budgets = rl.parse_budgets(args.budgets) if args.fixed_budget is None else [int(args.fixed_budget)]
    if args.fixed_budget is not None and args.fixed_budget < 2:
        raise ValueError("--fixed-budget must be >= 2")
    if "bastion" not in args.modes:
        budgets = []
    caps = rl.parse_caps(args.cap)
    datasets = list(dict.fromkeys(d for item in args.datasets for d in (ub.DATASET_GROUPS.get(item) or [item])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.run_name}.jsonl"
    meta = RunMeta(out_dir / f"{args.run_name}.meta.json", args.run_name)
    meta.update(
        argv=sys.argv, args=vars(args), budgets=budgets, datasets=datasets, caps=caps,
        method_env=env_values,
        upstream_commit=_git(["rev-parse", "HEAD"], rl.UPSTREAM),
        upstream_dirty=bool(_git(["status", "--porcelain"], rl.UPSTREAM)),
        da4022_commit=_git(["rev-parse", "HEAD"], rl.ROOT),
        da4022_dirty_files=(_git(["status", "--porcelain", "--", ".", ":!runs", ":!results"], rl.ROOT) or "").splitlines(),
        cache_dir=str(CACHE_DIR),
    )

    # Seeds: upstream hard-codes 0 for all of these; --seed defaults to 0.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (upstream cost model and timing use CUDA).")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    gpu_registration = rl.register_local_gpu_specs()
    baseline_attn_impl = ub._get_baseline_attn_impl()
    bastion_attn_impl = "sdpa"

    # Dataset selection first (cheap) so a bad dataset name fails before model loading.
    selected = load_selected_datasets(datasets, args.max_samples, caps)
    meta.update(
        env=collect_env_metadata(torch, device),
        gpu_registration=gpu_registration,
        dataset_selection={
            name: {k: v for k, v in d.items() if k != "rows"} for name, d in selected.items()
        },
    )

    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=baseline_attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    bastion_target = target  # one target copy serves every method (as upstream)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=baseline_attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.eos_token_id

    model_key = ub.infer_cost_model_name_from_target(args.model)
    gpu_key = ub.infer_cost_gpu_type_from_device(device)
    first_ds = selected[datasets[0]]["rows"]
    calib = ensure_calibration_and_profile(
        args=args, target=bastion_target, draft_model=draft_model, tokenizer=tokenizer, device=device,
        block_size=block_size, model_key=model_key, gpu_key=gpu_key,
        probe_turn=first_ds[0]["turns"][0], bastion_attn_impl=bastion_attn_impl,
    )
    draft_latency_s = calib["draft_latency_s"]
    non_verify_latency_s = calib["non_verify_latency_s"]

    meta.update(
        model_key=model_key, gpu_key=gpu_key, block_size=block_size,
        hf_revisions={
            "target": {**_hf_cached_revision(args.model), "config_commit_hash": getattr(target.config, "_commit_hash", None)},
            "draft": {**_hf_cached_revision(args.draft_model), "config_commit_hash": getattr(draft_model.config, "_commit_hash", None)},
        },
        attention_backends={
            "ar": {"target": baseline_attn_impl, "draft": "unused (block_size=1)"},
            "dflash": {"target": baseline_attn_impl, "draft": baseline_attn_impl},
            "bastion": {"target": bastion_attn_impl, "draft": baseline_attn_impl},
        },
        calibration=calib,
        eos_token_id=eos,
        status="running",
    )

    recorder = rl.TreeSizeRecorder().install()
    writer = rl.JsonlWriter(jsonl_path)
    n_existing = len(writer.records)
    if n_existing:
        print(f"[resume] {jsonl_path}: {n_existing} records already present, skipping those keys")

    def _bastion_kwargs(budget: int | None) -> dict:
        kw = dict(
            cost_model_name=model_key,
            cost_gpu_type=gpu_key,
            draft_latency_s=draft_latency_s,
            non_verify_latency_s=non_verify_latency_s,
        )
        if budget is not None:
            kw.update(min_tree_size=budget, max_tree_size=budget)
        return kw

    def generate(mode: str, input_ids, budget: int | None, max_new_tokens: int):
        if mode in ("ar", "dflash"):
            ub._set_attn_impl(target, baseline_attn_impl)
            return dflash_generate(
                draft_model, target=target, input_ids=input_ids, max_new_tokens=max_new_tokens,
                stop_token_ids=[eos], temperature=args.temperature,
                block_size=1 if mode == "ar" else block_size, return_stats=True,
            )
        ub._set_attn_impl(bastion_target, bastion_attn_impl)
        return tree_draft.bastion_generate(
            draft_model, target=bastion_target, input_ids=input_ids, max_new_tokens=max_new_tokens,
            stop_token_ids=[eos], temperature=args.temperature, block_size=block_size,
            return_stats=True, **_bastion_kwargs(budget),
        )

    plan: list[tuple[str, int | None]] = [(m, None) for m in ("ar", "dflash") if m in args.modes]
    plan += [("bastion", b) for b in budgets]

    # Optional warm-up (upstream does none); outputs discarded, recorder inactive.
    if args.warmup > 0:
        msgs = [{"role": "user", "content": first_ds[0]["turns"][0]}]
        ids = tokenizer.encode(ub._apply_chat_template(tokenizer, msgs, args.enable_thinking), return_tensors="pt").to(device)
        for _ in range(args.warmup):
            for mode, budget in plan:
                generate(mode, ids, budget, args.warmup_tokens)
        torch.cuda.synchronize()

    history_records: dict[tuple, dict] = {}
    if args.history_jsonl:
        for rec in rl.load_jsonl(args.history_jsonl):
            if rec.get("mode") == "ar" and rec.get("model") == args.model:
                history_records[(rec["dataset"], int(rec["orig_idx"]), int(rec["turn"]))] = rec
        print(f"[history] {len(history_records)} AR records loaded from {args.history_jsonl}")

    t_session = time.time()
    base_fields = {
        "run_name": args.run_name, "model": args.model, "draft_model": args.draft_model,
        "model_key": model_key, "gpu_key": gpu_key, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    }
    try:
        run_loop(
            datasets=datasets, selected=selected, plan=plan, generate=generate, tokenizer=tokenizer,
            device=device, writer=writer, recorder=recorder, history_records=history_records,
            base_fields=base_fields, enable_thinking=args.enable_thinking, max_new_tokens=args.max_new_tokens,
            eos=eos, block_size=block_size,
            mem_reset=lambda: torch.cuda.reset_peak_memory_stats(device),
            mem_peak_mib=lambda: torch.cuda.max_memory_allocated(device) / 2**20,
            attn_impl=lambda: target.config._attn_implementation,
        )
        meta.update(status="completed")
    except BaseException as exc:
        meta.update(status=f"aborted: {exc!r}")
        raise
    finally:
        recorder.uninstall()
        writer.close()
        meta.update(end=_now(), new_records=len(writer.records) - n_existing,
                    session_seconds=time.time() - t_session)

    print_summary(list(writer.records.values()), block_size, datasets)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="dataset names from upstream DATASETS (group alias paper-short is expanded)")
    p.add_argument("--max-samples", type=int, default=None, help="default per-dataset cap (upstream semantics)")
    p.add_argument("--cap", action="append", default=[], metavar="DATASET=N", help="per-dataset cap override")
    p.add_argument("--modes", nargs="+", choices=list(rl.MODES), default=list(rl.MODES))
    p.add_argument("--budgets", nargs="+", default=["adaptive"],
                   help="BASTION budgets: 'adaptive' and/or integers N (fixed tree of N nodes)")
    p.add_argument("--fixed-budget", type=int, default=None, help="shorthand for --budgets N")
    p.add_argument("--history-jsonl", default=None,
                   help="AR records of another run (matched on dataset, orig_idx, turn) used as multi-turn "
                        "history when AR is not run here")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="python/numpy/torch seed (upstream uses 0)")
    p.add_argument("--warmup", type=int, default=0, help="discarded warm-up passes over the plan (upstream: none)")
    p.add_argument("--warmup-tokens", type=int, default=128)
    p.add_argument("--out-dir", default=str(rl.ROOT / "runs"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--allow-env-overrides", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the sample selection and exit (no model loading)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.enable_thinking and any(x in args.model.lower() for x in ["qwen3-4b", "qwen3-8b"]):
        raise ValueError("DFlash drafters for Qwen3-4B/8B were not trained with thinking traces (upstream check).")
    if args.temperature != 0.0:
        print("[warn] temperature > 0 is passed through to upstream but this harness was only validated at T=0",
              file=sys.stderr)
    if args.dry_run:
        caps = rl.parse_caps(args.cap)
        datasets = list(dict.fromkeys(d for item in args.datasets for d in (ub.DATASET_GROUPS.get(item) or [item])))
        sel = load_selected_datasets(datasets, args.max_samples, caps)
        for name, d in sel.items():
            turns = sum(len(r["turns"]) for r in d["rows"])
            print(f"{name}: size={d['dataset_size']} cap={d['cap']} selected={len(d['rows'])} turns={turns} "
                  f"orig_idx[:10]={d['orig_indices'][:10]}")
        print("budgets:", rl.parse_budgets(args.budgets) if args.fixed_budget is None else [args.fixed_budget])
        return
    run(args)


if __name__ == "__main__":
    main()
