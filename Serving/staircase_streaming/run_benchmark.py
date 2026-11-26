"""Single reproduction command: baseline MoA vs staircase streaming on identical prompts.

From the repository root::

    python -m Serving.staircase_streaming.run_benchmark --config Serving/staircase_streaming/configs/default.yaml
    python -m Serving.staircase_streaming.run_benchmark --config ... --sanity   # 3 prompts, print texts

What it does:

1. loads the prompt subsets from the dataset registry (fixed seed);
2. warm-up requests (not recorded);
3. runs every variant on every prompt, interleaved per prompt so slow drift (thermal,
   caches) affects all variants equally. Variants: baseline, staircase sweep over
   first_chunk_size x redundancy, and the prefix-cache template at the default
   setting;
4. GSM8K exact match for baseline vs default staircase;
5. writes raw per-request JSONL (resumable), ``metrics.csv`` (exact repository schema),
   ``summary_detailed.csv``, ``summary.json`` and plots into ``experiment.output_dir``.

Unmeasured quantities are left empty, never 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import Backend, OpenAICompatBackend
from .config import build_backend, build_moa, build_staircase, load_config, model_metadata
from .moa_baseline import run_moa_baseline
from .quality import exact_match, extract_prediction, extract_reference, format_gsm8k
from .staircase import StaircaseConfig, run_staircase
from .timing import GpuMemorySampler, MoAResult, median, percentile

METRICS_COLUMNS = (
    "experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,"
    "device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,"
    "quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes"
).split(",")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_loader() -> Callable[..., list]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from Dataloaders.evaluation.text_prompts import load_prompts  # noqa: PLC0415

    return load_prompts


# ------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    id: str
    method: str  # baseline | staircase | staircase_prefix_cache
    staircase: StaircaseConfig | None = None


def build_variants(cfg: dict[str, Any]) -> list[Variant]:
    n = len(cfg["moa"]["proposers"])
    default = build_staircase(cfg)
    variants = [Variant("baseline", "baseline")]
    sweep = cfg["benchmark"].get("sweep") or {}
    for f in sweep.get("first_chunk_size", [default.first_chunk_size]):
        for r in sweep.get("redundancy", [default.redundancy]):
            if r < n:
                variants.append(Variant(f"staircase_f{f}_r{r}", "staircase",
                                        build_staircase(cfg, first_chunk_size=f, redundancy=r,
                                                        prefix_cache=False)))
    if cfg["benchmark"].get("prefix_cache_variant"):
        variants.append(Variant(f"staircase_pc_f{default.first_chunk_size}_r{default.redundancy}",
                                "staircase_prefix_cache",
                                build_staircase(cfg, prefix_cache=True)))
    return variants


def default_staircase_variant(cfg: dict[str, Any]) -> Variant:
    d = build_staircase(cfg, prefix_cache=False)
    return Variant(f"staircase_f{d.first_chunk_size}_r{d.redundancy}", "staircase", d)


def run_variant(backend: Backend, cfg: dict[str, Any], v: Variant, query: str, qid: str,
                seed: int) -> MoAResult:
    moa = build_moa(cfg, seed=seed)
    if v.staircase is None:
        return run_moa_baseline(backend, moa, query, qid)
    return run_staircase(backend, moa, v.staircase, query, qid)


# ------------------------------------------------------------------- metadata
def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return [line.strip() for line in out.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


def collect_metadata(cfg: dict[str, Any], backend: Backend) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpus": [] if cfg["backend"]["type"] == "mock" else gpu_names(),
        "backend": cfg["backend"]["type"],
        "models": model_metadata(cfg),
    }
    if isinstance(backend, OpenAICompatBackend):
        meta["served_models"] = backend.check_models()
        meta["server_versions"] = backend.server_versions()
    return meta


def config_hash(cfg: dict[str, Any]) -> str:
    relevant = {k: cfg[k] for k in ("backend", "servers", "mock_models", "moa", "staircase")
                if k in cfg}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


# -------------------------------------------------------------------- raw rows
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def latest_rows(rows: list[dict[str, Any]], chash: str) -> list[dict[str, Any]]:
    """Last row per (phase, variant, dataset, prompt) for the current config hash."""
    keep: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if r.get("config_hash") == chash:
            keep[(r["phase"], r["variant"], r["dataset_id"], r["prompt_id"])] = r
    return list(keep.values())


def execute(backend: Backend, cfg: dict[str, Any], v: Variant, rec: Any, query: str, seed: int,
            phase: str, chash: str) -> dict[str, Any]:
    row: dict[str, Any] = {"phase": phase, "variant": v.id, "method": v.method,
                           "dataset_id": rec.dataset_id, "prompt_id": rec.id, "seed": seed,
                           "config_hash": chash, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        res = run_variant(backend, cfg, v, query, rec.id, seed)
        row.update(status="ok", result=res.to_dict())
        if phase == "quality" and rec.reference is not None:
            row["prediction"] = extract_prediction(res.text)
            row["reference"] = extract_reference(rec.reference)
            row["exact_match"] = exact_match(res.text, rec.reference)
    except Exception as exc:  # noqa: BLE001 - failures are reported, not dropped
        row.update(status="error", error=repr(exc))
    return row


# ------------------------------------------------------------------- summaries
def _vals(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [r["result"][key] for r in rows if r["result"].get(key) is not None]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    s: dict[str, Any] = {"n": len(ok), "failures": len(rows) - len(ok)}
    if not ok:
        return s
    ms = 1000.0
    ttft, e2e = _vals(ok, "ttft_s"), _vals(ok, "e2e_s")
    s.update(
        ttft_median_ms=_scale(median(ttft), ms), ttft_p95_ms=_scale(percentile(ttft, 95), ms),
        e2e_median_ms=_scale(median(e2e), ms), e2e_p95_ms=_scale(percentile(e2e, 95), ms),
        tps_median=median(_vals(ok, "tps")),
        agg_prefill_median_ms=_scale(median(_vals(ok, "aggregator_prefill_s")), ms),
        agg_decode_median_ms=_scale(median(_vals(ok, "aggregator_decode_s")), ms),
        agg_calls_mean=_mean([r["result"]["aggregator_calls_n"] for r in ok]),
        agg_tokens_median=median(_vals(ok, "aggregator_tokens")),
        agg_prompt_tokens_mean=_mean(_vals(ok, "aggregator_prompt_tokens")),
        agg_cached_tokens_mean=_mean(_vals(ok, "aggregator_cached_tokens")),
        proposer_prompt_tokens_mean=_mean(_vals(ok, "proposer_prompt_tokens")),
        proposers_cancelled=sum(r["result"]["proposers_cancelled"] for r in ok),
        input_tokens_median=median([r["result"]["proposer_prompt_tokens"] / len(r["result"]["proposers"])
                                    for r in ok if r["result"].get("proposer_prompt_tokens") is not None]),
    )
    if s["agg_prompt_tokens_mean"] is not None and s["proposer_prompt_tokens_mean"] is not None:
        s["total_prompt_tokens_mean"] = s["agg_prompt_tokens_mean"] + s["proposer_prompt_tokens_mean"]
    em = [r["exact_match"] for r in ok if "exact_match" in r]
    if em:
        s["exact_match"] = sum(em) / len(em)
    return s


def _scale(x: float | None, k: float) -> float | None:
    return None if x is None else x * k


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_summaries(rows: list[dict[str, Any]], variants: list[Variant]) -> list[dict[str, Any]]:
    """One summary per (phase, variant, scope); scope 'all' pools the latency prompt sets."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["phase"], r["variant"], r["dataset_id"])].append(r)
        if r["phase"] == "latency":
            groups[(r["phase"], r["variant"], "all")].append(r)
    method_of = {v.id: v.method for v in variants}
    out = []
    for (phase, variant, scope), grp in sorted(groups.items()):
        out.append({"phase": phase, "variant": variant, "method": method_of.get(variant, variant),
                    "scope": scope, **summarize(grp)})
    for s in out:  # prompt-token overhead relative to the baseline of the same phase/scope
        base = next((b for b in out if b["phase"] == s["phase"] and b["scope"] == s["scope"]
                     and b["variant"] == "baseline"), None)
        for key, name in (("agg_prompt_tokens_mean", "agg_prompt_token_ratio_vs_baseline"),
                          ("total_prompt_tokens_mean", "total_prompt_token_ratio_vs_baseline"),
                          ("ttft_median_ms", "ttft_ratio_vs_baseline")):
            if base and s.get(key) is not None and base.get(key):
                s[name] = s[key] / base[key]
    return out


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def metrics_rows(summaries: list[dict[str, Any]], cfg: dict[str, Any], meta: dict[str, Any],
                 peak_mib: dict[int, int], mem_method: str, datasets_all: str) -> list[dict[str, str]]:
    exp = cfg["experiment"]
    models = meta["models"]
    m = cfg["moa"]

    def ident(field: str) -> str:
        if not models:
            return "mock"
        props = "|".join(str(models[p][field]) for p in m["proposers"])
        return f"proposers={props};aggregator={models[m['aggregator']][field]}"

    dtypes = sorted({str(s.get("dtype", "auto")) for s in (cfg.get("servers") or {}).values()}) or ["n/a"]
    if cfg["backend"]["type"] == "mock":
        fw = f"mock backend; python {meta['python']}"
    else:
        versions = sorted({v for v in (meta.get("server_versions") or {}).values() if v})
        fw = f"vllm {'/'.join(versions) or 'unknown'}; python {meta['python']}"
    peak_total = sum(peak_mib.values()) if peak_mib else None
    rows = []
    for s in summaries:
        if s["n"] == 0 or (s["phase"] == "latency" and s["scope"] != "all"):
            continue
        quality = s["phase"] == "quality"
        notes = [
            "latency=TTFT (request start -> first aggregator token)",
            "throughput=paper TPS (aggregator tokens / request start->last token), median",
            f"e2e_median_ms={_fmt(s.get('e2e_median_ms'))}", f"e2e_p95_ms={_fmt(s.get('e2e_p95_ms'))}",
            f"agg_prefill_median_ms={_fmt(s.get('agg_prefill_median_ms'))}",
            f"agg_decode_median_ms={_fmt(s.get('agg_decode_median_ms'))}",
            f"agg_calls_mean={_fmt(s.get('agg_calls_mean'), 2)}",
            f"agg_prompt_tokens_mean={_fmt(s.get('agg_prompt_tokens_mean'))}",
            f"agg_prompt_token_ratio_vs_baseline={_fmt(s.get('agg_prompt_token_ratio_vs_baseline'), 2)}",
            f"total_prompt_token_ratio_vs_baseline={_fmt(s.get('total_prompt_token_ratio_vs_baseline'), 2)}",
            f"agg_cached_tokens_mean={_fmt(s.get('agg_cached_tokens_mean'))}",
            f"n={s['n']}", f"failures={s['failures']}",
            f"peak_memory=whole run, sum over GPUs; {mem_method}" if peak_total else "peak_memory=not measured",
            "energy not_measured",
        ]
        rows.append({
            "experiment_id": f"{exp['name']}:{s['variant']}:{'gsm8k' if quality else 'latency'}",
            "paper_id": exp["paper_id"],
            "method": s["variant"] if s["variant"] != "baseline" else "baseline_moa",
            "commit": meta["commit"],
            "backbone_id": ident("backbone_id"),
            "backbone_revision": ident("revision"),
            "dataset_id": s["scope"] if quality else datasets_all,
            "split": "test" if quality else "eval",
            "seed": str(exp["seed"]),
            "device": "; ".join(meta["gpus"]) or ("cpu (mock)" if cfg["backend"]["type"] == "mock" else "unknown"),
            "framework_version": fw,
            "dtype": "/".join(dtypes),
            "batch_size": "1",
            "input_length": _fmt(s.get("input_tokens_median"), 0),
            "output_length": _fmt(s.get("agg_tokens_median"), 0),
            "quality_metric": "exact_match" if quality else "",
            "quality_value": _fmt(s.get("exact_match"), 4) if quality else "",
            "latency_median_ms": _fmt(s.get("ttft_median_ms")),
            "latency_p95_ms": _fmt(s.get("ttft_p95_ms")),
            "throughput": _fmt(s.get("tps_median"), 2),
            "peak_memory_mb": str(peak_total) if peak_total else "",
            "energy_j": "",
            "status": exp.get("status", "partial"),
            "notes": "; ".join(notes),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    cols = columns or sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


# ------------------------------------------------------------------------ main
def load_prompt_sets(cfg: dict[str, Any], limit: int | None) -> list[Any]:
    load_prompts = _import_loader()
    seed = cfg["experiment"]["seed"]
    prompts = []
    for ps in cfg["benchmark"]["prompt_sets"]:
        n = ps["num_samples"] if limit is None else min(limit, ps["num_samples"])
        prompts += load_prompts(ps["dataset_id"], n, seed)
    return prompts


def sanity(backend: Backend, cfg: dict[str, Any], prompts: list[Any], out: Path) -> None:
    k = cfg["benchmark"].get("sanity_prompts", 3)
    variants = [Variant("baseline", "baseline"), default_staircase_variant(cfg)]
    path = out / "sanity.jsonl"
    for i, rec in enumerate(prompts[:k]):
        print(f"\n=== prompt {rec.dataset_id}:{rec.id} ===\n{rec.prompt[:300]}")
        for v in variants:
            row = execute(backend, cfg, v, rec, rec.prompt, cfg["experiment"]["seed"] + 100 * i,
                          "sanity", "sanity")
            append_jsonl(path, row)
            if row["status"] != "ok":
                print(f"--- {v.id}: ERROR {row['error']}")
                continue
            r = row["result"]
            print(f"--- {v.id}: TTFT {r['ttft_s']:.3f}s  E2E {r['e2e_s']:.2f}s  "
                  f"agg tokens {r['aggregator_tokens']}  agg calls {r['aggregator_calls_n']}")
            for j, t in enumerate(r["proposer_texts"]):
                print(f"    proposer {j + 1}: {t[:160]!r}")
            print(f"    answer: {r['text'][:600]!r}")
    print(f"\nSanity rows appended to {path}")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Baseline MoA vs staircase streaming benchmark")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sanity", action="store_true", help="run baseline + default staircase on a few prompts and print texts")
    ap.add_argument("--limit", type=int, default=None, help="max prompts per prompt set (quick runs)")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing raw rows (no resume)")
    ap.add_argument("--output-dir", default=None, help="override experiment.output_dir")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    backend = build_backend(cfg)
    meta = collect_metadata(cfg, backend)
    prompts = load_prompt_sets(cfg, args.limit)
    seed0 = cfg["experiment"]["seed"]
    print(f"[bench] {len(prompts)} latency prompts; backend={cfg['backend']['type']}; commit={meta['commit']}")

    if args.sanity:
        sanity(backend, cfg, prompts, out)
        return {}

    variants = build_variants(cfg)
    chash = config_hash(cfg)
    raw_path = out / "raw_requests.jsonl"
    if args.fresh and raw_path.exists():
        raw_path.rename(out / f"raw_requests.{int(time.time())}.bak.jsonl")
    existing = latest_rows(read_jsonl(raw_path), chash)
    done = {(r["phase"], r["variant"], r["dataset_id"], r["prompt_id"]) for r in existing if r["status"] == "ok"}
    print(f"[bench] {len(variants)} variants: {[v.id for v in variants]}; resuming {len(done)} finished requests")

    for i, rec in enumerate(prompts[: cfg["experiment"].get("warmup_requests", 0)]):
        for v in (variants[0], default_staircase_variant(cfg)):
            row = execute(backend, cfg, v, rec, rec.prompt, seed0 + 100 * i, "warmup", chash)
            print(f"[warmup] {v.id} {rec.id}: {row['status']}")

    sampler = GpuMemorySampler(cfg["benchmark"].get("gpu_memory_sample_s", 0.5))
    if cfg["backend"]["type"] == "mock":
        sampler.available = False  # mock runs use no GPU; do not report one
    sampler.start()
    t_start = time.time()
    total = len(prompts) * len(variants)
    count = 0
    for i, rec in enumerate(prompts):
        for v in variants:
            count += 1
            if ("latency", v.id, rec.dataset_id, rec.id) in done:
                continue
            row = execute(backend, cfg, v, rec, rec.prompt, seed0 + 100 * i, "latency", chash)
            append_jsonl(raw_path, row)
            r = row.get("result") or {}
            ttft = f"{r['ttft_s'] * 1000:.0f}ms" if r.get("ttft_s") is not None else "-"
            print(f"[latency {count}/{total}] {v.id:<22} {rec.dataset_id}:{rec.id[:10]:<10} "
                  f"{row['status']} ttft={ttft} ({time.time() - t_start:.0f}s elapsed)", flush=True)

    q = cfg.get("quality") or {}
    if q.get("enabled") and not args.skip_quality:
        load_prompts = _import_loader()
        n_q = q["num_samples"] if args.limit is None else min(args.limit, q["num_samples"])
        qprompts = load_prompts(q["dataset_id"], n_q, seed0)
        qvariants = [v for v in (variants[0], default_staircase_variant(cfg))
                     if ("baseline" if v.method == "baseline" else "staircase") in q.get("methods", [])]
        for i, rec in enumerate(qprompts):
            for v in qvariants:
                if ("quality", v.id, rec.dataset_id, rec.id) in done:
                    continue
                row = execute(backend, cfg, v, rec, format_gsm8k(rec.prompt), seed0 + 100 * i,
                              "quality", chash)
                append_jsonl(raw_path, row)
                print(f"[quality {i + 1}/{len(qprompts)}] {v.id:<22} {rec.id}: {row['status']} "
                      f"em={row.get('exact_match')}", flush=True)
    peak = sampler.stop()

    rows = latest_rows(read_jsonl(raw_path), chash)
    summaries = build_summaries(rows, variants)
    datasets_all = "+".join(ps["dataset_id"] for ps in cfg["benchmark"]["prompt_sets"])
    mrows = metrics_rows(summaries, cfg, meta, peak, sampler.method, datasets_all)
    write_csv(out / "metrics.csv", mrows, METRICS_COLUMNS)
    write_csv(out / "summary_detailed.csv", summaries)
    summary = {"metadata": meta, "config": cfg, "config_hash": chash,
               "peak_memory_mib_per_gpu": peak, "memory_method": sampler.method if peak else None,
               "summaries": summaries}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    try:
        from .plots import make_plots  # noqa: PLC0415

        made = make_plots(summaries, rows, out)
        print(f"[bench] plots: {made}")
    except ImportError as exc:
        print(f"[bench] plots skipped ({exc})")
    print_table(summaries)
    print(f"[bench] wrote {out / 'metrics.csv'}, {out / 'summary_detailed.csv'}, {out / 'summary.json'}")
    return summary


def print_table(summaries: list[dict[str, Any]]) -> None:
    print(f"\n{'phase':<8} {'variant':<22} {'n':>3} {'TTFT med':>9} {'TTFT p95':>9} {'E2E med':>9} "
          f"{'TPS':>7} {'aggP/base':>9} {'EM':>6}")
    for s in summaries:
        if s["phase"] == "latency" and s["scope"] != "all":
            continue
        print(f"{s['phase']:<8} {s['variant']:<22} {s['n']:>3} {_fmt(s.get('ttft_median_ms'), 0):>9} "
              f"{_fmt(s.get('ttft_p95_ms'), 0):>9} {_fmt(s.get('e2e_median_ms'), 0):>9} "
              f"{_fmt(s.get('tps_median'), 1):>7} {_fmt(s.get('agg_prompt_token_ratio_vs_baseline'), 2):>9} "
              f"{_fmt(s.get('exact_match'), 3):>6}")


if __name__ == "__main__":
    main()
