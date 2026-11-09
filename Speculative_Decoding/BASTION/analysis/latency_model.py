#!/usr/bin/env python
"""E5 (paper Fig. 9a / App. H.4): accuracy of the verification-latency model.

Inputs are the files upstream writes to ``upstream/cache`` (format from
``bastion/cost_model.py::collect_calibration_data`` and
``bastion/benchmark.py``):

* ``calib_probe_<gpu>_<model>.sigma_sequence_v1.jsonl`` - one JSON object per
  (s, c) grid point: ``model_name_or_path, gpu_type, attn_impl, s, c,
  verification_latency_s`` (mean of 5 timed repeats), ``..._std_s``, ...
  Grid: s = 16..512 step 16, c in {64,128,...,2048}; dummy tokens, causal
  ``sdpa`` forward with ``output_hidden_states=True``.
* ``calibration_<gpu>_<model>.json`` - ``{"fit_parameters": {alpha_m, beta_m,
  alpha_c, beta_c}, "fit_version", "sigma_scale", "attn_impl"}``.

Models compared at every probe point:
  uncalibrated roofline  max(t_mem, t_comp) from upstream ``compute_cost_model_latencies``
  calibrated fit         max(alpha_m t_mem + beta_m, alpha_c t_comp + beta_c)
                          (cross-checked against upstream's
                          ``build_verified_latency_estimator(...).estimate(s)``,
                          which the controller actually calls)
RMSE in ms is reported over all probe points and over the paper's plotted
contexts c in {64, 256, 1024}. The calibrated RMSE is in-sample (the fit used
the same points); a leave-one-context-out refit with upstream's
``fit_roofline_calibration`` gives an out-of-sample number.

Optional in-situ check (``--runs``): BASTION records store per-step realized
tree size, context length and measured verification latency (tree mask,
``sdpa``); these are compared with the calibrated prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_records  # noqa: E402

PROBE_RE = re.compile(r"^calib_probe_(?P<gpu>[A-Za-z0-9]+)_(?P<model>.+)\.(?P<ver>sigma_sequence_v1)\.jsonl$")
PAPER_CONTEXTS = (64, 256, 1024)


def _cost_model():
    import repro_lib

    repro_lib.register_local_gpu_specs()
    from bastion import cost_model

    return cost_model


def find_cache_sets(cache_dir: str | Path) -> list[dict]:
    cache_dir = Path(cache_dir)
    out = []
    if not cache_dir.exists():
        return out
    for p in sorted(cache_dir.iterdir()):
        m = PROBE_RE.match(p.name)
        if not m:
            continue
        gpu, model = m.group("gpu"), m.group("model")
        calib = cache_dir / f"calibration_{gpu}_{model}.json"
        profile = cache_dir / f"calibration_{gpu}_{model}.profile.json"
        out.append({"gpu_key": gpu, "model_key": model, "probe": p,
                    "calibration": calib if calib.exists() else None,
                    "profile": profile if profile.exists() else None})
    return out


def rmse_ms(pred: np.ndarray, meas: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(meas)) ** 2)) * 1e3)


def evaluate_probe(probe_records: list[dict], params: dict | None, *, calibration_path: str | Path | None = None,
                   sigma_scale: float = 30.0) -> dict:
    cm = _cost_model()
    agg = cm.aggregate_latency_records(probe_records)
    first = probe_records[0]
    model_name = first.get("model_name_or_path") or first.get("model_name")
    gpu_type = first.get("gpu_type")
    model_key, _ = cm.resolve_model_spec(model_name)
    gpu_key, gpu_spec = cm.resolve_gpu_spec(gpu_type)

    t_mem, t_comp, y, sc = cm.prepare_arrays_for_fitting(agg, model_name=model_name, gpu_type=gpu_type)
    uncal = np.maximum(t_mem, t_comp)
    out: dict = {
        "model_key": model_key, "gpu_key": gpu_key, "gpu_spec": gpu_spec,
        "attn_impl": first.get("attn_impl"), "n_points": len(y),
        "contexts": sorted({c for _, c in sc}), "sequence_sizes": [min(s for s, _ in sc), max(s for s, _ in sc)],
    }
    points = [{"s": s, "c": c, "measured_s": float(m), "t_mem_s": float(tm), "t_comp_s": float(tc), "uncal_s": float(u)}
              for (s, c), m, tm, tc, u in zip(sc, y, t_mem, t_comp, uncal)]

    refit = cm.fit_roofline_calibration(t_mem, t_comp, y, s_values=np.array([s for s, _ in sc], float), sigma_scale=sigma_scale)
    out["refit_params"] = dict(zip(["alpha_m", "beta_m", "alpha_c", "beta_c"], refit[:4]))
    if params is None:
        params = out["refit_params"]
        out["params_source"] = "refit from probe (no calibration cache)"
    else:
        out["params_source"] = "calibration cache"
    out["params"] = params
    cal = np.maximum(params["alpha_m"] * t_mem + params["beta_m"], params["alpha_c"] * t_comp + params["beta_c"])
    for p, v in zip(points, cal):
        p["cal_s"] = float(v)

    # Cross-check with the estimator object used by the controller.
    if calibration_path is not None:
        old = os.environ.get("CALIBRATION_JSON_PATH")
        os.environ["CALIBRATION_JSON_PATH"] = str(calibration_path)
        try:
            diffs = []
            cache: dict[int, object] = {}
            for p in points:
                est = cache.get(p["c"])
                if est is None:
                    est = cache[p["c"]] = cm.build_verified_latency_estimator(
                        context_tokens=p["c"], model_name=model_key, gpu_type=gpu_key)
                diffs.append(abs(est.estimate(p["s"]) - p["cal_s"]))
            out["controller_estimator_max_abs_diff_s"] = float(max(diffs))
        finally:
            if old is None:
                os.environ.pop("CALIBRATION_JSON_PATH", None)
            else:
                os.environ["CALIBRATION_JSON_PATH"] = old

    def _rm(sel):
        idx = [i for i, p in enumerate(points) if sel(p)]
        if not idx:
            return None
        m = y[idx]
        r_u, r_c = rmse_ms(uncal[idx], m), rmse_ms(cal[idx], m)
        return {"n": len(idx), "rmse_uncal_ms": r_u, "rmse_cal_ms": r_c,
                "reduction_pct": 100 * (1 - r_c / r_u) if r_u > 0 else None,
                "mean_bias_uncal_ms": float(np.mean(uncal[idx] - m) * 1e3),
                "mean_bias_cal_ms": float(np.mean(cal[idx] - m) * 1e3)}

    out["rmse_all"] = _rm(lambda p: True)
    out["rmse_paper_contexts"] = _rm(lambda p: p["c"] in PAPER_CONTEXTS)
    out["rmse_by_context"] = {str(c): _rm(lambda p, c=c: p["c"] == c) for c in out["contexts"]}

    # Leave-one-context-out refit (out-of-sample calibrated RMSE).
    loco = {}
    s_arr = np.array([s for s, _ in sc], float)
    c_arr = np.array([c for _, c in sc])
    for c in out["contexts"]:
        tr, te = c_arr != c, c_arr == c
        if tr.sum() < 4:
            continue
        a_m, b_m, a_c, b_c, _ = cm.fit_roofline_calibration(t_mem[tr], t_comp[tr], y[tr], s_values=s_arr[tr], sigma_scale=sigma_scale)
        pred = np.maximum(a_m * t_mem[te] + b_m, a_c * t_comp[te] + b_c)
        loco[str(c)] = rmse_ms(pred, y[te])
    out["loco_rmse_cal_ms"] = loco
    out["points"] = points
    return out


def in_situ(records: list[dict], params: dict, model_key: str, gpu_key: str) -> dict:
    """Calibrated prediction vs per-step measured verification latency inside bastion_generate."""
    cm = _cost_model()
    _, mspec = cm.resolve_model_spec(model_key)
    rows = []
    for r in records:
        if r.get("mode") != "bastion" or r.get("model_key") != model_key or r.get("gpu_key") != gpu_key:
            continue
        sizes, ctxs, lats = r.get("tree_sizes") or [], r.get("tree_contexts") or [], r.get("verification_latencies") or []
        if not (len(sizes) == len(ctxs) == len(lats)):
            continue
        for s, c, m in zip(sizes, ctxs, lats):
            rows.append((s, c, m))
    if not rows:
        return {"n_steps": 0}
    s = np.array([x[0] for x in rows]); c = np.array([x[1] for x in rows]); m = np.array([x[2] for x in rows])
    cache = {}
    tm, tc = np.empty(len(rows)), np.empty(len(rows))
    for i, (si, ci) in enumerate(zip(s, c)):
        key = (int(si), int(ci))
        if key not in cache:
            cache[key] = cm.compute_cost_model_latencies(model_name=model_key, gpu_type=gpu_key, s=int(si), c=int(ci))
        tm[i], tc[i] = cache[key]
    uncal = np.maximum(tm, tc)
    cal = np.maximum(params["alpha_m"] * tm + params["beta_m"], params["alpha_c"] * tc + params["beta_c"])
    bins = defaultdict(list)
    for i, si in enumerate(s):
        b = int(2 ** np.floor(np.log2(max(si, 1))))
        bins[b].append(i)
    by_bin = {str(b): {"n": len(ix), "measured_ms": float(m[ix].mean() * 1e3), "cal_ms": float(cal[ix].mean() * 1e3),
                       "uncal_ms": float(uncal[ix].mean() * 1e3)} for b, ix in sorted(bins.items())}
    return {"n_steps": len(rows), "rmse_uncal_ms": rmse_ms(uncal, m), "rmse_cal_ms": rmse_ms(cal, m),
            "mean_bias_cal_ms": float(np.mean(cal - m) * 1e3), "by_tree_size_bin": by_bin,
            "note": "measured = verification forward with tree mask inside bastion_generate (includes first-step / "
                    "context effects); calibration probe used causal dummy forwards"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "upstream" / "cache"))
    ap.add_argument("--runs", nargs="*", default=[], help="harness JSONL files for the in-situ comparison")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = find_cache_sets(args.cache_dir)
    if not sets:
        print(f"no calibration probe files in {args.cache_dir}; nothing to evaluate")
        (out / "latency_model.json").write_text(json.dumps({"status": "no probe files", "cache_dir": args.cache_dir}), encoding="utf-8")
        return
    records = load_records(args.runs) if args.runs else []
    results = []
    for st in sets:
        probe = [json.loads(line) for line in st["probe"].read_text(encoding="utf-8").splitlines() if line.strip()]
        if not probe:
            print(f"[skip] empty probe file {st['probe']}")
            continue
        calib = json.loads(st["calibration"].read_text()) if st["calibration"] else None
        params = calib.get("fit_parameters") if calib else None
        sigma = float(calib.get("sigma_scale", 30.0)) if calib else 30.0
        res = evaluate_probe(probe, params, calibration_path=st["calibration"], sigma_scale=sigma)
        res["probe_file"] = str(st["probe"])
        res["calibration_file"] = str(st["calibration"]) if st["calibration"] else None
        res["profile"] = json.loads(st["profile"].read_text()) if st["profile"] else None
        if records:
            res["in_situ"] = in_situ(records, res["params"], res["model_key"], res["gpu_key"])
        results.append(res)
        with open(out / f"latency_points_{st['gpu_key']}_{st['model_key']}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res["points"][0].keys()))
            w.writeheader()
            w.writerows(res["points"])
    slim = [{k: v for k, v in r.items() if k != "points"} for r in results]
    (out / "latency_model.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    lines = ["| GPU | Model | points | RMSE uncal (ms) | RMSE cal (ms) | reduction | RMSE uncal c∈{64,256,1024} | RMSE cal c∈{64,256,1024} | LOCO cal RMSE (mean over held-out c) | in-situ RMSE cal (ms) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in slim:
        a, p = r["rmse_all"], r["rmse_paper_contexts"] or {}
        loco = np.mean(list(r["loco_rmse_cal_ms"].values())) if r["loco_rmse_cal_ms"] else None
        ins = (r.get("in_situ") or {}).get("rmse_cal_ms")
        f = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"| {r['gpu_key']} | {r['model_key']} | {r['n_points']} | {f(a['rmse_uncal_ms'])} | {f(a['rmse_cal_ms'])} | "
                     f"{f(a['reduction_pct'])}% | {f(p.get('rmse_uncal_ms'))} | {f(p.get('rmse_cal_ms'))} | {f(loco)} | {f(ins)} |")
    md = ("# Verification-latency model accuracy (E5)\n\n" + "\n".join(lines) +
          "\n\nPaper (A100, App. H.4): 29.2→2.3 ms (Qwen3-4B), 30.6→3.1 ms (Qwen3-8B), 26.4→3.5 ms (Llama-3.1-8B). "
          "Calibrated RMSE here is in-sample unless stated (LOCO column).\n")
    (out / "latency_model.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
