"""Stage 2 (GPU): run two-model cascades *live* at batch size 1 to measure real
end-to-end latency and peak GPU memory.

The offline study (collect_generations + run_sweep) measures quality and a
compute-cost proxy. Latency and memory, however, must be measured on a real
cascade, so this script:

1. loads the stored generations, takes split 0 (seed 42) of the paper's
   stratified calibration/test protocol, and per dataset picks
   L = cheapest model (FLOP cost) and H = most accurate model on calibration;
2. on calibration, searches the exhaustive L->H threshold frontier with the
   configured score and selects the policy for each budget fraction
   (B = c_L + f (c_H - c_L));
3. loads L and H together on the GPU and, for the first ``n`` test queries,
   runs every policy end to end: generate with L, compute the score from that
   generation, escalate to H iff score < tau (via CascadeStoppingPolicy).
   Baselines always-L and always-H run through the same code path.

Outputs ``<results>/live/live_summary.csv`` and ``live_per_query.jsonl.gz``.
Both models stay resident, so peak memory is that of an L+H deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment_metadata, generation_path, load_config, read_jsonl, write_jsonl  # noqa: E402

from Adaptive_Stopping.methods.cascade_escalation.data import CascadeTable, cheapest_model, stratified_split  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.policy import CascadePolicy, CascadeStoppingPolicy  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.scores import token_scores_from_topk  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.search import enumerate_sequence, pareto_frontier  # noqa: E402
from Dataloaders.evaluation import cascade_benchmarks as bench  # noqa: E402


def load_table(cfg: dict, results: Path, dataset: str, tag: str = "") -> CascadeTable:
    rows, params = {}, {}
    for m in cfg["models"]:
        p = generation_path(results, dataset, m["slug"] + tag)
        if p.exists():
            rows[m["slug"]] = read_jsonl(p)
            params[m["slug"]] = m["num_parameters"]
    if len(rows) < 2:
        raise FileNotFoundError(f"need generations of >= 2 models for {dataset} under {results}")
    return CascadeTable.from_rows(dataset, rows, params)


def choose_policies(table: CascadeTable, score: str, fractions, seed: int, cal_fraction: float):
    C = table.C["flops"]
    strata = table.U[:, cheapest_model(C)]
    cal, test = stratified_split(strata, seed, cal_fraction)
    accs = table.U[cal].mean(axis=0)
    costs = C[cal].mean(axis=0)
    L = int(np.argmin(costs))
    best = np.flatnonzero(accs == accs.max())
    H = int(best[np.argmin(costs[best])])
    if H == L:
        H = int(np.argsort(-accs)[1])
    score_fn = lambda m, nxt=None: table.S[score][:, m]  # noqa: E731
    cands = [enumerate_sequence((L,), table.U, C, score_fn, cal), enumerate_sequence((H,), table.U, C, score_fn, cal),
             enumerate_sequence((L, H), table.U, C, score_fn, cal)]
    fr = pareto_frontier(cands)
    out = []
    for f in fractions:
        b = costs[L] + f * (costs[H] - costs[L])
        k = fr.select(b)
        pol = fr.policies[k]
        out.append({"name": f"cascade@{f:.2f}", "budget_fraction": f, "policy": pol,
                    "cal_accuracy": float(fr.acc[k]), "cal_cost_tflops": float(fr.cost[k])})
    return L, H, cal, test, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=None, help="override queries per dataset")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import torch
    from generation import load_hf_generator

    cfg = load_config(args.config)
    live = cfg["live"]
    sweep = cfg["sweep"]
    results = Path(args.results)
    out_dir = results / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    slug2id = {m["slug"] + args.tag: m["id"] for m in cfg["models"]}
    slug2params = {m["slug"] + args.tag: m["num_parameters"] for m in cfg["models"]}
    summary, per_query = [], []
    loaded: dict = {}
    cuda = torch.cuda.is_available() and (args.device or "cuda").startswith("cuda")

    def ensure(slugs):
        """Keep exactly ``slugs`` resident on the device."""
        import gc

        for s in list(loaded):
            if s not in slugs:
                del loaded[s]
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        for slug in slugs:
            if slug in loaded:
                continue
            before = torch.cuda.memory_allocated() if cuda else 0
            g = load_hf_generator(slug2id[slug], dtype=cfg.get("dtype", "bfloat16"),
                                  top_k=cfg.get("top_k_logprobs", 15), device=args.device)
            g.weights_mb = ((torch.cuda.memory_allocated() - before) / 2**20) if cuda else None
            loaded[slug] = g
        return [loaded[s] for s in slugs]

    for ds, n_default in live["datasets"].items():
        n = args.n if args.n is not None else n_default
        table = load_table(cfg, results, ds, args.tag)
        L, H, cal, test, pols = choose_policies(table, live.get("score", "mean_negentropy"), live["budget_fractions"],
                                                sweep.get("split_seed0", 42), sweep.get("cal_fraction", 0.5))
        sL, sH = table.models[L], table.models[H]
        records = {r["uid"]: r for r in bench.load_records(ds, num_examples=cfg["datasets"][ds]["num_examples"],
                                                            seed=cfg.get("seed", 2026))}
        uids = [table.uids[i] for i in test[:n]]
        mnt = int(cfg["datasets"][ds]["max_new_tokens"])
        policies = [{"name": f"always:{sL}", "policy": CascadePolicy((0,), ())},
                    {"name": f"always:{sH}", "policy": CascadePolicy((1,), ())}]
        for p in pols:
            policies.append({**p, "policy": CascadePolicy((0, 1), p["policy"].thresholds)
                             if p["policy"].depth == 2 else CascadePolicy((0 if p["policy"].sequence[0] == L else 1,), ())})
        # Checkpoint per (dataset, policy): a finished measurement is reused on re-run as
        # long as the same models, thresholds, queries and settings would be measured.
        todo = []
        for pinfo in policies:
            pol = pinfo["policy"]
            pinfo["fingerprint"] = {"dataset": ds, "policy": pinfo["name"], "small": sL, "large": sH,
                                    "sequence": list(pol.sequence), "thresholds": [float(t) for t in pol.thresholds],
                                    "uids": uids, "max_new_tokens": mnt, "score": live.get("score", "mean_negentropy")}
            ck = ckpt_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ds}__{pinfo['name']}") + ".json")
            pinfo["ckpt"] = ck
            if ck.exists():
                try:
                    saved = json.loads(ck.read_text())
                except json.JSONDecodeError:
                    saved = {}
                if saved.get("fingerprint") == pinfo["fingerprint"]:
                    summary.append(saved["summary"])
                    per_query.extend(saved["per_query"])
                    print(f"[live] {ds} {pinfo['name']}: from checkpoint", flush=True)
                    continue
            todo.append(pinfo)
        if not todo:
            continue
        gL, gH = ensure([sL, sH])
        gens = [gL, gH]
        params = [slug2params[sL], slug2params[sH]]
        for g in gens:  # warm-up
            for u in uids[: live.get("warmup", 3)]:
                g.generate_one_timed(records[u]["prompt"], mnt)
        for pinfo in todo:
            pol: CascadePolicy = pinfo["policy"]
            pq = []
            sp = CascadeStoppingPolicy(pol)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            lat, corr, esc, flops = [], [], [], []
            for u in uids:
                rec = records[u]
                sp.reset({"uid": u})
                t0 = time.perf_counter()
                stage, f = 0, 0.0
                while True:
                    m = pol.sequence[stage]
                    r = gens[m].generate_one_timed(rec["prompt"], mnt)
                    f += 2.0 * params[m] * (r["n_in"] + r["n_out"]) / 1e12
                    s = token_scores_from_topk(r["chosen_logprobs"], r["topk_logprobs"],
                                               k=cfg.get("top_k_logprobs", 15))[live.get("score", "mean_negentropy")]
                    if sp.should_stop(stage, {"score": s}):
                        break
                    stage += 1
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                ok = bench.grade(r["response"], rec)
                lat.append(dt)
                corr.append(ok)
                esc.append(stage > 0)
                flops.append(f)
                pq.append({"dataset": ds, "policy": pinfo["name"], "uid": u, "latency_s": dt,
                           "correct": ok, "stages": stage + 1, "tflops": f})
            lat = np.asarray(lat)
            row = {
                "dataset": ds, "policy": pinfo["name"], "small": sL, "large": sH,
                "thresholds": json.dumps([float(t) for t in pol.thresholds]),
                "budget_fraction": pinfo.get("budget_fraction", ""),
                "cal_accuracy": pinfo.get("cal_accuracy", ""), "n": len(uids),
                "accuracy": float(np.mean(corr)), "escalation_rate": float(np.mean(esc)),
                "latency_mean_ms": 1e3 * float(lat.mean()), "latency_median_ms": 1e3 * float(np.median(lat)),
                "latency_p95_ms": 1e3 * float(np.percentile(lat, 95)),
                "throughput_qps": float(len(lat) / lat.sum()),
                "mean_tflops": float(np.mean(flops)),
                "peak_memory_allocated_mb": (torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
                "weights_mb_small": gL.weights_mb, "weights_mb_large": gH.weights_mb,
            }
            summary.append(row)
            per_query.extend(pq)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tmp = pinfo["ckpt"].with_suffix(".tmp")
            tmp.write_text(json.dumps({"fingerprint": pinfo["fingerprint"], "summary": row, "per_query": pq}))
            tmp.replace(pinfo["ckpt"])
            print(f"[live] {ds} {pinfo['name']}: acc={row['accuracy']:.3f} esc={row['escalation_rate']:.2f} "
                  f"lat_med={row['latency_median_ms']:.0f}ms peak={row['peak_memory_allocated_mb']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)  # recreate if removed during a long run
    with open(out_dir / "live_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    write_jsonl(out_dir / "live_per_query.jsonl.gz", per_query)
    (out_dir / "live_meta.json").write_text(json.dumps({"environment": environment_metadata(),
                                                        "batch_size": 1, "timing": "wall clock per query incl. "
                                                        "tokenisation, all stages, cuda synchronised; models resident",
                                                        "config": live}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
