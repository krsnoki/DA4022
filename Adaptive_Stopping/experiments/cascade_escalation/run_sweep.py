"""Stage 3 (CPU): the paper's offline cascade study on the stored generations.

For each dataset, cost definition and deferral score, over ``n_splits``
stratified calibration/test splits (seeds 42, 43, ...):

* build calibration frontiers for the nested policy classes S_1 ... S_D
  (S_1 = best single model, S_2 = two-model cascades, ...);
* on a grid of ``n_budgets`` budgets between the cheapest model's cost and the
  most accurate model's cost, freeze the calibration-selected policy and
  evaluate it on the test split;
* evaluate the unmodified baselines on the same test split: always-small
  (cheapest), always-large (most accurate on calibration) and the random router
  between them, at the same budgets.

Outputs (``<results>/sweep/``):

* ``curves.npz``            - test accuracy / cost / escalation per budget, split, class.
* ``summary.csv``           - area-averaged test accuracy per (dataset, cost, score, class).
* ``depth_gains.csv``       - paper Table 1 analogue: S2-S1 and S_d-S2 in percentage points.
* ``theory.csv``            - Prop. 3.2 check (gain over random, both sides) and FOC diagnostics.
* ``models.csv``            - single-model accuracy / cost (paper Table 4 analogue).
* ``plots/*.png``           - accuracy vs budget, depth gains, escalation rate, live latency.
* ``Performance_Metrics/results/cascade_escalation.csv`` - shared-schema rows.

Example::

    python Adaptive_Stopping/experiments/cascade_escalation/run_sweep.py \
        --config Adaptive_Stopping/experiments/cascade_escalation/configs/qwen25_phi35_pool.yaml \
        --results results/cascade --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, git_commit, load_config  # noqa: E402
from live_cascade import load_table  # noqa: E402

from Adaptive_Stopping.methods.cascade_escalation import theory  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.data import CascadeTable, stratified_split  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.learned_scores import ScoreProvider  # noqa: E402
from Adaptive_Stopping.methods.cascade_escalation.search import (  # noqa: E402
    budget_grid,
    build_frontiers,
    enumerate_sequence,
    pareto_frontier,
    select_and_evaluate,
)

COST_UNITS = {"flops": "TFLOPs/query (2*N_params*(n_in+n_out))", "gpu_time": "GPU-s/query (apportioned)"}


def trapz_mean(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Trapezoid integral over a normalised budget grid on [0, 1]."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    if n == 1:
        return np.take(y, 0, axis=axis)
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    return np.tensordot(y, w, axes=([axis], [0]))


def run_split(job: Dict) -> Dict:
    """One (dataset, cost, split): all scores, all depths, plus baselines."""
    table: CascadeTable = job["table"]
    cost_name, r, sw = job["cost"], job["split"], job["sweep"]
    U = table.U
    C = table.C[cost_name]
    seed = int(sw.get("split_seed0", 42)) + r
    cheapest_overall = int(np.argmin(C.mean(axis=0)))
    cal, test = stratified_split(U[:, cheapest_overall], seed, sw.get("cal_fraction", 0.5))
    cal_acc, cal_cost = U[cal].mean(0), C[cal].mean(0)
    budgets = budget_grid(cal_acc, cal_cost, int(sw.get("n_budgets", 500)))
    L = int(np.argmin(cal_cost))
    best = np.flatnonzero(cal_acc == cal_acc.max())
    H = int(best[np.argmin(cal_cost[best])])
    D = int(sw.get("max_depth", 5))
    D = min(D, U.shape[1])
    grid_sizes = {int(k): v for k, v in (sw.get("grid_sizes") or {}).items()}
    out = {"split": r, "seed": seed, "budgets": budgets, "L": L, "H": H,
           "n_cal": len(cal), "n_test": len(test), "curves": {}, "policies": {}}
    # baselines (no confidence signal)
    tL, tH = U[test, L].mean(), U[test, H].mean()
    t = np.clip((budgets - cal_cost[L]) / max(cal_cost[H], 1e-300), 0, 1) if H != L else np.zeros_like(budgets)
    out["baselines"] = {
        "always_small": {"accuracy": tL, "mean_cost": C[test, L].mean()},
        "always_large": {"accuracy": tH, "mean_cost": C[test, H].mean()},
        "random_router": {"accuracy": (1 - t) * tL + t * tH, "mean_cost": C[test, L].mean() + t * C[test, H].mean(),
                          "escalation_rate": t},
    }
    for score in job["scores"]:
        sp = ScoreProvider(table.S, score, U=U, cal_idx=cal, seed=seed)
        frs = build_frontiers(U, C, sp, cal, D, grid_sizes)
        for d, fr in frs.items():
            rows = select_and_evaluate(fr, budgets, U, C, sp, test)
            acc = np.array([x.get("accuracy", np.nan) for x in rows])
            cst = np.array([x.get("mean_cost", np.nan) for x in rows])
            esc = np.array([x.get("escalation_rate", np.nan) for x in rows])
            calls = np.array([x.get("mean_models_called", np.nan) for x in rows])
            eff = np.array([x["policy"].effective_depth if x.get("feasible") else np.nan for x in rows])
            out["curves"][(score, d)] = {"accuracy": acc, "mean_cost": cst, "escalation_rate": esc,
                                         "mean_models_called": calls, "effective_depth": eff}
            if r == 0:
                mid = rows[len(rows) // 2]
                out["policies"][(score, d)] = mid["policy"].to_dict(table.models) if mid.get("feasible") else None
        # theory diagnostics for the (L, H) pair on calibration, split 0 only (cheap)
        if r == 0 and H != L:
            s = sp(L, H)[cal]
            gain = theory.escalation_gain_over_random(s, U[cal, L], U[cal, H], np.ones(len(cal), bool))
            rhs = theory.prop32_rhs(s, U[cal, L], U[cal, H], np.ones(len(cal), bool))
            cs = enumerate_sequence((L, H), U, C, sp, cal)
            fr = pareto_frontier([cs])
            taus = np.array([p.thresholds[0] if p.depth == 2 else np.nan for p in fr.policies])
            slope = theory.frontier_slopes(fr.cost, fr.acc)
            est = theory.local_benefit_cost(sp(L, H)[cal], U[cal, L], U[cal, H], C[cal, H], taus)
            ok = np.isfinite(slope) & np.isfinite(est["ratio"])
            corr = float(np.corrcoef(slope[ok], est["ratio"][ok])[0, 1]) if ok.sum() > 3 else float("nan")
            out.setdefault("theory", {})[score] = {
                "small": table.models[L], "large": table.models[H],
                "gain_over_random_direct_pp": 100 * gain, "gain_over_random_prop32_pp": 100 * rhs["rhs_form1"],
                "auroc_small_correct": rhs["auroc_own"], "auroc_large_correct": rhs["auroc_downstream"],
                "foc_slope_vs_ratio_corr": corr,
                "stagewise_dominance_frac": theory.stagewise_dominance(sp(L, H)[cal], U[cal, L], U[cal, H]),
                "frontier_concavity_violations": theory.concavity_violations(fr.cost, fr.acc),
            }
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plots(curves: Dict, models_rows, out_dir: Path, primary_score: str, primary_cost: str, live_csv: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[sweep] matplotlib unavailable: {e}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#1b6ca8", "#e07b39", "#3a9d5d", "#8e5ea2", "#c43c39"]
    for ds, by_cost in curves.items():
        if primary_cost not in by_cost:
            continue
        c = by_cost[primary_cost]
        x = np.linspace(0, 1, c["budgets"].shape[1])
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for d in sorted(k[1] for k in c["acc"] if k[0] == primary_score):
            y = np.nanmean(c["acc"][(primary_score, d)], axis=0)
            style = {1: ("-", 2.0), 2: ("-", 2.4)}.get(d, ((0, (4, 2 + d)), 1.2))
            ax.plot(x, 100 * y, color=palette[(d - 1) % 5], ls=style[0], lw=style[1],
                    label={1: "S1 best single model", 2: "S2 two-model cascade"}.get(d, f"S{d} up to {d} models"))
        ax.plot(x, 100 * np.nanmean(c["random"], axis=0), color="#777", ls="--", lw=1.2, label="random router")
        ax.set_xlabel("normalised budget (cheapest model -> most accurate model)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{ds}: cascade depth vs accuracy ({primary_score}, cost={primary_cost})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"accuracy_vs_budget_{ds}_{primary_cost}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
        if (primary_score, 2) in c["esc"]:
            ax.plot(x, 100 * np.nanmean(c["esc"][(primary_score, 2)], axis=0), color=palette[1], label="S2 cascade")
        ax.plot(x, 100 * np.nanmean(c["random_esc"], axis=0), color="#777", ls="--", label="random router")
        ax.set_xlabel("normalised budget")
        ax.set_ylabel("escalation rate (%)")
        ax.set_title(f"{ds}: share of queries escalated past the first model", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"escalation_rate_{ds}_{primary_cost}.png")
        plt.close(fig)

    if live_csv.exists():
        import pandas as pd

        live = pd.read_csv(live_csv)
        for ds, g in live.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=150)
            for _, row in g.iterrows():
                is_c = str(row["policy"]).startswith("cascade")
                ax.scatter(row["latency_median_ms"], 100 * row["accuracy"], s=40,
                           color=palette[1] if is_c else palette[0], zorder=3)
                ax.annotate(str(row["policy"]).replace("always:", ""), (row["latency_median_ms"], 100 * row["accuracy"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax.set_xlabel("median end-to-end latency per query, batch size 1 (ms)")
            ax.set_ylabel("accuracy (%)")
            ax.set_title(f"{ds}: live cascade vs single models (measured)", fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"live_latency_{ds}.png")
            plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--scores", nargs="*", default=None)
    ap.add_argument("--costs", nargs="*", default=None)
    ap.add_argument("--n-splits", type=int, default=None)
    ap.add_argument("--n-budgets", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--metrics-csv", default=str(REPO_ROOT / "Performance_Metrics" / "results" / "cascade_escalation.csv"))
    ap.add_argument("--status", default="partial", help="reproduction status label for metrics rows")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sw = dict(cfg["sweep"])
    for k, v in (("n_splits", args.n_splits), ("n_budgets", args.n_budgets), ("max_depth", args.max_depth)):
        if v is not None:
            sw[k] = v
    scores = args.scores or sw["scores"]
    costs = args.costs or sw["costs"]
    datasets = args.datasets or list(cfg["datasets"])
    results = Path(args.results)
    out = results / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    tag = ""

    curves: Dict = {}
    summary, gains, theory_rows, model_rows, metric_rows = [], [], [], [], []
    npz = {}
    t_start = time.time()
    for ds in datasets:
        try:
            table = load_table(cfg, results, ds, tag)
        except FileNotFoundError as e:
            print(f"[sweep] skip {ds}: {e}")
            continue
        for j, m in enumerate(table.models):
            model_rows.append({"dataset": ds, "model": m, "n": table.n, "accuracy": table.U[:, j].mean(),
                               **{f"mean_cost_{c}": table.C[c][:, j].mean() for c in table.C},
                               "mean_n_in": table.extra["n_in"][:, j].mean(),
                               "mean_n_out": table.extra["n_out"][:, j].mean()})
        for cost in costs:
            if cost == "gpu_time" and not np.all(np.isfinite(table.C[cost])):
                print(f"[sweep] {ds}: gpu_time missing, skipping that cost")
                continue
            jobs = [{"table": table, "cost": cost, "split": r, "sweep": sw, "scores": scores}
                    for r in range(int(sw["n_splits"]))]
            t0 = time.time()
            if args.workers > 1:
                with ProcessPoolExecutor(args.workers) as ex:
                    res = list(ex.map(run_split, jobs))
            else:
                res = [run_split(j) for j in jobs]
            print(f"[sweep] {ds} cost={cost}: {len(res)} splits in {time.time() - t0:.0f}s", flush=True)
            keys = list(res[0]["curves"])
            c = {"budgets": np.stack([x["budgets"] for x in res]),
                 "acc": {k: np.stack([x["curves"][k]["accuracy"] for x in res]) for k in keys},
                 "esc": {k: np.stack([x["curves"][k]["escalation_rate"] for x in res]) for k in keys},
                 "cost": {k: np.stack([x["curves"][k]["mean_cost"] for x in res]) for k in keys},
                 "calls": {k: np.stack([x["curves"][k]["mean_models_called"] for x in res]) for k in keys},
                 "depth": {k: np.stack([x["curves"][k]["effective_depth"] for x in res]) for k in keys},
                 "random": np.stack([x["baselines"]["random_router"]["accuracy"] for x in res]),
                 "random_esc": np.stack([x["baselines"]["random_router"]["escalation_rate"] for x in res])}
            curves.setdefault(ds, {})[cost] = c
            npz[f"{ds}/{cost}/budgets"] = c["budgets"]
            npz[f"{ds}/{cost}/random_router"] = c["random"]
            for k in keys:
                npz[f"{ds}/{cost}/{k[0]}/S{k[1]}/accuracy"] = c["acc"][k]
                npz[f"{ds}/{cost}/{k[0]}/S{k[1]}/mean_cost"] = c["cost"][k]
                npz[f"{ds}/{cost}/{k[0]}/S{k[1]}/escalation_rate"] = c["esc"][k]
            small = table.models[res[0]["L"]]
            large = table.models[res[0]["H"]]
            area_rand = trapz_mean(c["random"])
            base = {"dataset": ds, "cost": cost, "n_splits": len(res), "n_budgets": c["budgets"].shape[1],
                    "small_model_split0": small, "large_model_split0": large}
            for name, v in (("always_small", [x["baselines"]["always_small"] for x in res]),
                            ("always_large", [x["baselines"]["always_large"] for x in res])):
                a = np.array([y["accuracy"] for y in v])
                summary.append({**base, "score": "-", "policy_class": name, "test_acc_area_mean": a.mean(),
                                "test_acc_area_std": a.std(), "mean_cost": np.mean([y["mean_cost"] for y in v])})
            summary.append({**base, "score": "-", "policy_class": "random_router",
                            "test_acc_area_mean": area_rand.mean(), "test_acc_area_std": area_rand.std(),
                            "mean_escalation_rate": trapz_mean(c["random_esc"]).mean()})
            for score in scores:
                area = {d: trapz_mean(c["acc"][(score, d)]) for (s_, d) in keys if s_ == score}
                for d, a in area.items():
                    summary.append({**base, "score": score, "policy_class": f"S{d}", "test_acc_area_mean": a.mean(),
                                    "test_acc_area_std": a.std(),
                                    "gain_vs_random_pp": 100 * (a - area_rand).mean(),
                                    "mean_escalation_rate": np.nanmean(c["esc"][(score, d)]),
                                    "mean_models_called": np.nanmean(c["calls"][(score, d)]),
                                    "mean_effective_depth": np.nanmean(c["depth"][(score, d)]),
                                    "example_policy_mid_budget_split0": json.dumps(res[0]["policies"].get((score, d)))})
                g = {"dataset": ds, "cost": cost, "score": score}
                if 1 in area and 2 in area:
                    g["S2_minus_S1_pp"] = 100 * (area[2] - area[1]).mean()
                    g["S2_minus_S1_pp_std"] = 100 * (area[2] - area[1]).std()
                for d in sorted(area):
                    if d >= 3:
                        diff = c["acc"][(score, d)] - c["acc"][(score, 2)]
                        g[f"S{d}_minus_S2_pp"] = 100 * (area[d] - area[2]).mean()
                        g[f"S{d}_better_than_S2_budget_frac"] = float(np.mean(np.nanmean(diff, axis=0) > 1e-12))
                gains.append(g)
            for score, th in res[0].get("theory", {}).items():
                theory_rows.append({"dataset": ds, "cost": cost, "score": score, "split": 0, **th})
            # shared-schema metric rows (area-averaged over budgets and splits)
            for row in summary:
                if row["dataset"] != ds or row["cost"] != cost or row.get("_emitted"):
                    continue
                row["_emitted"] = True
                metric_rows.append({
                    "experiment_id": cfg["experiment_id"], "paper_id": cfg["paper_id"],
                    "method": "cascade_escalation" if row["policy_class"].startswith("S") else row["policy_class"],
                    "commit": git_commit(), "backbone_id": "+".join(table.models), "backbone_revision": "see generations/meta",
                    "dataset_id": f"{ds} (upstream prompts @3872d8a, n={table.n}, seed={cfg.get('seed')})",
                    "split": f"stratified 50/50 cal/test x{len(res)} (seeds {sw.get('split_seed0', 42)}..{sw.get('split_seed0', 42) + len(res) - 1})", "seed": "",
                    "device": "offline analysis (generations: see live/meta)", "framework_version": "",
                    "dtype": cfg.get("dtype"), "batch_size": "", "input_length": "", "output_length": "",
                    "quality_metric": "test accuracy, area-averaged over normalised budget",
                    "quality_value": round(float(row["test_acc_area_mean"]), 5),
                    "latency_median_ms": "", "latency_p95_ms": "", "throughput": "", "peak_memory_mb": "",
                    "energy_j": "", "status": args.status if row["policy_class"].startswith("S") else "baseline",
                    "notes": "latency/memory: see live rows", "policy_class": row["policy_class"],
                    "score_name": row["score"], "model_sequence": "", "thresholds": "", "budget_index": "all (area)",
                    "budget": "", "cost_unit": COST_UNITS.get(cost, cost), "mean_cost": row.get("mean_cost", ""),
                    "escalation_rate": row.get("mean_escalation_rate", ""),
                    "mean_models_called": row.get("mean_models_called", ""),
                    "effective_depth": row.get("mean_effective_depth", ""), "n_cal": res[0]["n_cal"],
                    "n_test": res[0]["n_test"], "threshold_grid": json.dumps(sw.get("grid_sizes")),
                })
    for row in summary:
        row.pop("_emitted", None)

    # live (measured) rows
    live_csv = results / "live" / "live_summary.csv"
    if live_csv.exists():
        import pandas as pd

        meta = {}
        mp = results / "live" / "live_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text()).get("environment", {})
        for _, row in pd.read_csv(live_csv).iterrows():
            metric_rows.append({
                "experiment_id": cfg["experiment_id"] + "_live", "paper_id": cfg["paper_id"],
                "method": "cascade_escalation" if str(row["policy"]).startswith("cascade") else "single_model_baseline",
                "commit": meta.get("commit", ""), "backbone_id": f"{row['small']}->{row['large']}",
                "backbone_revision": "see generations/meta", "dataset_id": row["dataset"],
                "split": "test split of seed 42", "seed": 42, "device": meta.get("device", ""),
                "framework_version": f"torch {meta.get('torch', '')}, transformers {meta.get('transformers', '')}",
                "dtype": cfg.get("dtype"), "batch_size": 1, "input_length": "", "output_length": "",
                "quality_metric": "accuracy", "quality_value": row["accuracy"],
                "latency_median_ms": row["latency_median_ms"], "latency_p95_ms": row["latency_p95_ms"],
                "throughput": f"{row['throughput_qps']:.3f} queries/s", "peak_memory_mb": row["peak_memory_allocated_mb"],
                "energy_j": "", "status": "measured", "notes": "both models resident; torch max_memory_allocated",
                "policy_class": "S2" if str(row["policy"]).startswith("cascade") else "S1", "score_name": cfg["live"].get("score"),
                "model_sequence": row["policy"], "thresholds": row["thresholds"], "budget_index": "",
                "budget": row.get("budget_fraction", ""), "cost_unit": COST_UNITS["flops"], "mean_cost": row["mean_tflops"],
                "escalation_rate": row["escalation_rate"], "mean_models_called": 1 + row["escalation_rate"],
                "effective_depth": "", "n_cal": "", "n_test": row["n"], "threshold_grid": "exhaustive (pair)",
            })

    np.savez_compressed(out / "curves.npz", **npz)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "depth_gains.csv", gains)
    write_csv(out / "theory.csv", theory_rows)
    write_csv(out / "models.csv", model_rows)
    schema = REPO_ROOT / "Performance_Metrics" / "results" / "cascade_escalation_schema.csv"
    if metric_rows:
        if schema.exists():
            header = schema.read_text().strip().split(",")
            metric_rows = [{h: r.get(h, "") for h in header} for r in metric_rows]
        write_csv(Path(args.metrics_csv), metric_rows)
    plots(curves, model_rows, out / "plots", scores[0], costs[0], live_csv)
    (out / "sweep_meta.json").write_text(json.dumps({"sweep": sw, "scores": scores, "costs": costs,
                                                     "datasets": datasets, "commit": git_commit(),
                                                     "runtime_s": time.time() - t_start}, indent=2))
    print(f"[sweep] done in {time.time() - t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
