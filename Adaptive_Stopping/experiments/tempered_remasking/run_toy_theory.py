"""CPU experiment: the anchor-fork model of Sec. 3.1 / Appendix D, end to end.

Runs the *real* sampler on :class:`~tempered_remasking.toy.AnchorForkToyModel`
and compares against exact dynamic-programming values:

1. Expected fork entropy vs T_pos for TLC (Prop. D.4), with the Lemma D.3
   bounds, LC (Remark D.5) and AR (Prop. D.9).
2. Semantic entropy of samples (Cor. D.7).
3. Toy pass@k and pass@NFE where the "correct" solution is the one the
   anchors bias the model *against* (the Fig. 1 scenario), for AR, LC, TLC,
   random, CT and TCT.

    python Adaptive_Stopping/experiments/tempered_remasking/run_toy_theory.py --out toy_results.json
"""

from __future__ import annotations

import argparse
import json
import math

import torch

import common  # noqa: F401  (sets sys.path)
from tempered_remasking import SamplerConfig, build_policy, generate
from tempered_remasking import metrics as M
from tempered_remasking import theory as th
from tempered_remasking.toy import FORK_B, AnchorForkToyModel


def build_model(length: int = 32, fork: int = 2) -> AnchorForkToyModel:
    # Fig. 1-like: an early fork ("import ?????") and several confident anchors later on.
    anchors = {6: (0.99, 0.12), 11: (0.985, 0.12), 17: (0.98, 0.1), 23: (0.975, 0.1), 29: (0.97, 0.08)}
    return AnchorForkToyModel(length=length, fork_position=fork, anchors=anchors, filler_confidence=0.9)


def run(model, policy, n, seed, token_temperature=1.0, block_length=None, confidence_mode="max"):
    cfg = SamplerConfig(gen_length=model.length, block_length=block_length or model.length,
                        token_temperature=token_temperature, confidence_mode=confidence_mode, record_entropy=True)
    res = generate(model, model.prompt(n), policy, cfg, model.mask_token_id, generator=torch.Generator().manual_seed(seed))
    f = model.fork_position
    return res, res.entropy_at_unmask[:, f], res.generated[:, f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="Monte Carlo samples per setting")
    ap.add_argument("--problems", type=int, default=50, help="toy 'problems' (seeds) for pass@k")
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=8, help="semi-AR block size for the pass@k study")
    ap.add_argument("--confidence-modes", nargs="+", default=["max", "sampled"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = build_model()
    out = {"model": {"length": m.length, "fork": m.fork_position, "anchors": m.anchors, "h0": m.h0,
                     "filler_confidence": m.filler_confidence}}

    # ---- 1/2: fork entropy vs T_pos --------------------------------------------------
    consts = m.lemma_constants()
    ratio = th.proposition_d4_threshold([d for d, _ in consts.values()], [D for _, D in consts.values()])
    rows = []
    print(f"H0={m.h0:.4f}  sum(eta)={sum(m.etas.values()):.3f}  LC min={m.lc_fork_entropy():.4f}  "
          f"AR={m.ar_fork_entropy():.4f}  Prop.D.4 ratio Δ/δ={ratio:.3f}")
    print(f"{'T_pos':>7} {'E[H] exact':>11} {'E[H] MC':>9} {'sem.H exact':>12} {'sem.H MC':>9} {'Lemma D.3 ok':>13}")
    for T in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, math.inf]:
        ex = m.exact_tlc(T)
        _, H, fv = run(m, build_policy("tlc", tpos=T), args.n, seed=1)
        pB = (fv == FORK_B).double().mean().item()
        ok = all(th.pairwise_order_bounds(d, D, T)[0] - 1e-12 <= ex["p_anchor_first"][a] <= th.pairwise_order_bounds(d, D, T)[1] + 1e-12
                 for a, (d, D) in consts.items())
        row = {"tpos": T, "expected_fork_entropy_exact": ex["expected_fork_entropy"], "expected_fork_entropy_mc": H.mean().item(),
               "semantic_entropy_exact": ex["semantic_entropy"], "semantic_entropy_mc": th.binary_entropy(pB), "lemma_d3_bounds_hold": ok}
        rows.append(row)
        print(f"{T:>7} {row['expected_fork_entropy_exact']:>11.4f} {row['expected_fork_entropy_mc']:>9.4f} "
              f"{row['semantic_entropy_exact']:>12.4f} {row['semantic_entropy_mc']:>9.4f} {str(ok):>13}")
    out["tlc_vs_tpos"] = rows
    baselines = {}
    for name, pol in [("lc", build_policy("lc")), ("ar", build_policy("ar")), ("ct", build_policy("ct", threshold=0.6)),
                      ("tct", build_policy("tct", threshold=0.6, tpos=0.1))]:
        res, H, fv = run(m, pol, args.n, seed=2)
        baselines[name] = {"expected_fork_entropy_mc": H.mean().item(),
                           "semantic_entropy_mc": th.binary_entropy((fv == FORK_B).double().mean().item()),
                           "mean_nfe": res.nfe.double().mean().item()}
        print(f"{name:>7}  E[H]={baselines[name]['expected_fork_entropy_mc']:.4f}  sem.H={baselines[name]['semantic_entropy_mc']:.4f}  "
              f"NFE={baselines[name]['mean_nfe']:.2f}")
    out["baselines"] = baselines

    # ---- 3: toy pass@k / pass@NFE (correct = minority solution FORK_B) ---------------
    settings = {"ar": build_policy("ar"), "lc": build_policy("lc"), "random": build_policy("random"),
                "tlc(T=1)": build_policy("tlc", tpos=1.0), "ct(λ=0.6)": build_policy("ct", threshold=0.6),
                "tct(T=0.1,λ=0.6)": build_policy("tct", threshold=0.6, tpos=0.1)}
    # Run under both confidence definitions: "max" (Sec. 2.2; the setting of the
    # formal analysis) and "sampled" (footnote 1; the paper's experiments). Under
    # "sampled", a fork that drew its minority value has lower confidence, which
    # biases confidence-ordered commits towards the majority value.
    out["toy_pass_at_nfe"] = {}
    for mode in args.confidence_modes:
        print(f"\nToy pass@k (correct = minority fork value), block_length={args.block_length}, T_token=0.8, confidence={mode}")
        passk = {}
        for name, pol in settings.items():
            correct, nfes = [], []
            for p in range(args.problems):
                res, _, fv = run(m, pol, args.k_max, seed=1000 + p, token_temperature=0.8,
                                 block_length=args.block_length, confidence_mode=mode)
                correct.append((fv == FORK_B).tolist())
                nfes.append(res.nfe_paper_convention.tolist())
            curve = M.pass_at_nfe_curve(correct, nfes)
            passk[name] = curve
            print(f"{name:>17}: " + "  ".join(f"k={c['k']}:{c['pass']:.3f}@{c['nfe']:.0f}NFE" for c in curve if c["k"] in (1, 4, 16, 64)))
        out["toy_pass_at_nfe"][mode] = passk

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
