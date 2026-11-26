"""Add outcome-reward-model scores to a sampling file (Sec. 4.2, ORM selection).

    python Adaptive_Stopping/experiments/tempered_remasking/score_orm.py \
        --in results/tempered_remasking/llada_gsm8k_tct.jsonl \
        --out results/tempered_remasking/llada_gsm8k_tct.orm.jsonl
"""

from __future__ import annotations

import argparse

from common import append_jsonl, read_jsonl
from tempered_remasking.orm import ACEMATH_RM_ID, HFRewardModelScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orm", default=ACEMATH_RM_ID)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    scorer = HFRewardModelScorer(args.orm, args.revision, args.device)
    done = {r["id"] for r in read_jsonl(args.out)}
    for rec in read_jsonl(args.inp):
        if rec["id"] in done:
            continue
        scores = scorer.score(rec["question"], [s["text"] for s in rec["samples"]])
        for s, sc in zip(rec["samples"], scores):
            s["orm_score"] = sc
        rec["orm_model"] = args.orm
        append_jsonl(args.out, rec)
        print(rec["id"], max(scores))


if __name__ == "__main__":
    main()
