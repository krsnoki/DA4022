"""Agreement of the rule-based step tagger with GPT-4o-mini labels released by the authors.

Data: ``yannisbelkhiter/ReasonType_Human_validation`` (MIT) at a pinned commit:
reasoning steps from DeepSeek-R1-Distill-Qwen-14B on MATH500, each labelled
by GPT-4o-mini with a ReasonType category (the paper's annotator). The two
files are a class-balanced sample (4 steps per class) and a sample following
the original label distribution; they are merged and de-duplicated.

The rules in ``tagger.py`` were written before looking at these steps and
were not tuned on them, so this is a held-out check of how well the substitute
tagger approximates the paper's annotator. Only one-vs-rest metrics are
comparable to the paper's binary BERT taggers (Fig. 4, which reports their
agreement with GPT-4o-mini on other data).

Files are downloaded to ``<run_dir>/external`` (not committed) and their sha256
is recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

from Adaptive_Stopping.experiments.st_es.io_utils import DEFAULT_RUN_DIR, sha256_file, write_json
from Adaptive_Stopping.methods.step_tagging.tagger import KeywordStepTagger
from Adaptive_Stopping.methods.step_tagging.taxonomy import TAG_IDS, to_tag_id

REPO = "yannisbelkhiter/ReasonType_Human_validation"
COMMIT = "efb7676e03b66b3ff4bfad871b451dbb8bcb355f"
FILES = ["data/TO_UPLOAD_uniform.json", "data/TO_UPLOAD_original_distribution.json"]


def fetch(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for f in FILES:
        p = dest / f"{COMMIT[:7]}__{f.replace('/', '__')}"
        if not p.exists():
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{f}"
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())
        out.append(p)
    return out


def binary_scores(gold: list[str], pred: list[str], tag: str) -> dict:
    tp = sum(g == tag and p == tag for g, p in zip(gold, pred))
    fp = sum(g != tag and p == tag for g, p in zip(gold, pred))
    fn = sum(g == tag and p != tag for g, p in zip(gold, pred))
    tn = len(gold) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # macro-F1 of the binary (tag vs rest) problem, as in the paper's Fig. 4b
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    return {"support": tp + fn, "predicted": tp + fp, "precision": prec, "recall": rec, "f1": f1,
            "binary_macro_f1": (f1 + f1_0) / 2, "binary_accuracy": (tp + tn) / len(gold)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    paths = fetch(args.run_dir / "external")
    items = {}
    for p in paths:
        for row in json.loads(p.read_text(encoding="utf-8")):
            items[(row["sample_id"], row["step_id"])] = row
    rows = list(items.values())
    tagger = KeywordStepTagger()
    gold = [to_tag_id(r["category"]) for r in rows]
    pred = [tagger.tag(r["step"]) for r in rows]
    per_tag = {t: binary_scores(gold, pred, t) for t in TAG_IDS if t != "other"}
    confusion = Counter(zip(gold, pred))
    result = {
        "source": {"repo": REPO, "commit": COMMIT, "files": FILES,
                   "sha256": {p.name: sha256_file(p) for p in paths}},
        "annotator": "GPT-4o-mini (labels released by the paper's authors)",
        "steps_from": "DeepSeek-R1-Distill-Qwen-14B on MATH500",
        "tagger": tagger.name, "n_steps": len(rows),
        "exact_agreement": sum(g == p for g, p in zip(gold, pred)) / len(rows),
        "gold_distribution": dict(Counter(gold)), "pred_distribution": dict(Counter(pred)),
        "per_tag": per_tag,
        "confusion_top": [{"gold": g, "pred": p, "count": c} for (g, p), c in confusion.most_common(25)],
    }
    out = args.out or (args.run_dir / "tagger_validation.json")
    write_json(out, result)
    print(f"n={len(rows)} exact agreement={result['exact_agreement']:.3f}")
    for t, s in per_tag.items():
        print(f"{t:26s} support={s['support']:2d} pred={s['predicted']:2d} P={s['precision']:.2f} "
              f"R={s['recall']:.2f} F1={s['f1']:.2f} binary-macroF1={s['binary_macro_f1']:.2f}")


if __name__ == "__main__":
    main()
