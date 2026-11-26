# C: Step-tagger error analysis, and v1 vs v2

Part of the ST-ES implementation ([paper card](../PAPER_CARD.md)).

The paper tags steps with GPT-4o-mini and with binary BERT classifiers trained on those labels. We use a rule-based substitute instead (`tagger.py`, v1). This page measures how well it approximates the paper's annotator, where it fails, and whether a second rule set (`tagger_v2.py`) helps.

## Data and protocol

- **Held-out check.** 99 reasoning steps from DeepSeek-R1-Distill-Qwen-14B on MATH500, each labelled by GPT-4o-mini. The authors released them in `yannisbelkhiter/ReasonType_Human_validation` (MIT, commit `efb7676`), downloaded at run time.
  - The v1 rules were written before these steps were seen.
  - The v2 rules were derived **only** from v1's output on our own calibration traces (GSM8K train and MATH train, 4,914 reasoning steps), and v2 was frozen before this check was run.
  - One caveat: v1's aggregate per-class scores on these steps had already been computed when v2 was written. The step texts had not been read.
- **Effect on ST-ES.** For each tagger the full grid is re-calibrated (13 step types × δ ∈ [0, 20]), the 85/90/95% configurations are re-selected, and they are evaluated on the held-out test split. Forced answers are shared through the per-trace cache.

```bash
python -m Adaptive_Stopping.experiments.st_es.tagger_analysis --configs Adaptive_Stopping/configs/st_es/gsm8k.yaml Adaptive_Stopping/configs/st_es/math500.yaml
python -m pytest Adaptive_Stopping/tests/test_tagger.py Adaptive_Stopping/tests/test_tagger_v2.py -q
```

Outputs are in [`experiments/st_es/results/tagger_analysis/`](../../../experiments/st_es/results/tagger_analysis/):

- `confusion_v1.png`, `confusion_v2.png`;
- `validation_v1_vs_v2.json`;
- `failure_examples_v1.md`;
- `calibration_effect_v1_vs_v2.csv`.

## Results on the 99 held-out steps

| | v1 (reported) | v2 |
|---|---:|---:|
| Exact 14-way agreement with GPT-4o-mini | **40.4%** | 38.4% |
| Mean one-vs-rest macro-F1 over 13 types | **0.642** | 0.630 |
| Verification: F1 (predicted / true) | 0.63 (9 / 9) | 0.64 (13 / 9) |
| Self-talk: F1 (predicted / true) | 0.69 (19 / 7) | 0.64 (24 / 7) |
| Exploration: F1 | 0.88 | 0.85 |
| Final conclusion: F1 | 0.84 | 0.88 |
| Formula substitution: F1 | 0.77 | 0.75 |

For comparison, the paper's BERT taggers score about 0.8–0.9 macro-F1 against GPT-4o-mini on their own data (Fig. 4).

**v2 does not improve agreement with the paper's annotator.** Its new rules were reasonable for the patterns we saw in our 1.5B model's traces. On the 14B model's steps, though, they mostly move ambiguous steps into self-talk, which v1 already over-predicts. v1 therefore remains the tagger of all reported results. v2 is kept as a documented negative result.

## Main failure modes (v1)

From `confusion_v1.png` and `failure_examples_v1.md`:

1. **A leading "Wait," decides the label.** GPT-4o-mini labels what the step *does* after the discourse marker: "Wait, maybe k is of the form 4m+1" is a heuristic; "Wait, let's think differently…" is exploration; "Wait, perhaps I'm misunderstanding the problem" is re-statement. v1 gives "wait" a strong self-talk weight, so these all become self-talk. This is the largest single source of error and explains why self-talk is over-predicted (19 predicted against 7 true).
2. **Verification without a check word.** Statements such as "So, sequence M, E, E is allowed" or "That also looks correct" are verification to GPT-4o-mini but contain no "check/verify" cue, so v1 recalls only 3 of 9 verification steps.
3. **Definition recall inside a calculation.** "φ(7) is equal to 6 because 7 is prime, and φ(p) = p − 1 …" is labelled definition recall, but its arithmetic makes v1 say formula substitution.
4. **Rare types are never predicted.** v1 never outputs heuristics or edge case, both of which GPT-4o-mini uses.

## Effect on calibrated ST-ES

Each tagger is re-calibrated on the calibration split, and the selected configurations are evaluated on the held-out split (from `calibration_effect_v1_vs_v2.csv`):

| Dataset | Tagger | `other` share (calibration steps) | 85% | 90% | 95% |
|---|---|---:|---|---|---|
| GSM8K (standard 0.830, 1786 tokens) | v1 | 16.6% | verification δ=0: 0.815 / 779 tokens | same as 85% | exploration δ=0: 0.830 / 922 |
| | v2 | 10.0% | verification δ=1: 0.825 / 856 | same | same |
| MATH-500 (standard 0.807, 3710 tokens) | v1 | 26.4% | self-talk δ=3: 0.720 / 1311 | self-talk δ=4: 0.727 / 1491 | self-talk δ=6: 0.767 / 1757 |
| | v2 | 18.9% | verification δ=1: 0.700 / 1303 | formula substitution δ=9: 0.740 / 2261 | exploration δ=1: 0.747 / 1799 |

v2 tags fewer steps as `other`. It shifts which step type calibration picks (towards verification with δ=1), but it does not move the accuracy–token trade-off:

- **GSM8K:** about the same (+1 point for +77 tokens).
- **MATH-500:** slightly worse at 90% and 95%.

Together with the held-out agreement result, this supports keeping v1 for the reported numbers. It also suggests that ST-ES's gains depend more on *some* consistent late-reasoning trigger than on exactly matching the paper's labels.

## What would help next

- Score the content after a leading discourse marker instead of the marker itself. This is the fix suggested by failure mode 1. It should be validated on new labelled steps, not on these 99.
- Use a small LLM annotator, or the paper's GPT-4o-mini + BERT pipeline if an API budget is available. The ST-ES code takes any object with a `tag(text) -> str` method (the `StepTagger` protocol).
