# B: Minimal step size k (Algorithm 1 of the proceedings version)

Part of the ST-ES implementation ([paper card](../PAPER_CARD.md)).

## Why

The proceedings-format PDF of the paper (`75_Step_Tagging_Early_Stopping.pdf`) is identical to arXiv v2 except for Algorithm 1. There it gains a "minimal step size k ∈ ℕ" and generates each step "where |s_i| > k". The paper gives no value for k. The main text (Eq. 1), arXiv v2 and the authors' released code all split at every `"\n\n"`, which is what our main results do (k = 0).

## Definition and implementation

- **Rule.** A `"\n\n"` delimiter closes a step only when the step is longer than k generated tokens, counted from the end of the previous step and including the delimiter. A shorter piece is merged into the next step.
- **Where it lives.** `common/step_merge.py`:
  - `StepMerger` consumes the raw step stream produced by the online `StepSegmenter`, so the live loop and offline replay merge identically.
  - `merge_steps` is the offline form; `segment_with_min_size(k)` plugs it into the replay (`replay_lib.prepare(..., steps_fn=...)`).
  - `k=None` keeps plain splitting.
- **Effects.** Merged steps end at original delimiters, so a stop can only happen at an original step boundary. A merged step leaves the thinking phase if any of its parts did.
- **Tests.** `tests/test_step_merge.py` has 19 CPU tests:
  - no-op for k=None or k=0;
  - text preserved;
  - every closed step longer than k;
  - online equal to offline;
  - stops only at original boundaries;
  - invalid k rejected.

## Protocol

```bash
python -m Adaptive_Stopping.experiments.st_es.ablation_min_step --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
python -m Adaptive_Stopping.experiments.st_es.ablation_min_step --config Adaptive_Stopping/configs/st_es/math500.yaml
```

For each k ∈ {none, 8, 16, 32, 64}:

1. Re-segment the cached traces with merging.
2. Replay the full grid on the calibration split (13 step types × δ ∈ [0, 20], plus step counts).
3. Select the 85/90/95% configurations.
4. Evaluate them on the held-out split.

Recalibrating per k is the same protocol as the main results. Keeping the k = 0 configuration fixed would bias the comparison towards k = 0. The `k = none` rows reproduce the main results exactly (GSM8K: 0.815 / 779 tokens and 0.830 / 922 tokens), which checks the pipeline. Results are in `experiments/st_es/results/<dataset>/ablation_min_step.csv`.

## Results

### GSM8K (200 held-out problems; standard 0.830 at 1786 tokens)

| k | Reasoning steps per trace | ST-ES 85/90% | ST-ES 95% | Step-count 95% |
|---|---:|---|---|---|
| none (main) | 39.1 | verification δ=0: **0.815** / 779 tokens | exploration δ=0: **0.830** / 922 | k=20: 0.840 / 947 |
| 8 | 36.3 | verification δ=0: 0.815 / 785 | exploration δ=0: 0.830 / 927 | k=18: 0.825 / 910 |
| 16 | 31.3 | verification δ=0: 0.810 / 795 | exploration δ=0: 0.835 / 930 | k=16: 0.820 / 893 |
| 32 | 24.6 | verification δ=0: 0.810 / 863 | formula substitution δ=4: 0.760 / 913 | k=14: 0.845 / 926 |
| 64 | 16.5 | 85%: formula substitution δ=2: 0.700 / 771; 90%: verification δ=0: 0.805 / 946 | formula substitution δ=3: 0.795 / 985 | k=9: 0.825 / 857 |

### MATH-500 (150 held-out problems; standard 0.807 at 3710 tokens)

| k | Reasoning steps per trace | ST-ES 85% | ST-ES 90% | ST-ES 95% |
|---|---:|---|---|---|
| none (main) | 90.3 | self-talk δ=3: **0.720** / 1311 | self-talk δ=4: **0.727** / 1491 | self-talk δ=6: **0.767** / 1757 |
| 8 | 81.3 | self-talk δ=3: 0.720 / 1317 | self-talk δ=4: 0.727 / 1497 | self-talk δ=5: 0.760 / 1632 |
| 16 | 69.0 | self-talk δ=3: 0.727 / 1337 | exploration δ=0: 0.687 / 1383 | self-talk δ=6: 0.767 / 1792 |
| 32 | 52.8 | self-talk δ=3: 0.733 / 1412 | exploration δ=0: 0.707 / 1473 | self-talk δ=5: 0.760 / 1760 |
| 64 | 35.0 | self-talk δ=2: 0.740 / 1404 | exploration δ=0: 0.720 / 1676 | formula substitution δ=6: 0.767 / 2451 |

(The step-count rows are in the CSV.) On MATH-500, k shifts accuracy by at most about 2 points at the 85% and 95% targets. The 90% target becomes unstable for k ≥ 16, because calibration switches to "exploration, δ=0" and that choice transfers worse. At k = 64 the 95% configuration needs 40% more tokens.

## Conclusion

- **Small k (≤ 16 tokens) changes almost nothing.** It only merges very short pieces ("Ok.", "Which is:"), which seldom trigger a stop. The selected configurations and their accuracy/token trade-off stay the same within about 1 point and 15 tokens.
- **Large k (≥ 32 tokens, and ≥ 16 on MATH-500) hurts.** A step must then span several reasoning moves. The tag of a merged step blurs (one keyword anywhere decides it), the first trigger arrives later (more tokens), and calibration starts picking less stable triggers such as formula substitution or exploration δ=0. These transfer worse to the test split: GSM8K 0.760 and 0.700 at some targets; MATH-500 0.687 at 90%, and 2451 tokens at 95% for k = 64.
- **Main results keep k = 0.** For this model and tagger, plain splitting at every `"\n\n"` (the paper's Eq. 1) is the right setting. A minimal step size only makes sense with a tagger that labels long, multi-part steps reliably.
