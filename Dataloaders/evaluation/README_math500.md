# Dataset card: MATH-500 (and MATH train)

- **Dataset ID:** `text.math500.test` (evaluation); `text.math.train` (train)
- **Source URL / version:**
  - MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (`test` split, 500 rows). This is OpenAI's 500-problem subset of the original MATH test split, published with the *Let's Verify Step by Step* paper; see the dataset card and https://github.com/openai/prm800k.
  - MATH train: https://huggingface.co/datasets/EleutherAI/hendrycks_math, configs `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`, each `train` split.
  - Revision: unpinned (the loaders use `load_dataset(..., split=...)` without a `revision` argument). Pin a commit hash before reporting results.
- **License/access terms:** `EleutherAI/hendrycks_math` is tagged `license: mit` on its HF card, matching the original [Hendrycks MATH repo](https://github.com/hendrycks/math) (MIT). `HuggingFaceH4/MATH-500` does not carry its own license tag; since it republishes a subset of the same MATH test split, treat it under the same MIT terms as the source data until a different term is published.
- **Modality/task:** text, free-form competition-math problem solving with a final answer.
- **Role:** MATH-500 is evaluation only. A seeded subset of MATH train is used to generate reasoning traces on which the TERMINATOR exit probe is trained (the backbone itself is not trained).
- **Split:** MATH-500 `test`; MATH train `train` (all 7 configs).
- **Number of examples:**
  - MATH-500: 500 (verified from the HF dataset card metadata).
  - MATH train: 7,500 total across the 7 configs (1744 + 771 + 870 + 1295 + 869 + 1205 + 746, per the HF dataset card's per-config `train` row counts). `load_math_train` keeps 7,498 of them: rows whose solution has no braced `\boxed{...}`/`\fbox{...}` are skipped (2 rows, `algebra/888` and `algebra/1011`, use the brace-less `\boxed 2` form). This keeps the seeded subset identical to the one used for the TERMINATOR traces.
- **Fields/labels:**
  - MATH-500: `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`.
  - MATH train (each config): `problem`, `level`, `type`, `solution`. There is no `answer` field; the gold answer is extracted from `solution` with `extract_boxed`.
- **Preprocessing ID/version:** `math_reasoning.extract_boxed` for the train gold answer; `math_reasoning.normalize_answer` / `math_reasoning.is_equiv` for scoring a model's extracted answer against gold.
- **Tokenizer/processor revision:** n/a here; tokenization is the backbone's responsibility.
- **Seed/sampling strategy:** with `n` given, both loaders draw a deterministic subset via `random.Random(seed).sample` over row indices, then sort the chosen indices back into the source's original order. With `n=None`, `load_math500` returns all 500 rows in original order.
- **Cache/download location:** the Hugging Face datasets cache (`HF_HOME`, default `~/.cache/huggingface`); not committed.

## Loader contract

`Dataloaders.evaluation.math_reasoning.load_math500(n=None, seed=0)` and `load_math_train(n, seed=0)` each return a `list[MathExample]`:

```python
@dataclass(frozen=True)
class MathExample:
    uid: str            # MATH-500: unique_id; train: f"train/{config}/{index}"
    problem: str
    answer: str          # gold final answer
    subject: str | None  # MATH-500: subject field; train: config name
    level: str | None    # kept as given by the source field, not normalized
```

No tensors, no padding/collation: this loader hands back problem/answer text; batching and tokenization belong to the consumer.

## Minimal smoke test

```bash
python -m pytest Dataloaders/tests/test_math_reasoning.py -q
```

The extraction/normalization/equivalence tests need no download. The loader tests (`load_math500`, `load_math_train`) are skipped unless `RUN_NETWORK_TESTS=1` is set, since they need the Hugging Face Hub.

## Overlap note

MATH-500 is drawn from the original MATH **test** split (5,000 problems total across all 7 subjects); the loaders here use the Hendrycks MATH **train** split (7,500 problems) for `load_math_train`. These are disjoint splits of the same underlying benchmark, so MATH-500 problems do not reappear in `load_math_train`, and the probe trained on MATH train traces is never trained on a MATH-500 problem.

## Known issues

- Answer equivalence checking is imperfect. `is_equiv` first compares the two answers after `normalize_answer`, then asks `math_verify` when it is installed, since it can confirm many non-string-identical equivalent forms (`1/2` vs `0.5`, unsimplified fractions, reordered coordinate tuples, etc.), but it is not a full computer-algebra system and can still both over- and under-match on heavily formatted LaTeX. The `normalize_answer` fallback is a plain string transform and will miss any equivalence it does not explicitly cover.
- `extract_boxed` takes the *last* `\boxed{...}`/`\fbox{...}` in a solution. This matches the common MATH convention of stating the final answer last, but a solution that boxes an intermediate quantity after its final boxed answer (rare) would be scored on the wrong value.
- MATH-500's `level` and `subject` fields, and the train configs' `level`/`type` fields, are not cross-walked to a single vocabulary; `level` is kept exactly as given by each source.
