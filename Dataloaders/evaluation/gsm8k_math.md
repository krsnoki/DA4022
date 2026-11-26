# Dataset card: math-reasoning splits (GSM8K, MATH train, MATH-500)

The four registry entries below are read by [`gsm8k_math.py`](gsm8k_math.py). Revisions, licences and split sizes were checked on the Hugging Face Hub on 2026-11-25.

| Dataset ID | Source / pinned revision | Licence | Role | Split | Examples |
|---|---|---|---|---|---:|
| `text.gsm8k.train` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` @ `740312a` | MIT | calibration | train | 7,473 |
| `text.gsm8k.test` | same | MIT | test | test | 1,319 |
| `text.math.train` | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math), 7 subjects @ `21a5633` | MIT | calibration | train | 7,500 |
| `text.math500.test` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) @ `6e4ed1a` | not stated on the card (subset of MATH, MIT; selected in openai/prm800k) | test | test | 500 |

- **Modality/task:** text; math word problems (GSM8K) and competition mathematics (MATH) with a single final answer.
- **Fields/labels:**
  - GSM8K: `question`; the gold answer is the number after `####` in `answer`, with commas removed.
  - MATH train: `problem`; the gold answer is the content of the last `\boxed{...}` in `solution`.
  - MATH-500: `problem` and `answer`. `level` and `subject` are kept as metadata.
- **Preprocessing ID/version:** `math_cot_boxed.v1` = `"{question}\nPlease reason step by step, and put your final answer within \boxed{}."` as a single user turn with no system prompt (the DeepSeek-R1 model card's advice). `raw_question.v1` (question only) is also available.
- **Tokenizer/processor revision:** the backbone's tokenizer and chat template (see the backbone card); tokenisation happens in the backbone wrapper, not here.
- **Seed/sampling strategy:**
  - `random.Random(seed).shuffle` over the row indices of the pinned split; then rows `[offset, offset + num_samples)` are taken.
  - The registry default seed is 42.
  - The first `n` rows of a larger subset are the `n`-row subset with the same seed, so live-benchmark subsets nest inside the replay subsets.
- **Cache/download location:** the Hugging Face cache (`HF_HOME`). The loader reads the exact files listed under `files:` in the registry at the pinned revision via `hf_hub_download`, so it also works offline once cached. Nothing is committed.

## Loader contract

`load_math_reasoning(dataset_id, num_samples=None, seed=None, offset=0) -> list[MathExample]`

Each `MathExample` has `uid` (`"<dataset_id>:<row index>"`), `dataset_id`, `index`, `question`, `gold`, `gold_format` (`"number"` | `"latex"`) and `meta`. It returns Python objects, not tensors: generation methods tokenise prompts through the backbone wrapper (`build_messages` → `HFCausalLMBackbone.preprocess`). There is no padding or collation at this level.

## Minimal smoke test

```bash
python -m pytest Dataloaders/tests -q          # canned rows, no download
python -c "from Dataloaders.evaluation.gsm8k_math import load_math_reasoning as L; print(L('text.math500.test', 3)[0])"
```

## Leakage and privacy checks

- Calibration (ST-ES threshold selection) uses only the `train` splits; evaluation uses only the `test` splits. The registry `role` field records this, and the unit tests check it.
- MATH-500 is a subset of the MATH test split, so it does not overlap with `text.math.train`.
- `check_disjoint()` compares two example lists by row and by question text.
- The data are public benchmark problems with no personal data. They are not redistributed here.
