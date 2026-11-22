# Dataset card: MATH-500 evaluation test split

- **Dataset ID:** `math.math500.test` (`HuggingFaceH4/MATH-500`)
- **Source URL / version:** https://huggingface.co/datasets/HuggingFaceH4/MATH-500 / `main` (pin exact commit SHA on first Hugging Face load)
- **License/access terms:** MIT (`mit`); public benchmark dataset via Hugging Face Datasets
- **Modality/task:** text / `mathematical_reasoning`
- **Role:** evaluation
- **Split:** test
- **Number of examples:** 500 (plus 4 synthetic examples in `sample_fixture.jsonl` for offline smoke tests)
- **Fields/labels:** `problem` (`str`), `solution` (`str`), `answer` (`str`, ground-truth label), `subject` (`str`), `level` (`int`), `unique_id` (`str`)
- **Preprocessing ID/version:** `dipa.math_cot_boxed.v1`
- **Tokenizer/processor revision:** N/A in loader (returns raw text and chat messages; downstream backbone applies its own tokenizer and chat template)
- **Seed/sampling strategy:** Deterministic subsampling via `random.Random(seed)` (`seed: 42` default) when `num_samples` is specified and smaller than dataset size; `num_samples: null` preserves full split order
- **Cache/download location:** Local Hugging Face cache (e.g., `~/.cache/huggingface/datasets`, never committed to Git)

## Loader contract

`load_math500(config: Math500LoaderConfig | None = None) -> list[dict[str, Any]]` returns plain Python dictionaries so `Dataloaders/` remains decoupled from model tokenizers, `Backbones/`, and `Adaptive_Stopping/`.

- **Output keys per item:**
  - `instance_id` (`str`): `unique_id`
  - `prompt` (`str`): raw `problem` text
  - `messages` (`list[dict[str, str]]`): `[{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}]`, defaulting to `DEFAULT_SYSTEM_PROMPT` (`"Please reason step by step, and put your final answer within \\boxed{}."`)
  - `ground_truth` (`str`): `answer`
  - `metadata` (`dict[str, Any]`): `{"subject": str, "level": int, "solution": str, "split": str, "dataset_id": str}`
- **Tensor shapes, dtypes, padding, collator:** None at the dataloader boundary; returned items are untokenized text records ready for backbone-specific chat templating and collation.
- **Offline mode vs Hugging Face mode:** Setting `local_jsonl_path` reads JSONL locally without importing `datasets`. Leaving `local_jsonl_path=None` lazy-imports `datasets.load_dataset(dataset_id, split="test", revision=revision)`.

## Minimal smoke test

```bash
pytest Dataloaders/tests/test_math500_loader.py
```

The test suite exercises `Dataloaders/evaluation/math500/sample_fixture.jsonl` offline with zero network calls or Hugging Face downloads.

## Leakage and privacy checks

- **Split separation:** `Math500LoaderConfig` restricts `split` to `"test"` only. This split is strictly held out for final evaluation; never draw training, tuning, or calibration subsets from `test`.
- **Permissions and redistribution:** Upstream `HuggingFaceH4/MATH-500` is MIT-licensed and contains public competition mathematics problems with no PII or sensitive user data.
- **Repository hygiene:** Do not commit downloaded Hugging Face archives or cache directories; only the 4-example original synthetic `sample_fixture.jsonl` is stored in the repository.
