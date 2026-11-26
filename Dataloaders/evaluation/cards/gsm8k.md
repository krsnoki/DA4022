# Dataset card: GSM8K test split

- **Dataset ID:** `text.gsm8k.test`
- **Source URL / version:** https://github.com/openai/grade-school-math, `grade_school_math/data/test.jsonl` at commit `3101c7d5072418e28b9008a6636bde82a006892c` (repo archived 2026-04-08; same data as HF `openai/gsm8k` config `main`, test split)
- **License/access terms:** MIT (repository LICENSE, Copyright (c) 2021 OpenAI)
- **Modality/task:** text / math word problems
- **Role:** test (quality evaluation only, never calibration)
- **Split:** test
- **Number of examples:** 1319 (counted after download)
- **Fields/labels:** `question` (prompt), `answer` (worked solution; the final answer follows `####`).
- **Preprocessing ID/version:** `raw_text.v1` (no transformation; text used as-is)
- **Tokenizer/processor revision:** none in the loader; tokenization happens in the serving model (see model cards)
- **Seed/sampling strategy:** `random.Random(seed).sample(range(N), k)`, returned in ascending index order; default seed 42
- **Cache/download location:** `$DA4022_DATA_CACHE/text.gsm8k.test/` (default `~/.cache/da4022/datasets/`), not committed; sha256 verified on every load

## Loader contract

`Dataloaders.evaluation.text_prompts.load_prompts("text.gsm8k.test", num_samples, seed)` returns a list of
`PromptRecord(id: str, prompt: str, reference: str | None, dataset_id: str)`. Plain strings, no tensors,
no padding or collation (prompts are sent one request at a time to a serving backend).

## Minimal smoke test

```bash
python -c "from Dataloaders.evaluation.text_prompts import load_prompts; r = load_prompts('text.gsm8k.test', 3, 42); print(len(r), r[0].id, r[0].prompt[:60])"
```

## Usage notes

Used for cheap exact-match quality comparison of baseline vs staircase MoA (the whole test split of 1319 questions in `configs/full.yaml`; `configs/default.yaml` uses a seeded subset of 50). The instruction wrapper and answer extraction live in `Serving/staircase_streaming/quality.py`, not in the loader.

## Leakage and privacy checks

Evaluation-only use; no training or calibration on this data. The file is downloaded from the pinned
source at run time and never committed. It contains public benchmark prompts and is not known to contain personal data.
