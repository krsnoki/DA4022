# Dataset card: MMLU-Pro

- **Dataset ID:** `text.mmlu_pro.test`
- **Source URL / version:** https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro, revision not pinned yet (set `dataset.revision` to a full commit hash before reporting)
- **License/access terms:** MIT (dataset card)
- **Modality/task:** text, multiple-choice question answering (up to 10 options)
- **Role:** benchmark
- **Split:** `test` (`validation` has 70 rows and is not used by SCPS)
- **Number of examples:** 12,032 questions in total per the dataset card's discipline table; the Hub viewer shows the test split as "12k rows" (rounded) and validation as 70 rows
- **Fields/labels:** `question_id` (int), `question` (str), `options` (list of 10 str, padded with `"N/A"`), `answer` (letter), `answer_index` (0 to 9), `cot_content`, `category` (14 classes), `src`
- **Preprocessing ID/version:** `mmlu_pro.strip_na.v1` (remove trailing `"N/A"` padding; letters and `answer_index` unchanged)
- **Tokenizer/processor revision:** not applicable at load time; prompts are tokenized by the backbone wrapper
- **Seed/sampling strategy:** sort by `question_id`, then `random.Random(47).sample(pool, 256)` after removing excluded IDs (`exclude_ids_file`, empty by default)
- **Cache/download location:** Hugging Face `datasets` cache (default `~/.cache/huggingface/datasets`), or a local `.jsonl`/`.parquet` export passed as `dataset.local_path`. Never committed.

```yaml
id: text.mmlu_pro.test
name: MMLU-Pro test
modality: text
task: multiple_choice_qa
source_url: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
license: MIT
version: unknown            # pin the Hub commit
role: evaluation
split: test
preprocessing_id: mmlu_pro.strip_na.v1
seed: 47
num_examples: 256           # sampled; full split size unverified beyond "12k rows"
labels_available: true
notes: The SCPS paper's 256 held-out IDs are not public, so this sample differs from the paper's.
```

## Loader contract

`Dataloaders.evaluation.mmlu_pro.load_mmlu_pro(split, revision, local_path)` returns a list of dicts with the fields above and `options` stripped of padding. `sample_questions(records, n, seed, exclude_ids)` returns `n` records. Items are processed one question at a time (batch size 1), so there is no collator.

## Minimal smoke test

```bash
python -m pytest -q Adaptive_Stopping/tests/test_mcq_data_metrics.py -k "strip_na or sample_questions"
```

## Leakage and privacy checks

- SCPS with the plug-in surrogate trains nothing and tunes nothing on MMLU-Pro, so no controller-training split is needed. If you tune coefficients, draw a disjoint tuning sample and pass its IDs through `exclude_ids_file`.
- Do not redistribute question text in results; per-question outputs store IDs, gold letters and answer classes only (completion text is dropped unless `to_dict(include_text=True)` is used).
- The data contains no personal information as far as the dataset card states.
