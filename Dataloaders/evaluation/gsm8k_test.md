# Dataset card: GSM8K test

- **Dataset ID:** text.gsm8k.test
- **Source URL / version:** https://huggingface.co/datasets/openai/gsm8k (config `main`). Dataset revision: `740312add88f781978c0658806c59bc2815b9866`.
- **License/access terms:** MIT
- **Modality/task:** Text. Grade-school math word problems with a numeric answer.
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 1319 in the full test split. We evaluate 3 subsets of 200 test questions each (subsets 0, 1 and 2), giving 600 distinct questions.
- **Fields/labels:** `question`, `answer`
- **Preprocessing ID/version:** `gsm8k.build_prompt.v1`. The user message is the question, a blank line, then the instruction `Solve this step by step. End your reply with "The final answer is <number>."`. The message is wrapped in the model's chat template.
- **Tokenizer/processor revision:** The backbone's tokenizer, OLMoE-1B-7B-0125-Instruct at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`, applied through its chat template (`tokenizer.apply_chat_template`).
- **Seed/sampling strategy:** One permutation of the 1319 test indices is made with `random.Random(42).shuffle`. The permutation is cut into consecutive blocks of 200; subsets 0, 1 and 2 are used. The blocks are disjoint, so the 600 questions are distinct.
- **Batching:** Consecutive batches of 8 or 32 (`make_batches`), left padding, no truncation (`collate_prompts`).
- **Cache/download location:** the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`). Not committed.

## Loader contract

- Output keys: `question` and `answer` (text).
- Metric: exact match with flexible answer extraction, as in lm-evaluation-harness.
- Tensor shapes and dtypes: unknown. Padding and collation are as described under Batching above.

## Minimal smoke test

```bash
python -m pytest Dataloaders/tests -q
```

The pure tests need no download. The asset tests (`Dataloaders/tests/test_gsm8k_assets.py`) skip when the dataset is not cached.

## Leakage and privacy checks

- This dataset is used only for evaluation. It is not used for training or calibration. XShare is calibration-free.
- The evaluation subsets come from the test split only. The tests confirm that no evaluation question appears in the train split and that the subsets are disjoint (`Dataloaders/tests/test_gsm8k_assets.py`).
- The dataset is MIT licensed. This repository does not commit a copy of the data, and the local cache stays out of Git. Result files contain model generations and question indices only.

## Registry entry

```yaml
id: text.gsm8k.test
name: GSM8K test
modality: text
task: grade_school_math_word_problems_numeric_answer
source_url: https://huggingface.co/datasets/openai/gsm8k
license: MIT
version: 740312add88f781978c0658806c59bc2815b9866
role: benchmark
split: test
preprocessing_id: gsm8k.build_prompt.v1
tokenizer: OLMoE-1B-7B-0125-Instruct, revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e (chat template)
seed: 42                         # random.Random(42).shuffle over the 1319 test indices
num_examples: 1319               # full test split; we evaluate subsets 0, 1, 2 of 200 each (600 distinct)
labels_available: true
notes: Config main. Fields question and answer. Metric is exact match with flexible answer extraction (lm-evaluation-harness). Evaluation only. Result files hold generations and question indices only.
```
