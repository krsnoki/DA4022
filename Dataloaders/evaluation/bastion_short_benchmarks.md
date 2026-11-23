# Dataset cards: BASTION short-context benchmarks

The eight short-context benchmarks from Table 1 of the BASTION paper, as loaded by the official BASTION code (`Speculative_Decoding/BASTION/upstream`, commit `1989b80`, `bastion/benchmark.py::DATASETS`) and by the reproduction harness `Speculative_Decoding/BASTION/run_benchmark.py`. All eight are **benchmark prompt sets**. The method uses them only to time generation (speedup, average accepted length τ) and to check losslessness. GSM8K, MATH-500 and AIME25 reference answers are also used for a boxed-answer accuracy check. No data here is used for training. The registry entries are in [`../registry.yaml`](../registry.yaml).

This file groups the eight cards because they share one loader, one prompt pipeline and one selection rule. The template fields are repeated per dataset below. The shared sections follow.

## Summary

| Key | HF dataset (config) | Split | Items (verified) | E1 cap | Sweep cap | Licence (dataset card) |
|---|---|---|---:|---:|---:|---|
| `gsm8k` | `openai/gsm8k` (`main`) | test | 1319 | 64 | 16 | MIT |
| `math500` | `HuggingFaceH4/MATH-500` | test | 500 | 64 | 16 | not stated (see card) |
| `aime25` | `math-ai/aime25` | test | 30 | 30 (all) | 16 | Apache-2.0 |
| `humaneval` | `openai/openai_humaneval` | test | 164 | 64 | 16 | MIT |
| `mbpp` | `google-research-datasets/mbpp` (`sanitized`) | test | 257 | 64 | 16 | CC-BY-4.0 |
| `lcb` | `livecodebench/code_generation_lite` (`release_v2`) | test | 511 | 64 | 16 | "cc" (variant not stated) |
| `mt-bench` | `HuggingFaceH4/mt_bench_prompts` | train | 80 × 2 turns | 40 (80 turns) | 16 (32 turns) | Apache-2.0 |
| `alpaca` | `tatsu-lab/alpaca` | train | 52002 | 64 | 16 | CC-BY-NC-4.0 |

- The item counts are the line counts of the upstream prompt cache (`upstream/cache/<key>.jsonl`). They match `len(load_dataset(...))` for every dataset except LCB, which is rebuilt (see below).
- **E1** is AR vs DFlash vs BASTION on Qwen3-4B and Qwen3-8B. Per mode and model it covers 6 × 64 + 30 + 40 = 454 items, or 494 generation turns.
- **Sweep** (E4) is the fixed-budget sweep on Qwen3-8B, 16 items per benchmark.
- The paper does not state per-benchmark sample counts. The caps above are our choice, made to fit the GPU time available, and are not the paper's setting.

## Per-dataset fields

### GSM8K
- **Dataset ID:** `text.gsm8k.main.test`
- **Source URL / version:** <https://huggingface.co/datasets/openai/gsm8k>, config `main`, revision `740312add88f781978c0658806c59bc2815b9866` (cached 2026-11-21)
- **License/access terms:** MIT (dataset card). Public, no gating.
- **Modality/task:** text / grade-school math word problems
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 1319
- **Fields/labels:** `question`, `answer` (worked solution ending `#### <number>`)
- **Preprocessing ID/version:** upstream format `"{question}\nPlease reason step by step, and put your final answer within \boxed{}."`
- **Tokenizer/processor revision:** target tokenizer (see [Loader contract](#loader-contract))
- **Seed/sampling strategy:** shuffle with seed 0, then the first 64 (E1) or 16 (sweep). See [Selection rule](#selection-rule).
- **Cache/download location:** `~/.cache/huggingface/` and `Speculative_Decoding/BASTION/upstream/cache/gsm8k.jsonl` (not committed)

### MATH-500
- **Dataset ID:** `text.math500.test`
- **Source URL / version:** <https://huggingface.co/datasets/HuggingFaceH4/MATH-500>, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- **License/access terms:** **not stated on the HF dataset card.** The card says the 500 problems are the MATH subset from OpenAI's *Let's Verify Step by Step* ([openai/prm800k](https://github.com/openai/prm800k)). The GitHub repos [openai/prm800k](https://github.com/openai/prm800k) and [hendrycks/math](https://github.com/hendrycks/math) are MIT-licensed, but neither page states a separate licence for the data. Treat the dataset licence as not verified.
- **Modality/task:** text / competition mathematics
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 500
- **Fields/labels:** `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`
- **Preprocessing ID/version:** upstream format `"{problem}\nPlease reason step by step, and put your final answer within \boxed{}."`
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 64 / 16
- **Cache/download location:** HF cache and `upstream/cache/math500.jsonl`

### AIME25
- **Dataset ID:** `text.aime25.test`
- **Source URL / version:** <https://huggingface.co/datasets/math-ai/aime25>, revision `563bb8404243c5f09de6ec262f2db674fe5bce9b`
- **License/access terms:** Apache-2.0 (dataset card)
- **Modality/task:** text / competition mathematics (AIME 2025)
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 30
- **Fields/labels:** `problem`, `answer`, `id`
- **Preprocessing ID/version:** upstream format `"{problem or question}\nPlease reason step by step, and put your final answer within \boxed{}."`
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** E1 uses all 30 in the original order, with no shuffle because cap ≥ size. The sweep uses the first 16 of the seed-0 permutation.
- **Cache/download location:** HF cache and `upstream/cache/aime25.jsonl`

### HumanEval
- **Dataset ID:** `text.humaneval.test`
- **Source URL / version:** <https://huggingface.co/datasets/openai/openai_humaneval>, config `openai_humaneval`, revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`
- **License/access terms:** MIT (dataset card)
- **Modality/task:** text / Python code generation
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 164
- **Fields/labels:** `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`
- **Preprocessing ID/version:** upstream format ``"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"``
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 64 / 16
- **Cache/download location:** HF cache and `upstream/cache/humaneval.jsonl`

### MBPP (sanitized)
- **Dataset ID:** `text.mbpp.sanitized.test`
- **Source URL / version:** <https://huggingface.co/datasets/google-research-datasets/mbpp>, config `sanitized`, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- **License/access terms:** CC-BY-4.0 (dataset card)
- **Modality/task:** text / Python code generation
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 257
- **Fields/labels:** `source_file`, `task_id`, `prompt`, `code`, `test_imports`, `test_list`
- **Preprocessing ID/version:** upstream format is the raw `prompt` field, with no instruction or tests appended
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 64 / 16
- **Cache/download location:** HF cache and `upstream/cache/mbpp.jsonl`

### LiveCodeBench (code_generation_lite, release_v2)
- **Dataset ID:** `text.livecodebench.release_v2.test`
- **Source URL / version:** <https://huggingface.co/datasets/livecodebench/code_generation_lite>, config `release_v2`, revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` (main as of 2026-11-21)
- **License/access terms:** the dataset card front matter says `license: cc`, without naming a CC variant. The [LiveCodeBench GitHub repo](https://github.com/LiveCodeBench/LiveCodeBench) is MIT, but it states no licence for the problems. **Not verified.** The problems come from public contest platforms.
- **Modality/task:** text / competitive-programming code generation
- **Role:** benchmark
- **Split:** test
- **Number of examples:** 511 (400 lines of `test.jsonl` + 111 lines of `test2.jsonl`)
- **Fields/labels:** `question_title`, `question_content`, `platform`, `question_id`, `contest_id`, `contest_date`, `starter_code`, `difficulty`, `public_test_cases`, `private_test_cases`, `metadata`
- **Preprocessing ID/version:** upstream format `"Write a complete solution to the following programming problem.\nReturn only the solution code.\n\n{question_content}"`. `starter_code` is not included.
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 64 / 16
- **Cache/download location:** HF hub cache (about 2 GB for the two JSONL files) and `upstream/cache/lcb.jsonl`

**Rebuild workaround (deviation from upstream).** Upstream calls `load_dataset("livecodebench/code_generation_lite", "release_v2", split="test")`. That HF repo contains only a loading script (`code_generation_lite.py`) plus raw JSONL files, and `datasets>=4` refuses dataset scripts. This includes the 4.8.5 pinned in upstream's `uv.lock`; this environment has `datasets` 5.0.1. The script's `ALLOWED_FILES["release_v2"]` is `["test.jsonl", "test2.jsonl"]`, and the config yields every line of these two files in that order. [`Speculative_Decoding/BASTION/tools/prepare_lcb.py`](../../Speculative_Decoding/BASTION/tools/prepare_lcb.py) downloads exactly these two files from the pinned revision, applies upstream's own `DATASETS["lcb"]["format"]`, and writes `upstream/cache/lcb.jsonl` in upstream's `{"turns": [...]}` format. Upstream's loader then reads that cache unchanged. This avoids downgrading to `datasets<4` with remote code enabled. Run it once before any run that includes `lcb`:

```bash
cd Speculative_Decoding/BASTION && .venv/bin/python tools/prepare_lcb.py
```

### MT-Bench
- **Dataset ID:** `text.mt_bench_prompts.train`
- **Source URL / version:** <https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts>, revision `e3a795c5e9a82ee40611c416b8a7786c73198991`
- **License/access terms:** Apache-2.0 (dataset card). The prompts were created by LMSYS ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).
- **Modality/task:** text / multi-turn chat
- **Role:** benchmark. `train` is the only split.
- **Split:** train
- **Number of examples:** 80 questions × 2 turns
- **Fields/labels:** `category`, `prompt` (list of 2 user turns), `reference`, `prompt_id`
- **Preprocessing ID/version:** upstream uses `prompt` as the list of turns (`multi_turn: True`). Turn 2 is sent after an assistant message that contains the **AR baseline's** turn-1 output, for every mode, so all modes see the same turn-2 context. The sweep has no AR mode, so it reuses E1's AR outputs through `--history-jsonl`.
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 40 (E1, 80 turns) / 16 (sweep, 32 turns)
- **Cache/download location:** HF cache and `upstream/cache/mt-bench.jsonl`

### Alpaca
- **Dataset ID:** `text.alpaca.train`
- **Source URL / version:** <https://huggingface.co/datasets/tatsu-lab/alpaca>, revision `dce01c9b08f87459cf36a430d809084718273017`
- **License/access terms:** CC-BY-NC-4.0 (dataset card). **Non-commercial use only.**
- **Modality/task:** text / instruction following
- **Role:** benchmark (prompts only). `train` is the only split.
- **Split:** train
- **Number of examples:** 52002
- **Fields/labels:** `instruction`, `input`, `output`, `text`. Only `instruction` and `input` are used.
- **Preprocessing ID/version:** upstream format is `instruction` if `input` is empty, else `"{instruction}\n\n{input}"`
- **Tokenizer/processor revision:** target tokenizer
- **Seed/sampling strategy:** seed 0 shuffle, first 64 / 16
- **Cache/download location:** HF cache and `upstream/cache/alpaca.jsonl`

## Selection rule

Upstream (`bastion/benchmark.py`):

1. `_run_transformers` calls `random.seed(0)`, `np.random.seed(0)` and `torch.manual_seed(0)`. This overrides the module-level `random.seed(42)` set at import. The call happens before the models and tokenizer are loaded.
2. `load_and_process_dataset(name)` reads `upstream/cache/<name>.jsonl`. On the first run it creates that file from `load_dataset(...)` in the dataset's native row order.
3. `_limit_dataset(dataset, max_samples)`: if `max_samples` is `None` or `len(dataset) <= max_samples`, the dataset is returned **unchanged, in its original order**. Otherwise it runs `random.shuffle(dataset)` on the global Python RNG and keeps `dataset[:max_samples]`.

Upstream runs one dataset per process, so every dataset is shuffled from a fresh seed-0 state. The harness runs several datasets per process. To match upstream, it uses `repro_lib.upstream_selection_indices(n, k)`, which is `random.Random(0).shuffle(list(range(n)))[:k]` with the same no-shuffle rule. This is computed independently per dataset and does not touch global RNG state. The resulting permutation is identical to upstream's only if nothing between `random.seed(0)` and the shuffle draws from Python `random`. `tests/test_selection.py` checks that loading models and tokenizers does not. Each record stores `orig_idx`, the item's index in the native order. Because a seed-0 permutation of `n` items does not depend on `k`, the 16-item sweep selection is a prefix of the 64-item E1 selection. For aime25 it is a subset of E1's full 30.

First selected `orig_idx` (from `run_benchmark.py --dry-run`, cap 64 / aime25 30 / mt-bench 40):

| Key | First 8 `orig_idx` |
|---|---|
| gsm8k | 1101, 120, 474, 991, 192, 1020, 350, 480 |
| math500 | 419, 459, 130, 431, 370, 26, 201, 56 |
| aime25 | 0, 1, 2, 3, 4, 5, 6, 7 (E1); 3, 10, 18, 17, 5, 0, 7, 26 (sweep, cap 16) |
| humaneval | 146, 67, 68, 94, 54, 50, 97, 71 |
| mbpp | 45, 118, 210, 1, 227, 76, 19, 99 |
| lcb | 451, 26, 236, 116, 466, 376, 264, 67 |
| mt-bench | 56, 44, 42, 50, 24, 31, 68, 11 |
| alpaca | 34051, 26774, 34659, 25861, 34768, 35334, 9077, 32375 |

## Loader contract

- **Item format:** each line of `upstream/cache/<key>.jsonl` is `{"turns": [str, ...]}`. There is 1 turn per item, except MT-Bench, which has 2.
- **Prompt construction:** messages are user turns only, with no system prompt. The prompt is `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`. Qwen3 runs in non-thinking mode, so the prompt ends with an empty `<think>\n\n</think>\n\n` block. Then `tokenizer.encode(text, return_tensors="pt")`.
- **Tokenizer:** the target model's tokenizer at the pinned revision (`Qwen/Qwen3-4B` @ `1cfa9a72…`, `Qwen/Qwen3-8B` @ `b968826d…`; see `Backbones/registry.yaml`).
- **Batch:** `input_ids` `int64[1, L]`, batch size 1. There is no padding, attention mask or truncation. Generation is capped at `max_new_tokens` = 2048 and stops at `<|im_end|>`.
- **Per-record metadata** (JSONL in `Speculative_Decoding/BASTION/runs/`, git-ignored): dataset, `orig_idx`, `sample_idx`, `turn`, `input_hash` (hash of the tokenized prompt), `mode`, `fixed_budget`, plus generation statistics. Run metadata stores the sha256 of each prompt cache file. Expected hashes are listed in `../registry.yaml` (`prompt_jsonl_sha256`).
- **Versioning caveat:** upstream calls `load_dataset` without `revision=`. The revisions above are the snapshots that were cached when the prompt files were built (2026-11-21). Compare the prompt-file sha256 before mixing runs from another machine.

## Minimal smoke test

Print the exact selection without loading any model or touching the GPU. This needs the prompt caches to exist, so run `tools/prepare_lcb.py` first for `lcb`:

```bash
cd Speculative_Decoding/BASTION
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python run_benchmark.py \
  --model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --datasets gsm8k math500 aime25 humaneval mbpp lcb mt-bench alpaca \
  --max-samples 64 --cap aime25=30 --cap mt-bench=40 --run-name dryrun --dry-run
# expected: one line per dataset, e.g. "mt-bench: size=80 cap=40 selected=40 turns=80 orig_idx[:10]=[56, 44, ...]"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_selection.py -q
```

## Leakage and privacy checks

- **No training or tuning on these sets.** BASTION is training-free, and no hyperparameter is tuned on these prompts. Two runtime calibrations exist. (1) The verification-latency roofline fit, which runs on dummy token batches and uses no dataset text. (2) The draft/non-verify latency profile, which in both upstream and the harness is measured on the first selected sample of the first dataset in the run that creates the per-(GPU, model) cache. The profile measures latency only and reads no labels.
- **Drafter training data unknown.** The DFlash drafter model cards do not state their training data. Overlap with these benchmarks (for example Alpaca, which is widely used for instruction tuning) cannot be ruled out. This could raise acceptance rates, but would not change output correctness at T=0.
- **Split use:** MT-Bench and Alpaca expose only a `train` split. They are used here purely as prompt sets.
- **Licences:** Alpaca is CC-BY-NC-4.0, which means non-commercial use only. The licences for MATH-500 and LiveCodeBench are not verified (see above). Do not redistribute the prompt caches or generated outputs beyond what each licence allows.
- **Privacy:** these are public benchmark prompts. No personal data is expected. Nothing from these datasets is committed. Prompt caches, HF caches and run records stay in git-ignored or user-cache locations.
