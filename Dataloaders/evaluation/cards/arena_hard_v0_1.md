# Dataset card: Arena-Hard-Auto v0.1 questions

- **Dataset ID:** `text.arena_hard_v0_1.eval`
- **Source URL / version:** https://huggingface.co/datasets/lmarena-ai/arena-hard-auto, file `data/arena-hard-v0.1/question.jsonl` at revision `15f3746e21432264ce9b453999bde4f3c946d2e6`
- **License/access terms:** Apache-2.0 (HF dataset card metadata, checked 2026-11-23). Not redistributed here.
- **Modality/task:** text / instruction following (prompts only)
- **Role:** benchmark
- **Split:** eval (v0.1 question set)
- **Number of examples:** 500 (counted after download)
- **Fields/labels:** `uid` (id), `prompt` (used as prompt), `category`, `cluster`, `language` (unused).
- **Preprocessing ID/version:** `raw_text.v1` (no transformation; text used as-is)
- **Tokenizer/processor revision:** none in the loader; tokenization happens in the serving model (see model cards)
- **Seed/sampling strategy:** `random.Random(seed).sample(range(N), k)`, returned in ascending index order; default seed 42
- **Cache/download location:** `$DA4022_DATA_CACHE/text.arena_hard_v0_1.eval/` (default `~/.cache/da4022/datasets/`), not committed; sha256 verified on every load

## Loader contract

`Dataloaders.evaluation.text_prompts.load_prompts("text.arena_hard_v0_1.eval", num_samples, seed)` returns a list of
`PromptRecord(id: str, prompt: str, reference: str | None, dataset_id: str)`. Plain strings, no tensors,
no padding or collation (prompts are sent one request at a time to a serving backend).

## Minimal smoke test

```bash
python -c "from Dataloaders.evaluation.text_prompts import load_prompts; r = load_prompts('text.arena_hard_v0_1.eval', 3, 42); print(len(r), r[0].id, r[0].prompt[:60])"
```

## Usage notes

The paper uses 64 Arena-Hard prompts for its TTFT/TPS measurements. We use a seeded subset of 64 prompts (`configs/full.yaml`, matching the paper; `configs/default.yaml` uses 32) as latency prompts only; judge-based win rates are not reproduced.

## Leakage and privacy checks

Evaluation-only use; no training or calibration on this data. The file is downloaded from the pinned
source at run time and never committed. It contains public benchmark prompts and is not known to contain personal data.
