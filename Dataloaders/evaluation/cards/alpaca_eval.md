# Dataset card: AlpacaEval evaluation set

- **Dataset ID:** `text.alpaca_eval.eval`
- **Source URL / version:** https://huggingface.co/datasets/tatsu-lab/alpaca_eval, file `alpaca_eval.json` at revision `2edc6fad8be6b14ea7230aabfd08188da6b8b814`
- **License/access terms:** CC-BY-NC-4.0 (HF dataset card metadata, checked 2026-11-23). Non-commercial research use; the file is not redistributed here.
- **Modality/task:** text / instruction following (prompts only; reference outputs are not used)
- **Role:** benchmark
- **Split:** eval (the single evaluation set)
- **Number of examples:** 805 (counted after download)
- **Fields/labels:** `instruction` (used as prompt), `output`, `generator`, `dataset` (unused). No id field, so the 0-based record index is the id.
- **Preprocessing ID/version:** `raw_text.v1` (no transformation; text used as-is)
- **Tokenizer/processor revision:** none in the loader; tokenization happens in the serving model (see model cards)
- **Seed/sampling strategy:** `random.Random(seed).sample(range(N), k)`, returned in ascending index order; default seed 42
- **Cache/download location:** `$DA4022_DATA_CACHE/text.alpaca_eval.eval/` (default `~/.cache/da4022/datasets/`), not committed; sha256 verified on every load

## Loader contract

`Dataloaders.evaluation.text_prompts.load_prompts("text.alpaca_eval.eval", num_samples, seed)` returns a list of
`PromptRecord(id: str, prompt: str, reference: str | None, dataset_id: str)`. Plain strings, no tensors,
no padding or collation (prompts are sent one request at a time to a serving backend).

## Minimal smoke test

```bash
python -c "from Dataloaders.evaluation.text_prompts import load_prompts; r = load_prompts('text.alpaca_eval.eval', 3, 42); print(len(r), r[0].id, r[0].prompt[:60])"
```

## Usage notes

The paper uses 64 AlpacaEval prompts for its TTFT/TPS measurements. We use a seeded subset of 64 prompts (`configs/full.yaml`, matching the paper; `configs/default.yaml` uses 32) as latency prompts only; win-rate evaluation is not reproduced.

## Leakage and privacy checks

Evaluation-only use; no training or calibration on this data. The file is downloaded from the pinned
source at run time and never committed. It contains public benchmark prompts and is not known to contain personal data.
