# Dataset card: Passkey retrieval (synthetic)

- **Dataset ID:** `text.passkey_retrieval.synthetic`
- **Source URL / version:** generated locally by [`passkey_retrieval.py`](passkey_retrieval.py); no external download
- **License/access terms:** none needed, text is produced by the generator in this repository
- **Modality/task:** text, long-context retrieval (needle in a haystack)
- **Role:** synthetic / benchmark
- **Split:** `synthetic` (single split, there is no training use)
- **Number of examples:** set by `num_samples` in the experiment config (20 in the DropKV benchmark)
- **Fields/labels:** `prompt` (str), `answer` (5-digit string), `depth` (float in [0, 1]), `n_filler` (int)
- **Preprocessing ID/version:** chat template of the consuming backbone, applied by the experiment script
- **Tokenizer/processor revision:** the backbone's tokenizer; only used to size the prompt to a target token count
- **Seed/sampling strategy:** `random.Random(seed)`; key and needle depth drawn uniformly per sample
- **Cache/download location:** nothing is cached

## Loader contract

`build_passkey_dataset(tokenizer, target_tokens, num_samples, seed)` returns a list of `PasskeySample`. Each prompt is an instruction, filler sentences with one sentence stating the pass key at a random depth, and a final question. Prompt length is within a few tokens of `target_tokens` before the chat template is added. `is_correct(generated_text, answer)` is true when the key appears in the generated text. No batching or padding is done by the loader.

## Minimal smoke test

```bash
python -c "
from transformers import AutoTokenizer
from Dataloaders.synthetic.passkey_retrieval import build_passkey_dataset
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
s = build_passkey_dataset(tok, 500, 2, seed=0)
print(len(tok(s[0].prompt).input_ids), s[0].answer, round(s[0].depth, 2))
"
```

## Leakage and privacy checks

Fully synthetic, no personal data, safe to regenerate and redistribute. There is no train/test split to leak between. Because the task is synthetic it only measures whether the stored key survives in the model's context; it does not replace RULER or LongBench.
