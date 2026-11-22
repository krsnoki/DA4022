# Dataset card: WikiText-2 (raw)

- **Dataset ID:** `text.wikitext2.raw`
- **Source URL / version:** https://huggingface.co/datasets/Salesforce/wikitext, config `wikitext-2-raw-v1`. Revision: **unpinned, pin before reporting**.
- **License/access terms:** listed on the Hugging Face card as CC BY-SA 3.0 / GFDL. Verify on the card when pinning.
- **Modality/task:** text, language-model perplexity
- **Role:** evaluation (benchmark). Never used for calibration.
- **Split:** `test` (`validation` is also available)
- **Number of examples:** unknown here; record the line and token counts from the pinned revision
- **Fields/labels:** `text` (one line per row; blank lines included)
- **Preprocessing ID/version:** rows joined with `"\n\n"`, tokenized once, cut into non-overlapping `seq_len` chunks with the remainder dropped (GPTQ-style protocol)
- **Tokenizer/processor revision:** the backbone's tokenizer at `backbone.revision`
- **Seed/sampling strategy:** deterministic, no sampling. `max_chunks` takes the first N chunks.
- **Cache/download location:** the Hugging Face datasets cache (`~/.cache/huggingface`); not committed

## Loader contract

`Dataloaders.evaluation.wikitext2.load_wikitext2_chunks(tokenizer, split, seq_len, max_chunks, revision)` returns an `int64` tensor of shape `(n_chunks, seq_len)`. There is no padding, so every token is real. Perplexity is `exp(mean NLL)` over next-token predictions within each chunk; the first token of a chunk is not scored.

## Minimal smoke test

```python
import torch
from Dataloaders.evaluation.wikitext2 import token_chunks
assert token_chunks(torch.arange(10), 4).shape == (2, 4)
```

## Leakage and privacy checks

CafeQ is calibration-free, so no data from any split reaches the quantizer. The public encyclopedic text has no known private data. Redistribution follows the licence above; the repo only stores the loader, not the data.
