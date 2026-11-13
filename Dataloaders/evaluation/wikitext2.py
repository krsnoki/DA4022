"""WikiText-2 (raw) for perplexity evaluation and LM calibration.

Dataset ids: ``text.wikitext2_raw.test`` (evaluation) and
``text.wikitext2_raw.train`` (calibration only).  Perplexity follows the common
GPTQ/AWQ/SpinQuant protocol: the test split is joined with ``"\\n\\n"``,
tokenized once and cut into non-overlapping windows of ``seq_len`` tokens.
"""

from __future__ import annotations

from typing import List, Optional

import torch

HF_ID = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def load_wikitext2_tokens(tokenizer, split: str = "test", revision: str = REVISION) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, CONFIG, split=split, revision=revision)
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def eval_windows(tokens: torch.Tensor, seq_len: int = 2048, max_windows: Optional[int] = None) -> List[torch.Tensor]:
    n = tokens.numel() // seq_len
    if max_windows is not None:
        n = min(n, max_windows)
    return [tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0) for i in range(n)]


def calibration_windows(tokens: torch.Tensor, num_samples: int = 128, seq_len: int = 2048, seed: int = 0) -> List[torch.Tensor]:
    """Random windows from the *train* split (never from the test split)."""
    g = torch.Generator().manual_seed(seed)
    hi = tokens.numel() - seq_len - 1
    starts = torch.randint(0, hi, (num_samples,), generator=g)
    return [tokens[s : s + seq_len].unsqueeze(0) for s in starts.tolist()]
