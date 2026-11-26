from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

from Dataloaders.causal_lm import (
    ROW_SEPARATOR,
    TokenBatch,
    corpus_rows,
    corpus_spec,
    encode_text,
    token_batches,
)

ZERO_SHOT_TASKS = (
    "winogrande", "social_iqa", "piqa", "openbookqa", "lambada_openai",
    "hellaswag", "arc_easy", "arc_challenge", "mmlu",
)


def _windows(ids: Iterable[int], seq_len: int) -> Iterator[list[int]]:
    ids = iter(ids)
    while len(window := list(islice(ids, seq_len))) == seq_len:
        yield window


def _stream_ids(tok, rows: Iterable[str]) -> Iterator[int]:
    for i, text in enumerate(rows):
        yield from encode_text(tok, ROW_SEPARATOR + text if i else text)


def make_evaluation_batches(
    dataset_id: str,
    tokenizer,
    *,
    seq_len: int = 2048,
    batch_size: int = 1,
    cache_dir: str | Path | None = None,
) -> Iterator[TokenBatch]:
    if seq_len < 2 or batch_size < 1:
        raise ValueError("evaluation requires seq_len >= 2 and batch_size >= 1")
    spec = corpus_spec(dataset_id, "evaluation")
    if spec["source"] == "local":
        raise ValueError("use make_synthetic_batches for the synthetic dataset")
    rows = corpus_rows(spec, cache_dir=cache_dir)
    if spec.get("streaming", False):
        ids = _stream_ids(tokenizer, rows)
    else:
        ids = encode_text(tokenizer, ROW_SEPARATOR.join(rows))
    windows = islice(_windows(ids, seq_len), spec.get("evaluation_windows"))
    yield from token_batches(windows, batch_size)
