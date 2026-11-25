from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor

_REGISTRY = Path(__file__).with_name("registry.yaml")
ROW_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor

    def __post_init__(self):
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input_ids and attention_mask must have the same [B,T] shape")
        if self.input_ids.dtype != torch.int64 or self.attention_mask.dtype != torch.bool:
            raise TypeError("expected int64 input_ids and a bool attention_mask")
        if self.input_ids.device != self.attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device")

    @classmethod
    def unpadded(cls, input_ids: Tensor) -> "TokenBatch":
        return cls(input_ids, torch.ones_like(input_ids, dtype=torch.bool))

    def to(self, device: str | torch.device) -> "TokenBatch":
        return TokenBatch(self.input_ids.to(device), self.attention_mask.to(device))


def corpus_spec(dataset_id: str, role: str) -> dict:
    with _REGISTRY.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)[dataset_id]
    if role not in spec["splits"]:
        raise ValueError(f"{dataset_id} has no {role} split")
    return {**spec, "split": spec["splits"][role]}


def corpus_rows(spec: dict, *, cache_dir: str | Path | None = None) -> Iterator[str]:
    from datasets import load_dataset

    split = spec["split"]
    source = spec["source"]
    if spec.get("format") == "text":
        url = source.format(revision=spec["revision"], split=split)
        ds = load_dataset("text", data_files={split: url}, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(
            source, spec["config"], split=split, revision=spec["revision"],
            streaming=spec.get("streaming", False), cache_dir=cache_dir,
        )
    for row in ds:
        text = row[spec["text_field"]]
        if text.strip():
            yield text


def encode_text(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False, verbose=False)


def token_batches(windows: Iterable[Sequence[int]], batch_size: int) -> Iterator[TokenBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            yield TokenBatch.unpadded(torch.tensor(batch, dtype=torch.long, device="cpu"))
            batch.clear()
    if batch:
        yield TokenBatch.unpadded(torch.tensor(batch, dtype=torch.long, device="cpu"))
