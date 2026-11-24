import json
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from Dataloaders.causal_lm import (
    ROW_SEPARATOR,
    TokenBatch,
    corpus_rows,
    corpus_spec,
    encode_text,
    token_batches,
)


@dataclass(frozen=True)
class CalibrationManifest:
    dataset_id: str
    revision: str
    split: str
    tokenizer_id: str
    tokenizer_revision: str | None
    seed: int
    seq_len: int
    starts: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationSet:
    manifest: CalibrationManifest
    windows: tuple[tuple[int, ...], ...]
    batch_size: int = 1

    def __iter__(self) -> Iterator[TokenBatch]:
        return token_batches(self.windows, self.batch_size)

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(
                {"manifest": asdict(self.manifest), "windows": self.windows, "batch_size": self.batch_size},
                f, separators=(",", ":"),
            )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationSet":
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        meta = data["manifest"]
        meta["starts"] = tuple(meta["starts"])
        return cls(
            CalibrationManifest(**meta), tuple(tuple(w) for w in data["windows"]), data["batch_size"]
        )


def make_calibration_batches(
    dataset_id: str,
    tokenizer,
    *,
    seed: int = 0,
    count: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
    tokenizer_revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> CalibrationSet:
    if min(count, seq_len, batch_size) < 1:
        raise ValueError("count, seq_len and batch_size must be positive")
    spec = corpus_spec(dataset_id, "calibration")
    if spec["source"] == "local":
        raise ValueError("use make_synthetic_batches for the synthetic dataset")
    ids = encode_text(tokenizer, ROW_SEPARATOR.join(corpus_rows(spec, cache_dir=cache_dir)))
    num_starts = len(ids) - seq_len + 1
    if num_starts < count:
        raise ValueError(f"{dataset_id} has only {max(0, num_starts)} distinct window starts")
    starts = tuple(random.Random(seed).sample(range(num_starts), count))
    manifest = CalibrationManifest(
        dataset_id, spec["revision"], spec["split"], str(tokenizer.name_or_path),
        tokenizer_revision, seed, seq_len, starts,
    )
    windows = tuple(tuple(ids[i:i + seq_len]) for i in starts)
    return CalibrationSet(manifest, windows, batch_size)
