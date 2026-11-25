"""Offline test of the C4 sampler: the Hub is replaced by a tiny in-memory streaming dataset."""

import random

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")

import Dataloaders.calibration.aaac_c4 as c4


class WordTokenizer:
    """One whitespace-separated word = one token (stable ids, unlike built-in hash())."""

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": [sum(map(ord, w)) % 997 for w in text.split()]}


def _fake_documents():
    rng = random.Random(1)
    for i in range(1500):
        n = rng.choice([40, 300, 2500])
        yield {"text": " ".join(f"w{(i * 7 + j) % 991}" for j in range(n))}


@pytest.fixture
def fake_hub(monkeypatch):
    seen = {}

    def fake_load_dataset(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return datasets.IterableDataset.from_generator(_fake_documents)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return seen


def _windows(seed, n=4):
    return list(
        c4.iter_c4_calibration_batches(
            WordTokenizer(), num_sequences=n, sequence_length=2048, seed=seed, shuffle_buffer_size=200
        )
    )


def test_windows_have_fixed_shape_no_padding_and_pinned_source(fake_hub):
    batches = _windows(42)
    assert len(batches) == 4
    for b in batches:
        assert b["input_ids"].shape == (1, 2048) and b["input_ids"].dtype == torch.long
        assert bool(b["attention_mask"].all())
    assert fake_hub["args"] == (c4.C4_DATASET_ID, c4.C4_CONFIG)
    assert fake_hub["kwargs"]["revision"] == c4.C4_REVISION
    assert fake_hub["kwargs"]["split"] == "train" and fake_hub["kwargs"]["streaming"] is True


def test_same_seed_reproduces_and_different_seed_differs(fake_hub):
    a, b, other = _windows(42), _windows(42), _windows(7)
    assert all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, b))
    assert not all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, other))


def test_running_out_of_long_documents_raises(fake_hub):
    with pytest.raises(RuntimeError, match="usable windows"):
        _windows(42, n=10_000)


def test_rejects_non_positive_arguments():
    with pytest.raises(ValueError):
        list(c4.iter_c4_calibration_batches(WordTokenizer(), num_sequences=0))
