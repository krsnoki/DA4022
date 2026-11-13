"""ImageNet-1K for SigLIP zero-shot classification.

Dataset ids (see Dataloaders/registry.yaml):

* ``vision.imagenet1k_256.train``      fine-tuning subset + calibration subset
* ``vision.imagenet1k_256.validation`` evaluation subset

Source: ``benjamin-paine/imagenet-1k-256x256`` (public parquet copy of ILSVRC-2012
resized to 256 px, labels = original 1000 ImageNet classes).  ImageNet's own terms
(non-commercial research) apply.  Data is *streamed*, so nothing is stored in the
repo and only the requested number of images is downloaded.

Disjointness: calibration images are the first ``num_calib`` of the seeded
train stream, training images are the following ``num_train``.  Evaluation uses
the validation split only.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterator, List, Optional, Tuple

import torch

HF_ID = "benjamin-paine/imagenet-1k-256x256"
REVISION = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
PROMPT = "a photo of a {}."


def class_names(revision: str = REVISION) -> List[str]:
    """First synonym of each ImageNet class ("tench, Tinca tinca" -> "tench")."""
    from datasets import load_dataset_builder

    names = load_dataset_builder(HF_ID, revision=revision).info.features["label"].names
    return [n.split(",")[0].strip() for n in names]


def stream(split: str, seed: int = 0, revision: str = REVISION, shuffle_buffer: int = 10_000):
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split=split, streaming=True, revision=revision)
    return ds.shuffle(seed=seed, buffer_size=shuffle_buffer)


def iter_images(split: str, start: int, count: int, seed: int = 0, revision: str = REVISION) -> Iterator[Tuple[object, int]]:
    for ex in islice(stream(split, seed, revision), start, start + count):
        yield ex["image"].convert("RGB"), int(ex["label"])


def batched_pixels(
    image_processor, split: str, start: int, count: int, batch_size: int, seed: int = 0
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(pixel_values [B,3,H,W], labels [B])`` batches."""
    imgs, labels = [], []
    for img, lab in iter_images(split, start, count, seed):
        imgs.append(img)
        labels.append(lab)
        if len(imgs) == batch_size:
            yield image_processor(images=imgs, return_tensors="pt")["pixel_values"], torch.tensor(labels)
            imgs, labels = [], []
    if imgs:
        yield image_processor(images=imgs, return_tensors="pt")["pixel_values"], torch.tensor(labels)


def synthetic_image_batches(
    num_batches: int, batch_size: int, image_size: int, num_classes: int, seed: int = 0
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Random images whose class is encoded in the mean colour (smoke tests only)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(num_batches):
        y = torch.randint(0, num_classes, (batch_size,), generator=g)
        x = torch.randn(batch_size, 3, image_size, image_size, generator=g) * 0.5
        x += (y.float() / max(1, num_classes - 1) * 2 - 1).view(-1, 1, 1, 1)
        out.append((x, y))
    return out
