"""SigLIP zero-shot helpers shared by train_siglip.py and eval_quant.py."""

from __future__ import annotations

import random
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F

from Dataloaders.evaluation import imagenet_zeroshot as inet


@torch.no_grad()
def build_class_embeddings(model, processor, dcfg: dict, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """L2-normalised text embeddings ``[num_classes, dim]`` of the class prompts (FP text tower)."""
    model.eval()
    if dcfg["dataset"] == "synthetic.images":
        g = torch.Generator().manual_seed(1234)
        C = int(dcfg.get("num_classes", 4))
        L = model.config.text_config.max_position_embeddings
        ids = torch.randint(0, model.config.text_config.vocab_size, (C, L), generator=g)
        feats = model.get_text_features(input_ids=ids.to(device))
    else:
        names = inet.class_names()
        prompts = [dcfg.get("prompt", inet.PROMPT).format(n) for n in names]
        out = []
        for i in range(0, len(prompts), batch_size):
            t = processor(text=prompts[i : i + batch_size], padding="max_length", max_length=64, return_tensors="pt")
            out.append(model.get_text_features(input_ids=t["input_ids"].to(device)))
        feats = torch.cat(out)
    return F.normalize(feats.float(), dim=-1)


def zero_shot_logits(model, pixel_values: torch.Tensor, class_emb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    img = model.get_image_features(pixel_values=pixel_values)
    img = F.normalize(img.float(), dim=-1)
    return scale.float() * img @ class_emb.t()


def image_batches(
    processor, dcfg: dict, split: str, start: int, count: int, batch_size: int, seed: int = 0, epoch: int = 0
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Batches of ``(pixel_values, labels)`` for the images ``[start, start+count)`` of a seeded stream.

    The *set* of images depends only on ``seed`` (so calibration/train/eval stay
    disjoint); ``epoch`` only reshuffles the order inside chunks of 1024 images.
    """
    if dcfg["dataset"] == "synthetic.images":
        size = int(dcfg.get("image_size", 32))
        offset = {"train": 0, "validation": 10_000, "calibration": 20_000}.get(split, 30_000)
        n_batches = max(1, count // batch_size)
        yield from inet.synthetic_image_batches(n_batches, batch_size, size, int(dcfg.get("num_classes", 4)), seed + offset + start)
        return
    ip = processor.image_processor if hasattr(processor, "image_processor") else processor
    if epoch == 0:
        yield from inet.batched_pixels(ip, split, start, count, batch_size, seed)
        return
    rng = random.Random(seed * 7919 + epoch)
    chunk = []

    def flush(items):
        rng.shuffle(items)
        for i in range(0, len(items), batch_size):
            part = items[i : i + batch_size]
            yield ip(images=[p[0] for p in part], return_tensors="pt")["pixel_values"], torch.tensor([p[1] for p in part])

    for item in inet.iter_images(split, start, count, seed):
        chunk.append(item)
        if len(chunk) == 1024:
            yield from flush(chunk)
            chunk = []
    if chunk:
        yield from flush(chunk)
