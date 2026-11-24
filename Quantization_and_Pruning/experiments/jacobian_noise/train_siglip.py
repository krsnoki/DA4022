"""Fine-tune SigLIP's image tower with (or without) Jacobian-guided noise.

The paper does not state the SigLIP training objective.  We use a zero-shot-head
cross-entropy (documented deviation): the text tower is frozen, the 1000
ImageNet class prompts ("a photo of a {class}.") are embedded once, and the
image tower is trained with
``CE(logit_scale * normalize(img_emb) @ normalize(class_emb).T, label)``.
Noise is injected only into the image tower's attention (``include="vision_model"``).

    python Quantization_and_Pruning/experiments/jacobian_noise/train_siglip.py \
        --config Quantization_and_Pruning/configs/jacobian_noise/siglip_base_224.yaml --set noise.mode=jacobian
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    cycle, env_metadata, load_config, pick_device, set_seed, train_loop, write_json,
)
from siglip_utils import build_class_embeddings, image_batches, zero_shot_logits  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from Backbones.registry import load_backbone  # noqa: E402
from Quantization_and_Pruning.quantization.qat.jacobian_noise import (  # noqa: E402
    JacobianNoiseConfig, attach_jacobian_noise, detach_jacobian_noise,
)


def run_name(cfg: dict) -> str:
    n = cfg["noise"]
    tag = n["mode"] if n["mode"] in ("none", "uniform_matched") else f"{n['mode']}_a{n.get('alpha', 0.5)}"
    return f"{cfg['experiment']['name']}__{tag}__s{cfg['experiment']['seed']}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    ecfg, bcfg, dcfg, tcfg, ncfg = cfg["experiment"], cfg["backbone"], cfg["data"], cfg["train"], cfg["noise"]

    seed = int(ecfg["seed"])
    set_seed(seed)
    device = pick_device(ecfg.get("device", "auto"))
    out_dir = os.path.join(ecfg["output_dir"], run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    model, processor, meta = load_backbone(bcfg["id"], dtype=torch.float32, attn_implementation="eager")
    model.to(device)
    class_emb = build_class_embeddings(model, processor, dcfg, device)  # [C, D], frozen
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vision_model.parameters():
        p.requires_grad_(True)
    if tcfg.get("gradient_checkpointing"):
        if ncfg.get("seed") is not None:
            raise ValueError("noise.seed uses a private RNG that checkpoint recomputation cannot replay; unset it")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ctrl = attach_jacobian_noise(model, JacobianNoiseConfig.from_dict(ncfg), include="vision_model")
    scale = model.logit_scale.exp().detach()

    def loss_fn(batch):
        x, y = batch
        return F.cross_entropy(zero_shot_logits(model, x, class_emb, scale), y)

    bs = int(tcfg.get("batch_size", 32))
    n_train = int(dcfg.get("num_train", 50_000))
    n_calib = int(dcfg.get("num_calib", 256))

    def epoch_iter(epoch):
        # calibration images [0, n_calib) are reserved for PTQ; training uses the next n_train
        for x, y in image_batches(processor, dcfg, "train", n_calib, n_train, bs, seed=dcfg.get("seed", 0), epoch=epoch):
            yield x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate():
        correct = total = 0
        for x, y in image_batches(processor, dcfg, "validation", 0, int(tcfg.get("eval_examples", 1000)), bs, seed=dcfg.get("seed", 0)):
            pred = zero_shot_logits(model, x.to(device), class_emb, scale).argmax(-1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
        return {"val_top1": correct / max(1, total), "val_examples": total}

    print(f"[train_siglip] {run_name(cfg)} on {device}; {n_train} train images, {class_emb.shape[0]} classes")
    stats = train_loop(
        model, model.vision_model.parameters(), cycle(epoch_iter), loss_fn, tcfg, device, ctrl=ctrl,
        evaluate=evaluate, log_path=log_path,
    )
    model.eval()
    final = evaluate()

    torch.save(ctrl.state_dict(), os.path.join(out_dir, "noise_state.pt"))
    detach_jacobian_noise(model, "eager")
    if tcfg.get("save", True):
        ckpt = os.path.join(out_dir, "checkpoint")
        model.save_pretrained(ckpt, safe_serialization=True)
        if processor is not None:
            processor.save_pretrained(ckpt)
    write_json(os.path.join(out_dir, "run.json"), {
        "run_name": run_name(cfg), "config": cfg, "backbone": meta, "env": env_metadata(device),
        "final": final, "noise_summary": ctrl.summary(), "noise_history": ctrl.history,
        **{k: v for k, v in stats.items() if k != "history"},
    })
    print(f"[train_siglip] done -> {out_dir}  final {final}")
    return out_dir


if __name__ == "__main__":
    main()
