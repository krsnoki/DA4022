"""Fine-tune a causal LM with (or without) Jacobian-guided noise injection.

    python Quantization_and_Pruning/experiments/jacobian_noise/train_lm.py \
        --config Quantization_and_Pruning/configs/jacobian_noise/qwen2_5_0_5b.yaml \
        --set noise.mode=jacobian experiment.seed=0

Writes to ``<output_dir>/<run_name>/``: ``checkpoint/`` (HF format, attention
implementation restored to eager), ``noise_state.pt``, ``train_log.jsonl``,
``run.json`` (resolved config + environment + timing).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    cycle, env_metadata, load_config, pick_device, set_seed, train_loop, write_json,
)

import torch  # noqa: E402

from Backbones.registry import load_backbone  # noqa: E402
from Dataloaders.supervised.alpaca import CausalLMCollator, load_alpaca, synthetic_lm_examples  # noqa: E402
from Quantization_and_Pruning.quantization.qat.jacobian_noise import (  # noqa: E402
    JacobianNoiseConfig, attach_jacobian_noise, detach_jacobian_noise,
)


def run_name(cfg: dict) -> str:
    n = cfg["noise"]
    tag = n["mode"] if n["mode"] in ("none", "uniform_matched") else f"{n['mode']}_a{n.get('alpha', 0.5)}"
    return f"{cfg['experiment']['name']}__{tag}__s{cfg['experiment']['seed']}"


def build_lora(model, lcfg: dict):
    from peft import LoraConfig, get_peft_model

    pc = LoraConfig(
        r=lcfg.get("r", 16), lora_alpha=lcfg.get("alpha", 32), lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=lcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]), task_type="CAUSAL_LM",
    )
    return get_peft_model(model, pc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.max_steps=200")
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

    model, tok, meta = load_backbone(bcfg["id"], dtype=torch.float32, attn_implementation="eager")
    model.to(device)
    if tcfg.get("gradient_checkpointing"):
        if ncfg.get("seed") is not None:
            raise ValueError("noise.seed uses a private RNG that checkpoint recomputation cannot replay; unset it")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    # ---- data
    if dcfg["dataset"] == "synthetic.lm":
        vocab = model.config.vocab_size
        train = synthetic_lm_examples(dcfg.get("num_train", 64), dcfg["max_length"], vocab, seed)
        val = synthetic_lm_examples(dcfg.get("num_val", 8), dcfg["max_length"], vocab, seed + 1)
        pad_id = 0
    elif dcfg["dataset"] == "text.alpaca.train":
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        train, val = load_alpaca(
            tok, dcfg["max_length"], dcfg.get("num_train"), dcfg.get("num_val", 500), dcfg.get("seed", 42),
            dcfg.get("train_on_prompt", False),
        )
        pad_id = tok.pad_token_id
    else:
        raise ValueError(f"unsupported dataset {dcfg['dataset']}")
    collate = CausalLMCollator(pad_id)
    bs = int(tcfg.get("batch_size", 1))

    def epoch_iter(epoch):
        order = list(range(len(train)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        for i in range(0, len(order) - bs + 1, bs):
            yield {k: v.to(device) for k, v in collate([train[j] for j in order[i : i + bs]]).items()}

    # ---- method
    if tcfg.get("lora"):
        model = build_lora(model, tcfg["lora"])
    ctrl = attach_jacobian_noise(model, JacobianNoiseConfig.from_dict(ncfg))

    def loss_fn(batch):
        return model(**batch, use_cache=False).loss

    @torch.no_grad()
    def evaluate():
        tot, n = 0.0, 0
        for i in range(0, min(len(val), int(tcfg.get("eval_examples", 200))), bs):
            b = {k: v.to(device) for k, v in collate(val[i : i + bs]).items()}
            tot += float(model(**b, use_cache=False).loss)
            n += 1
        return {"val_loss": tot / max(1, n)}

    print(f"[train_lm] {run_name(cfg)}: {len(train)} train / {len(val)} val examples on {device}")
    stats = train_loop(
        model, model.parameters(), cycle(epoch_iter), loss_fn, tcfg, device, ctrl=ctrl,
        on_micro_batch=lambda b: ctrl.set_query_mask(b["attention_mask"]), evaluate=evaluate, log_path=log_path,
    )
    model.eval()
    final_val = evaluate()

    # ---- save
    torch.save(ctrl.state_dict(), os.path.join(out_dir, "noise_state.pt"))
    detach_jacobian_noise(model, "eager")
    if tcfg.get("lora"):
        model = model.merge_and_unload()
    if tcfg.get("save", True):
        ckpt = os.path.join(out_dir, "checkpoint")
        model.save_pretrained(ckpt, safe_serialization=True)
        if tok is not None:
            tok.save_pretrained(ckpt)
    write_json(os.path.join(out_dir, "run.json"), {
        "run_name": run_name(cfg), "config": cfg, "backbone": meta, "env": env_metadata(device),
        "num_train_examples": len(train), "final": final_val, "noise_summary": ctrl.summary(),
        "noise_history": ctrl.history, **{k: v for k, v in stats.items() if k != "history"},
    })
    print(f"[train_lm] done -> {out_dir}  final {final_val}")
    return out_dir


if __name__ == "__main__":
    main()
