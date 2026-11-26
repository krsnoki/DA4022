"""Time-budgeted diffu-GRPO post-training with TCT / CT / AR rollouts (Sec. 4.3, Fig. 5).

    python Adaptive_Stopping/experiments/tempered_remasking/run_grpo.py \
        --dataset gsm8k --rollout tct --hours 72 --out runs/grpo_gsm8k_tct

Requires ``peft`` for LoRA (d1 trains LoRA adapters on LLaDA-8B-Instruct; see
``LoRA/`` for the adapter conventions). Logs one JSON line per training step
(reward mean, within-group reward std, rollout NFEs, wall time) to
``<out>/train_log.jsonl`` and saves the adapter to ``<out>/adapter``.
Evaluate afterwards with ``run_sampling.py --model-path ... --preset ct
--token-temperature 0 --threshold {0.6,0.8,1.0} --n-samples 1``.

Note: the paper used 4xA100 per run; this single-process script is the
algorithmic reference, not a multi-GPU launcher.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random

import torch

from common import environment_metadata, grade, load_config, load_problems, policy_from_preset, sampler_config
from tempered_remasking.grpo import DiffuGRPOConfig, DiffuGRPOTrainer
from tempered_remasking.models import load_hf_dllm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    ap.add_argument("--train-split", default="train", help="training split (never the evaluation split)")
    ap.add_argument("--rollout", choices=["tct", "ct", "ar"], required=True)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    g = cfg["grpo"]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    handle = load_hf_dllm(g["model"], cfg["models"][g["model"]]["model_id"], cfg["models"][g["model"]].get("revision"),
                          args.device, torch.bfloat16)
    policy_model = handle.model_fn
    reference = None
    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(r=g["lora"]["r"], lora_alpha=g["lora"]["alpha"], lora_dropout=g["lora"].get("dropout", 0.05),
                          target_modules=g["lora"]["target_modules"], bias="none")
        policy_model.model = get_peft_model(policy_model.model, lcfg)

        class _Ref(torch.nn.Module):
            """Reference policy = base weights with adapters disabled (no second 8B copy).

            The policy is held outside the module tree so that the trainer's
            ``reference.eval()`` / ``requires_grad_(False)`` cannot touch it.
            """

            def __init__(self, adapter):
                super().__init__()
                self._adapter = [adapter]

            def forward(self, x):
                adapter = self._adapter[0]
                with adapter.model.disable_adapter():
                    return adapter(x)

        reference = _Ref(policy_model)

    # Training data: the *train* split (the paper evaluates on the test split).
    data_cfg = dict(cfg)
    data_cfg["datasets"] = {k: dict(v) for k, v in cfg["datasets"].items()}
    data_cfg["datasets"][args.dataset]["split"] = args.train_split
    data_cfg["datasets"][args.dataset]["limit"] = None
    problems = load_problems(data_cfg, args.dataset)
    random.shuffle(problems)
    examples = ({**p, "prompt_ids": handle.encode_chat(p["prompt"], torch.device(args.device))} for p in itertools.cycle(problems))

    def reward_fn(tokens, ex):
        # Correctness reward (d1 additionally uses format rewards; add them here to match d1 exactly).
        _, ok = grade(args.dataset, handle.decode(tokens), ex)
        return 2.0 if ok else 0.0

    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=g["learning_rate"])
    gcfg = DiffuGRPOConfig(
        num_generations=g["num_generations"], p_mask_prompt=g["p_mask_prompt"], num_iterations=g["num_iterations"],
        clip_eps=g["clip_eps"], beta=g["beta"],
        time_budget_s=3600 * (args.hours if args.hours is not None else g["time_budget_hours"]), max_steps=args.max_steps,
    )
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")

    def log(rec):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)

    trainer = DiffuGRPOTrainer(
        policy_model, reference, opt, policy_from_preset(cfg, args.rollout), sampler_config(cfg),
        reward_fn, handle.mask_token_id, handle.eos_token_ids, gcfg,
        generator=torch.Generator(device=args.device).manual_seed(args.seed), log_fn=log,
    )
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"env": environment_metadata(), **trainer.config_dict(), "dataset": args.dataset, "seed": args.seed}, f, indent=2)
    hist = trainer.train(examples)
    print(f"completed {len(hist)} steps")
    if not args.no_lora:
        policy_model.model.save_pretrained(os.path.join(args.out, "adapter"))


if __name__ == "__main__":
    main()
