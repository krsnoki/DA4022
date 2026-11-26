"""GPU smoke test for the XShare integration (about two minutes).

Checks, on a handful of GSM8K questions:

1. The registered backbone loads and generates.
2. XShare with every expert kept produces exactly the same tokens as the
   unpruned baseline, so the router replacement matches the model's router.
3. Pruning configurations run, keep fewer experts, and report timings.

    python -m Mixture_of_Experts.experiments.xshare.smoke_test --batch-size 8

Prints SMOKE_OK when check 2 passes. The timings use only a few short
generations; use ``run_sweep.py`` for measurements.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from Backbones.Generative_Pretrained.hf_causal_lm import HFCausalLMBackbone
from Dataloaders.evaluation import gsm8k
from Mixture_of_Experts.execution_layouts.xshare_hf import install_xshare
from Mixture_of_Experts.experiments.xshare.run_sweep import generate_batch
from Mixture_of_Experts.routing.xshare import XShareConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="text.olmoe-1b-7b-0125-instruct")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()

    backbone = HFCausalLMBackbone.load(args.backbone, device="cuda")
    meta = backbone.get_model_metadata()
    n_experts, top_k = meta["n_experts"], meta["top_k"]
    gate_norm = "softmax_topk_renorm" if meta["norm_topk_prob"] else "softmax_topk"
    print(f"loaded in {meta['load_time_s']:.1f} s: {n_experts} experts, top-{top_k}, gate_norm={gate_norm}")
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    controller = install_xshare(backbone.model, None)
    eos_ids = backbone.eos_token_ids()
    examples = gsm8k.load_gsm8k_subset(args.batch_size, seed=0)
    prompts = backbone.preprocess([gsm8k.build_prompt(e["question"]) for e in examples])

    def run(label, config):
        controller.config = config
        generate_batch(backbone, controller, prompts, 8, eos_ids)  # warm-up
        controller.reset_stats()
        out = generate_batch(backbone, controller, prompts, args.max_new_tokens, eos_ids)
        stats = controller.stats()
        print(
            f"{label:>22}: {out['live_tokens'] / out['decode_s']:7.1f} tok/s, "
            f"step median {statistics.median(out['step_ms']):6.2f} ms, "
            f"kept {stats['mean_kept_experts']:5.1f}, "
            f"activated {stats['mean_activated_experts']:5.1f} of {n_experts}"
        )
        return out

    def cfg(k0, budget):
        return XShareConfig(top_k=top_k, warmup_k0=k0, budget=budget, gate_norm=gate_norm)

    base = run("baseline", None)
    full = run("xshare, keep all", cfg(0, n_experts))
    same = base["texts"] == full["texts"]
    print(f"keep-all output identical to baseline: {same}")
    run("OEA k0=1 (b=0)", cfg(1, 0))
    run("xshare k0=1, b=8", cfg(1, 8))
    print("--- first baseline generation ---")
    print(base["texts"][0][:400])
    print(f"gold answer: {examples[0]['answer']}")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("SMOKE_OK" if same else "SMOKE_MISMATCH")


if __name__ == "__main__":
    main()
