# Speculative Decoding and Serving

## Purpose

Methods that reduce per-token decode latency without changing the target model's output distribution. A cheap **drafter** proposes several future tokens and the **target** model checks them in one forward pass. The methods differ in the drafter (a small AR model, self-drafting heads, or a block-diffusion drafter), in the shape of the draft (a single chain or a tree) and in how the verification budget is chosen.

This module holds the speculative-decoding papers listed in [`PAPER_INDEX.md`](../PAPER_INDEX.md#speculative-decoding-and-serving). Target models and their tokenizers stay in [`Backbones/Generative_Pretrained/`](../Backbones/Generative_Pretrained/README.md). Benchmark prompt sets are documented in [`Dataloaders/`](../Dataloaders/README_dataloader.md).

## Implementations

| Paper | Folder | Status |
|---|---|---|
| [BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting](https://openreview.net/forum?id=uqeOxztSIS) | [`BASTION/`](BASTION/) | `partial-reproduction` (Qwen3-4B/8B, T=0, one RTX PRO 4500). The qualitative claims reproduce; the speedup magnitudes do not (BASTION 4.50× / 4.47× vs 6.93× / 6.61× on A100). See [`BASTION/PAPER_CARD.md`](BASTION/PAPER_CARD.md) and [`BASTION/REPORT.md`](BASTION/REPORT.md) |

## Other papers in scope

- [ShadowSpec: Towards Zero Speculation Overhead for Substitute Speculative Decoding](https://openreview.net/forum?id=ss50E1FoBU)
- [Fast Inference via Hierarchical Speculative Decoding](https://openreview.net/forum?id=i2IDDWcoEw)
- [Layer Verification Accelerates Speculative Tree Decoding](https://openreview.net/forum?id=XrqcUo6B6L)
- [PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding](https://openreview.net/forum?id=VKD70LJaDy)
- [Characterizing self-speculative decoding approaches for accelerating LLMs](https://openreview.net/forum?id=uXQyCxN7r9)
- [EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts](https://openreview.net/forum?id=zEJ6juNLZi)

## Measurement rules specific to speculative decoding

Follow [`Performance_Metrics/README_metrics.md`](../Performance_Metrics/README_metrics.md) and also:

- **The baseline is target-only AR decoding.** Use the same target checkpoint, prompts, chat template, `max_new_tokens`, temperature, batch size and GPU. Report speedup as a ratio of per-token latencies measured on the same inputs.
- **Report average accepted length (AAL, τ)** next to speedup. AAL does not depend on the hardware, while speedup does.
- **Check losslessness.** At temperature 0 the speculative output should equal the AR output token for token. Report the exact-match rate rather than assuming it. bf16 numerics can cause divergences, and they need not be rare: in the BASTION reproduction only 30–35% of bf16 outputs matched AR, while fp32 matched exactly on the prompts tested.
- **Separate prefill from decode**, and state whether one-off costs (calibration, profiling, warm-up) are excluded from the timings.
- **Name the attention backend** (SDPA, FlashAttention, etc.) for each mode, because tree verification usually needs a custom mask that rules out some kernels.
