# Mixture-of-Experts (MoE), Routing and Elastic Capacity

## Purpose

Implement sparse expert models and resource-aware expert computation: expert selection/routing, expert pruning, load balancing, expert sharing, elastic capacity, and MoE-specific quantization or execution. Keep the base architecture in `Backbones/`; keep system measurements in `Performance_Metrics/`.

## Suggested structure

```text
Mixture_of_Experts/
├── README_moe.md
├── routing/
├── expert_pruning/
├── elastic_capacity/
├── load_balancing/
├── execution_layouts/
├── configs/
└── tests/
```

## Paper shortlist

- [A Recipe for an Elastic Mixture: One Mixture-of-Experts for Every Resource Budget](https://openreview.net/forum?id=0stDFvtMGN)
- [TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](https://openreview.net/forum?id=Zfgv1Sa7Ot)
- [LExI: Layer-Adaptive Active Experts for Efficient MoE Inference](https://openreview.net/forum?id=oYiauQToKN)
- [DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices](https://openreview.net/forum?id=YYycBS8yxj)
- [Selective Sinkhorn Routing for Improved Sparse Mixture of Experts](https://openreview.net/forum?id=qRQU6W1vJ4)
- [XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference](https://openreview.net/forum?id=KmY5XUgEYe)
- [GreenMoE: Exploiting Dynamic Load Imbalance for Energy-Efficient Long-Context MoE Training](https://openreview.net/forum?id=ShvjLjCOGB)
- [Layout and Fusion Trade-offs for Mixture-of-Experts Inference under Single-Node Tensor Parallelism](https://openreview.net/forum?id=y2sY80Q3fh)
- [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://openreview.net/forum?id=UZ5V7ub8mi)
- [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://openreview.net/forum?id=ZqRh99OQEK)
- [SRA-MoE: Output-Aware Selective Router Alignment for MoE Quantization](https://openreview.net/forum?id=H0NoX02erJ)

The official [accepted-paper list](https://adaptfm.gitlab.io/accepted-papers) is the source of truth. Some papers cross-list with quantization, pruning, or runtime systems.

## Shared interfaces and requirements

Document expert count, top-k routing, capacity factor, auxiliary routing losses, expert placement, routing dtype, distributed strategy, and any assumptions about batch/sequence shape. The output should include routing diagnostics where feasible: token-to-expert assignments, dropped tokens, per-expert load, and number of active experts.

Benchmarks must report quality, end-to-end latency, throughput, peak memory, active/total experts, routing overhead, and communication costs for distributed variants. Do not equate fewer active parameters with faster wall-clock execution without measuring it on supported hardware.
