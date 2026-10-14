# Attention, Context and Token Efficiency

## Purpose

Implement methods that reduce the cost of attention or the context processed by foundation models. This module covers sparse/linear/low-rank attention, conditional memory access, context compression, and token selection/merging. KV-cache policies may live here initially, but use `PAPER_INDEX.md` to cross-reference the paper and avoid duplicating a shared KV-cache utility if a dedicated module is later created.

## Suggested structure

```text
Attention/
├── README_attention.md
├── kernels/
├── methods/
│   ├── sparse/
│   ├── linear/
│   ├── low_rank/
│   ├── context_compression/
│   └── token_selection/
├── configs/
├── tests/
└── experiments/
```

## Paper shortlist

- [A3: an Analytical Low-Rank Approximation Framework for Attention](https://openreview.net/forum?id=ByrYoK4vWy) — low-rank attention.
- [NOSA: Native and Offloadable Sparse Attention](https://openreview.net/forum?id=rxozYQe8xn) — sparse attention and offload.
- [WildCat: Near-Linear Attention in Theory and Practice](https://openreview.net/forum?id=LqWW1SrKcH) — near-linear attention.
- [Learning When to Attend: Conditional Memory Access for Long-Context LLMs](https://openreview.net/forum?id=8ubnKeIh0Y) — conditional access.
- [Reducing Attention Distribution Error with Unified Tail Aggregation for Sparse Attention](https://openreview.net/forum?id=mEpi4pgzpI) — sparse attention approximation.
- [LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning](https://openreview.net/forum?id=oqacjavnry) — context compression.
- [Decoupling Spatial and Semantic Token Compression for Vision-Language Model Acceleration](https://openreview.net/forum?id=4ylINNNS6y) — multimodal token compression.
- [QJL is 1-bit Compressive Sensing: An Equivalence and Its Consequences for KV Cache Compression in LLMs](https://openreview.net/forum?id=B0N2TDGiOl) — compressed KV state.
- [StreamAttention: Energy-Efficient and High-Utilization Attention on Systolic Hardware](https://openreview.net/forum?id=SRQuPOMzkX) — hardware mapping.
- [Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts](https://openreview.net/forum?id=llfDaQar98) — hybrid linear attention.

See also the [full module-to-paper index](../PAPER_INDEX.md).

## Inputs and outputs

At minimum, document: tensor layout and shape, head dimensions, mask semantics, causal/non-causal support, dtype support, output tolerance, cache behavior, and any constraints on sequence length. A kernel that only supports one fixed shape must declare that limitation in its config and tests.

## Required tests and benchmarks

- Numerical comparison against a trusted reference implementation with explicit absolute/relative tolerances.
- Correctness for causal masks, padding, short sequences, and representative long sequences.
- Warm-up and synchronization before timing asynchronous GPU work.
- Separate prefill and decode measurements for autoregressive language models.
- Peak memory, throughput, batch size, sequence length, dtype, hardware, and kernel version.
- For token compression/pruning, include downstream task quality and a no-compression baseline.

## Do not claim a speedup from FLOPs alone

The method must be benchmarked on actual supported hardware. Sparse operations can be slower than dense kernels at small batch sizes or unsupported sparsity patterns; write down where a method helps and where it does not.
