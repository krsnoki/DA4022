# AdaptFM paper index by implementation module

This is an engineering-oriented shortlist from the [official AdaptFM accepted-paper list](https://adaptfm.gitlab.io/accepted-papers). It is **not** a claim that every listed paper provides source code, pretrained weights, or a complete reproduction recipe. Links below point to the paper's OpenReview forum page where verified from the official list. Papers are cross-listed if a single technique touches several modules.

## Adaptive stopping and adaptive compute

- [TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning](https://openreview.net/forum?id=GwTFovD6SM) — learned exit/stop decisions.
- [Step-Tagging Early-Stopping: Toward controlling the generation of Language Reasoning Models through black-box step monitoring](https://openreview.net/forum?id=bNn5OusO22) — black-box stop signal.
- [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://openreview.net/forum?id=xExWu1Bhuw) — stopping in looped networks.
- [Learning Adaptive Reasoning Budgets via Constraint-Rectified Training](https://openreview.net/forum?id=1inovdFo6Z) — budget policy/training.
- [DIPA: Difficulty-Informed Probabilistic Allocation of Test-Time Compute via Training-Free Proxies](https://openreview.net/forum?id=9Joja0tx5c) — input-dependent compute allocation.
- [Block-Level Recursion: Adaptive Test-Time Routing in Large Language Models](https://openreview.net/forum?id=hLFRcxXKxY) — dynamic depth/recursion.
- [Fixed-Point Reasoners: Stable and Adaptive Deep Looped Models](https://openreview.net/forum?id=s66nL9PEw1) — adaptive depth/looped computation, not necessarily early exit.
- [Learning Adaptive LLM Decoding](https://openreview.net/forum?id=14yhYy9m7S) — adaptive decoding policy.
- [Resource-Adaptive Foundation Model Reasoning via Semantic Coverage](https://openreview.net/forum?id=suD1tqogTd) — input-adaptive reasoning budget.

## Attention, context and token efficiency

- [A3: an Analytical Low-Rank Approximation Framework for Attention](https://openreview.net/forum?id=ByrYoK4vWy) — low-rank attention.
- [NOSA: Native and Offloadable Sparse Attention](https://openreview.net/forum?id=rxozYQe8xn) — sparse attention and offload.
- [WildCat: Near-Linear Attention in Theory and Practice](https://openreview.net/forum?id=LqWW1SrKcH) — near-linear attention.
- [Learning When to Attend: Conditional Memory Access for Long-Context LLMs](https://openreview.net/forum?id=8ubnKeIh0Y) — conditional memory access.
- [Reducing Attention Distribution Error with Unified Tail Aggregation for Sparse Attention](https://openreview.net/forum?id=mEpi4pgzpI) — sparse attention approximation.
- [Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts](https://openreview.net/forum?id=llfDaQar98) — hybrid linear attention.
- [On State Reduction in Linear Attention](https://openreview.net/forum?id=KkN6enKAlU) — state-memory reduction.
- [LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning](https://openreview.net/forum?id=sGzXExjoIf) — context compression.
- [StreamAttention: Energy-Efficient and High-Utilization Attention on Systolic Hardware](https://openreview.net/forum?id=SRQuPOMzkX) — hardware mapping and attention kernels.

## Mixture-of-Experts, routing and elastic capacity

- [A Recipe for an Elastic Mixture: One Mixture-of-Experts for Every Resource Budget](https://openreview.net/forum?id=0stDFvtMGN) — elastic MoE capacity.
- [TEAM: Temporal–Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](https://openreview.net/forum?id=Zfgv1Sa7Ot) — dynamic expert activation.
- [Structural Outlier aware post-training Quantization](https://openreview.net/forum?id=oqacjavnry) — outlier aware-quantization
- [LExI: Layer-Adaptive Active Experts for Efficient MoE Inference](https://openreview.net/forum?id=oYiauQToKN) — layer-specific expert activation.
- [MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models](https://openreview.net/forum?id=0yK6aZLoEF) — elastic expert capacity.
- [DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices](https://openreview.net/forum?id=YYycBS8yxj) — sparse MoE for edge devices.
- [Selective Sinkhorn Routing for Improved Sparse Mixture of Experts](https://openreview.net/forum?id=qRQU6W1vJ4) — routing algorithm.
- [XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference](https://openreview.net/forum?id=KmY5XUgEYe) — expert execution/sharing.
- [GreenMoE: Exploiting Dynamic Load Imbalance for Energy-Efficient Long-Context MoE Training](https://openreview.net/forum?id=ShvjLjCOGB) — energy/load imbalance.
- [Layout and Fusion Trade-offs for Mixture-of-Experts Inference under Single-Node Tensor Parallelism](https://openreview.net/forum?id=y2sY80Q3fh) — runtime layout/fusion.
- [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](https://openreview.net/forum?id=UZ5V7ub8mi) — MoE pruning/distillation.
- [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://openreview.net/forum?id=ZqRh99OQEK) — MoE-to-dense compression.
- [SRA-MoE: Output-Aware Selective Router Alignment for MoE Quantization](https://openreview.net/forum?id=H0NoX02erJ) — belongs to both MoE and quantization.

## Quantization, pruning and compression

- [AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization](https://openreview.net/forum?id=ddOlFn6d0B) — weight quantization.
- [CafeQ: Calibration-free Quantization via Learned Transformations and Adaptive Rounding](https://openreview.net/forum?id=CBtx52vftl) — PTQ.
- [COAT: COrrelation-Aware Orthogonal Transform for LLM Quantization](https://openreview.net/forum?id=6JGbm48VoG) — transform-based quantization.
- [Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models](https://openreview.net/forum?id=wxKn35RAhW) — quantization robustness.
- [Leech Lattice Vector Quantization for Efficient LLM Compression](https://openreview.net/forum?id=HFk5TQ2ILj) — vector quantization.
- [What Matters for NVFP4 Training? A Scaling Study of Low-Precision Pre-Training Recipes](https://openreview.net/forum?id=jlkIyaG32w) — low-precision training recipes.
- [Activation Quantization of Vision Encoders Needs Prefixing Registers](https://openreview.net/forum?id=p6y2soVAlT) — vision encoder quantization.
- [Zero-Shot Quantization for Vision-Language-Action Models via Trajectory Curvature and Attention Guidance](https://openreview.net/forum?id=9Ljgbk6H3O) — VLA quantization.
- [LEAP: Learnable End-to-End Adaptive Pruning of Large Language Models](https://openreview.net/forum?id=tIsX30TADF) — LLM pruning.
- [SubspacePath Pruner: Inference-time Pruning via Probe-based Representation–Parameter Coupling](https://openreview.net/forum?id=nsIOhilpBE) — inference-time pruning.
- [Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning](https://openreview.net/forum?id=QuWWYol2nP) — layer/depth pruning.
- [Weight Concentration Regularization for Improving Pruning Robustness Under High Sparsity](https://openreview.net/forum?id=sRdWDNWenw) — robust pruning.
- [Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs](https://openreview.net/forum?id=U37zxXMEE0) — recovery after layer pruning.
- [Vision Token Pruning via Query–Vision Interaction Decomposition](https://openreview.net/forum?id=UgbjqumIWe) — VLM visual-token pruning.
- [Decoupling Spatial and Semantic Token Compression for Vision-Language Model Acceleration](https://openreview.net/forum?id=4ylINNNS6y) — multimodal token compression.
- [SparseSAM: Structured Sparsification of Activations in Segment Anything Models](https://openreview.net/forum?id=RzsvMA4pyG) — structured activation sparsity.
- [SFPruner: Single-Forward Visual Token Subset Selection for Resource-Efficient Multimodal Foundation Model Inference](https://openreview.net/forum?id=d9H5uBnHeL) — visual token pruning.
- [Modality-Aware Block Rotation for Vision-Language-Action Model Quantization](https://openreview.net/forum?id=07e99XvHxF) — VLA quantization.
- [QJL is 1-bit Compressive Sensing: An Equivalence and Its Consequences for KV Cache Compression in LLMs](https://openreview.net/forum?id=B0N2TDGiOl) — KV-cache compression; cross-listed with Attention.

## KV cache and long-context memory

- [DropKV: Decoupling Residual-Output Perturbation for Near-Optimal KV-Cache Eviction](https://openreview.net/forum?id=MqfNzH3TVH) — KV eviction.
- [KVgrad: Query-Agnostic KV Cache Eviction via Gradient-based Global Importance Scoring](https://openreview.net/forum?id=cg1wTCJjjk) — KV eviction.
- [AgentKV: Phase-Aware KV Eviction for Agentic LLMs](https://openreview.net/forum?id=s80e8otSxP) — phase-aware eviction.
- [HYBRIDKV: Exploiting Head-Dominant Reconstruction for Efficient Query-Agnostic KV Cache Compression](https://openreview.net/forum?id=sWF4OAXw7u) — cache compression.
- [Recency/Frequency Adaptive KV Caching for Large Language Model Serving](https://openreview.net/forum?id=WklS2q5SD9) — cache retention policy.
- [Cache You Later: Post-Compression KV Repair for Long-Context Agentic LLM Inference](https://openreview.net/forum?id=LsrmZrp7tW) — repair after compression.
- [Sigmoid Attention as a Better Substrate for Learned KV Cache Eviction](https://openreview.net/forum?id=NiWI3Hrnd5) — attention/cache co-design.
- [QJL is 1-bit Compressive Sensing](https://openreview.net/forum?id=B0N2TDGiOl) — compressed KV state.

## Speculative decoding and serving

- [BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting](https://openreview.net/forum?id=uqeOxztSIS) — budget-aware speculative decoding.
- [ShadowSpec: Towards Zero Speculation Overhead for Substitute Speculative Decoding](https://openreview.net/forum?id=ss50E1FoBU) — speculative decoding.
- [Fast Inference via Hierarchical Speculative Decoding](https://openreview.net/forum?id=i2IDDWcoEw) — hierarchical speculative decoding.
- [Layer Verification Accelerates Speculative Tree Decoding](https://openreview.net/forum?id=XrqcUo6B6L) — verification strategy.
- [PRESTO: Prefix-Aligned Tree Drafting for Diffusion Speculative Decoding](https://openreview.net/forum?id=VKD70LJaDy) — tree drafting.
- [Characterizing self-speculative decoding approaches for accelerating LLMs](https://openreview.net/forum?id=uXQyCxN7r9) — self-speculative baselines.
- [EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts](https://openreview.net/forum?id=zEJ6juNLZi) — serving/training rollout overlap.

## Profiling, systems and hardware-aware evaluation

- [One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training](https://openreview.net/forum?id=QSv1lTk06x) — energy-constrained runtime.
- [Prelude: Execution-Class Aware Serving for Decision-Style LLM Inference](https://openreview.net/forum?id=UU82BHpcjP) — serving/runtime.
- [Staircase Streaming for Low-Latency Multi-Agent Inference](https://openreview.net/forum?id=9VqvhSZwN6) — latency and serving.
- [StreamAttention: Energy-Efficient and High-Utilization Attention on Systolic Hardware](https://openreview.net/forum?id=SRQuPOMzkX) — hardware mapping.
- [Layout and Fusion Trade-offs for Mixture-of-Experts Inference under Single-Node Tensor Parallelism](https://openreview.net/forum?id=y2sY80Q3fh) — kernel/runtime design.

## LoRA and parameter-efficient adaptation

I did not identify a paper in the accepted-paper titles whose primary subject is LoRA. Keep LoRA as a **shared baseline/adapter technique**, and do not assign an AdaptFM paper to this module solely because it fine-tunes a model. If a contributor finds an accepted paper whose full text materially uses LoRA, add it here with a short justification and the direct forum link.

## Backbones and dataloaders

Backbones and dataloaders are **infrastructure categories**, not paper categories. Use the model and dataset registries documented in [`Backbones/README_backbones.md`](Backbones/README_backbones.md) and [`Dataloaders/README_dataloader.md`](Dataloaders/README_dataloader.md). Register a paper method against one or more supported backbone IDs and dataset IDs rather than copying data/model setup into a paper implementation.

## Fixing a paper entry

Before implementation begins, verify the paper title, latest version, and OpenReview URL against the [official accepted-paper list](https://adaptfm.gitlab.io/accepted-papers). If any forum link redirects or no longer resolves, use the official list rather than guessing an ID.
