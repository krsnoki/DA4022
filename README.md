# AdaptFM: Adaptive and Resource-Efficient Foundation Model Inference

This repository organizes implementations and study materials for techniques that make foundation model inference adaptive, efficient, and resource-aware.

The repository is structured around seven topics:

- **Adaptive Stopping** — dynamically deciding when to stop computation.
- **Attention** — optimizing attention computation and allocating resources to informative tokens.
- **Dataloaders** — preparing reproducible and efficient data pipelines.
- **LoRA** — parameter-efficient adaptation of pretrained models.
- **Mixture of Experts** — conditional computation through expert selection and routing.
- **Performance Metrics** — evaluating accuracy, latency, memory, and efficiency trade-offs.
- **Quantization and Pruning** — reducing model computation and memory requirements.

The primary reading resource is the [ICML 2026 Workshop on AdaptFM: Resource-Adaptive Foundation Model Inference](https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM#tab-accept-poster).

The workshop focuses on flexible inference under different resource budgets, including compute, memory, latency, energy, and cost. Its papers provide research directions for understanding and implementing adaptive inference techniques.

## Repository Structure

| Directory | Main objective |
|---|---|
| `Adaptive_Stopping/` | Adaptive-depth inference and early termination |
| `Attention/` | Efficient attention and adaptive token selection |
| `Dataloaders/` | Data loading, preprocessing, and reproducible evaluation |
| `LoRA/` | Low-rank parameter-efficient adaptation |
| `Mixture_of_Experts/` | Conditional computation and expert routing |
| `Performance_Metrics/` | Quality, latency, memory, and efficiency evaluation |
| `Quantization_and_Pruning/` | Reduced-precision inference and model compression |

## 1. Adaptive Stopping

**Objective:** Reduce unnecessary inference computation by terminating processing when additional computation is unlikely to improve the output sufficiently.

### Recommended workshop reading

1. [Step-Tagging Early-Stopping: Toward Controlling the Generation of Language Reasoning Models through Black-Box Step Monitoring](https://openreview.net/search?term=Step-Tagging%20Early-Stopping) — studies early stopping using signals derived from intermediate reasoning steps.
2. [Block-Level Recursion: Adaptive Test-Time Routing in Large Language Models](https://openreview.net/search?term=Block-Level%20Recursion%20Adaptive%20Test-Time%20Routing) — motivates adaptive computation through selective execution of model blocks.

### What to learn

- How intermediate outputs or hidden-state signals can indicate whether further computation is necessary.
- How to define a stopping criterion and its associated computational cost.
- How early termination affects output quality, average latency, and worst-case latency.
- How to distinguish an effective stopping policy from one that simply terminates too early.

### Implementation and evaluation

Start with a fixed-computation baseline. Implement an adaptive stopping rule and compare both approaches using the same model and evaluation data.

Report task quality, average number of executed steps, mean latency, and the fraction of inputs terminated early.

## 2. Attention

**Objective:** Reduce the computational and memory costs associated with attention while preserving task-relevant information.

### Recommended workshop reading

1. [StreamAttention: Energy-Efficient and High-Utilization Attention on Systolic Hardware](https://openreview.net/search?term=StreamAttention) — investigates efficient attention execution and hardware utilization.
2. [SFPruner: Single-Forward Visual Token Subset Selection for Resource-Efficient Multimodal Foundation Model Inference](https://openreview.net/pdf?id=d9H5uBnHeL) — studies efficient selection of informative visual tokens.
3. [CARES: Context-Aware Resolution Selector for VLMs](https://openreview.net/pdf?id=JP7NNuHG8O) — selects sufficient input resolution based on the image and query before processing by a vision-language model.

### What to learn

- Why attention cost increases with sequence length.
- How token selection, token reduction, and efficient attention kernels differ.
- How to allocate computation to informative tokens.
- Why selection overhead must be included when evaluating speed improvements.

For standard self-attention, the attention-matrix computation scales approximately as \(O(N^2d)\), where \(N\) is the sequence length and \(d\) is the feature dimension.

Reducing the sequence to \(k\) tokens can reduce this component of the computation approximately in proportion to

\[
\frac{k^2}{N^2},
\]

although actual end-to-end speedups depend on other operations and the overhead of selecting tokens.

### Implementation and evaluation

Compare the original attention or token-processing pipeline with the optimized version. Measure task quality, token count, peak memory, attention latency, and end-to-end latency.

## 3. Dataloaders

**Objective:** Build reliable, efficient data pipelines for training and evaluating adaptive inference methods.

### Workshop resource

Consult the [AdaptFM workshop paper collection](https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM#tab-accept-poster) for papers with relevant experimental setups, evaluation procedures, and resource measurements.

This folder is primarily an engineering component rather than a standalone research method. No direct one-to-one mapping to a specific workshop paper is assumed.

### What to implement

- Dataset loading and preprocessing.
- Train, validation, and test separation.
- Reproducible sampling and batching.
- Support for different input lengths, resolutions, or sequence sizes when required.
- Efficient worker configuration, pinned memory, and asynchronous data transfer where appropriate.
- Logging of data-loading time separately from model inference time.

### Evaluation

Measure data-loading throughput, GPU utilization, batch-processing time, and reproducibility. Ensure that optimized data pipelines do not change the evaluation split or preprocessing protocol.

## 4. LoRA

**Objective:** Adapt pretrained models using a small number of trainable parameters instead of updating the entire model.

### Workshop resource

Start with the [AdaptFM accepted-poster collection](https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM#tab-accept-poster), particularly papers addressing resource-aware adaptation and task-specific efficiency.

The workshop papers should be treated as related research where applicable; this section does not assume that every paper directly evaluates LoRA.

### Core idea

Given a pretrained weight matrix \(W_0\), LoRA parameterizes its update as

\[
W = W_0 + \Delta W,
\qquad
\Delta W = BA,
\]

where \(A\) and \(B\) are low-rank matrices and the rank \(r\) is much smaller than the original matrix dimensions.

For \(W_0 \in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\), the update requires approximately

\[
r(d_{\mathrm{in}}+d_{\mathrm{out}})
\]

trainable parameters rather than \(d_{\mathrm{in}}d_{\mathrm{out}}\) parameters for a full weight update.

### What to implement

- A pretrained-model baseline.
- LoRA modules with configurable rank.
- Training and checkpoint saving for adapter parameters.
- Adapter loading and inference.
- Comparisons across rank, target modules, and adaptation settings.

### Evaluation

Report task quality, trainable parameter count, training memory, training time, inference latency, and checkpoint size. Distinguish parameter-efficient training from inference acceleration: LoRA does not automatically make inference faster.

## 5. Mixture of Experts

**Objective:** Allocate computation conditionally by activating selected experts rather than applying every expert to every input.

### Recommended workshop reading

[GreenMoE: Exploiting Dynamic Load Imbalance for Energy-Efficient Long-Context MoE Training](https://openreview.net/search?term=GreenMoE) — investigates load imbalance and energy efficiency in mixture-of-experts training.

### Core idea

For expert functions \(E_1,\ldots,E_M\), a routing function selects or weights experts for each input. A simplified MoE layer is

\[
y(x)=\sum_{i=1}^{M}g_i(x)E_i(x),
\]

where \(g_i(x)\) represents the routing weight for expert \(i\). Sparse routing activates only a subset of experts.

### What to learn

- Expert routing and top-\(k\) selection.
- The trade-off between active computation and total parameter storage.
- Load balancing and expert utilization.
- Communication overhead and its impact on actual latency.

### Implementation and evaluation

Begin with a dense baseline and a small MoE configuration. Compare quality, active parameters, total parameters, expert utilization, peak memory, throughput, and end-to-end latency.

Do not assume that fewer active parameters guarantee lower latency; routing and communication can offset the savings.

## 6. Performance Metrics

**Objective:** Evaluate adaptive inference using a consistent quality–resource trade-off rather than relying on a single efficiency metric.

### Recommended workshop reading

Use the [AdaptFM workshop collection](https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM#tab-accept-poster) as a source of experimental designs and evaluation criteria. In particular, compare methods that explicitly measure resource use alongside output quality.

### Metrics to report

| Metric | Purpose |
|---|---|
| Task quality | Measures accuracy or task-specific performance |
| Mean latency | Measures average inference time |
| P50/P95 latency | Measures typical and tail latency |
| Throughput | Measures completed samples or tokens per second |
| Peak memory | Measures maximum memory consumption |
| FLOPs or MACs | Estimates computational work |
| Energy per inference | Measures energy efficiency when supported by reliable measurements |
| Model and adapter size | Measures storage requirements |

### Quality–resource trade-off

A general formulation is

\[
\max_{\pi}\ \mathbb{E}[Q(\pi)]
\quad\text{subject to}\quad
\mathbb{E}[C(\pi)]\le B,
\]

where \(\pi\) is an inference policy, \(Q\) is task quality, \(C\) is resource consumption, and \(B\) is the available budget.

Alternatively, one can minimize a combined objective:

\[
\mathcal{L}(\pi)
=
\mathcal{L}_{\mathrm{task}}(\pi)
+
\lambda C(\pi),
\]

where \(\lambda\) controls the trade-off between task performance and resource use.

### Benchmarking protocol

- Use the same model, dataset, hardware, and evaluation split for all methods.
- Separate preprocessing and data-loading costs from model inference.
- Use warm-up iterations and synchronized timing for GPU measurements.
- Report repeated measurements and variability.
- Include adaptive-policy overhead in end-to-end latency.
- Compare fixed-budget baselines against adaptive methods at comparable resource budgets.

## 7. Quantization and Pruning

**Objective:** Reduce model size, memory traffic, or computation while maintaining acceptable task quality.

### Recommended workshop reading

1. [You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations](https://openreview.net/search?term=You%20Had%20One%20Job%20Per-Task%20Quantization) — explores task-specific quantization using hidden representations.
2. [SFPruner: Single-Forward Visual Token Subset Selection for Resource-Efficient Multimodal Foundation Model Inference](https://openreview.net/pdf?id=d9H5uBnHeL) — provides a related example of pruning input representations to reduce inference cost.

These papers address different levels of the pipeline: model-weight precision and input-token selection, respectively.

### What to learn

**Quantization**
- Post-training quantization and quantization-aware training.
- The effects of precision on memory, throughput, and quality.
- Weight-only versus weight-and-activation quantization.

**Pruning**
- Structured versus unstructured pruning.
- Weight pruning versus token or activation pruning.
- The difference between sparsity and hardware-realized speedup.

### Implementation and evaluation

Establish a full-precision baseline. Apply one compression technique at a time and evaluate the resulting model under identical conditions.

Report model size, peak memory, latency, throughput, and task quality. Where possible, compare theoretical computational savings with measured hardware performance.

## Recommended Workflow

1. **Read the paper:** Identify the problem, assumptions, proposed method, and evaluation protocol.
2. **Establish a baseline:** Run the original model and record task quality and resource consumption.
3. **Reproduce the core method:** Implement the smallest experiment that tests the paper's main claim.
4. **Measure overhead:** Include preprocessing, routing, selection, and other additional computation.
5. **Run controlled comparisons:** Change one factor at a time and maintain the same evaluation conditions.
6. **Document results:** Record configurations, hardware, software versions, metrics, and limitations.
7. **Extend the method:** Explore alternative resource budgets, input distributions, models, or hardware configurations.

## General References

- [AdaptFM Workshop — ICML 2026](https://openreview.net/group?id=ICML.cc/2026/Workshop/AdaptFM#tab-accept-poster)
- [AdaptFM Workshop Website](https://adaptfm.gitlab.io/)
- [Repository: AdaptFM-DA4022](https://github.com/krsnoki/AdaptFM-DA4022)

## Research and Reproducibility Notes

The workshop papers are intended as research references, not as claims that the repository already reproduces their results. Each folder should document its implementation status, dependencies, configuration, commands, and experimental results.

When a method is adapted from a paper, cite the paper and clearly distinguish the original method from any modifications made in this repository.