# Performance Metrics and Benchmark Protocol

## Purpose

This is the shared measurement contract for the repository. A method is not considered “hardware-aware” merely because it reduces FLOPs or parameter count; it should demonstrate measured resource behavior on named hardware and report task quality alongside it.

## Core metrics

| Metric | Definition to record | Notes |
|---|---|---|
| Quality | Task-specific score and evaluator/version | Include unmodified baseline and quality delta. |
| Latency | Wall-clock time with units and timing scope | For generation, separate prefill and decode; report median and p95 when enough samples exist. |
| Throughput | samples/s, tokens/s, or frames/s | State batch size and whether it includes preprocessing. |
| Peak memory | Peak allocated and/or device memory, with measurement method | Do not mix allocated/reserved/system memory without labels. |
| Model size | Parameter count and on-disk weight size | State dtype/format and whether adapters are included. |
| Energy | Joules per request/token/sample, if measured | Record sensor/API, power sampling interval, idle subtraction, and scope. Use `not_measured` rather than inventing values. |
| Cost | Cloud/hardware cost per unit, if applicable | Give provider, instance, pricing date and calculation. |
| Robustness | Quality across seeds, distributions, or hardware | Include only when evaluated. |

## Minimum experiment metadata

Every result row should include:

- paper/method ID, implementation commit, and status (`reproduced`, `partial`, etc.);
- backbone ID/revision and dataset ID/split/revision;
- preprocessing/tokenizer version, seed, number of examples, task metric;
- device make/model, accelerator memory, driver/runtime version;
- Python/framework/library versions and kernel backend;
- batch size, input/output/sequence length or image resolution;
- dtype/quantization format, method settings, warm-up count, measured iterations;
- timing boundaries (includes load? preprocess? tokenization? generation? synchronization?);
- quality, latency, throughput, memory, energy and cost fields, using explicit null/status when not measured.

## Measurement procedure

1. Fix an environment and capture software/hardware metadata.
2. Run correctness/quality checks first.
3. Warm up the model and kernels. Synchronize asynchronous GPU work at measurement boundaries.
4. Use enough repeated runs to report stable summary statistics; retain raw samples when practical.
5. Measure baseline and adapted variant using the same inputs and settings.
6. Report preprocessing, transfer, compilation, and initialization costs separately if they are excluded from steady-state latency.
7. Report failures and unsupported configurations; do not silently drop them.

## Required report format

Use one machine-readable CSV/JSON row per experiment configuration. At minimum:

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
```

Unknown/not-measured quantities should be empty/null plus a `status` or notes entry—never `0` unless zero was actually measured.

## Recommended plots

- quality vs. latency;
- quality vs. peak memory;
- tokens/s vs. sequence length and batch size;
- quality vs. compression ratio/sparsity/bit-width;
- energy per token/request vs. quality, only for measured energy;
- Pareto frontiers across configurations.

## Paper references with system/energy relevance

- [GreenMoE: Exploiting Dynamic Load Imbalance for Energy-Efficient Long-Context MoE Training](https://openreview.net/forum?id=ShvjLjCOGB)
- [Layout and Fusion Trade-offs for Mixture-of-Experts Inference under Single-Node Tensor Parallelism](https://openreview.net/forum?id=y2sY80Q3fh)
- [EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts](https://openreview.net/forum?id=uCthChFfbe)
- [official AdaptFM accepted-paper list](https://adaptfm.gitlab.io/accepted-papers) for StreamAttention, Prelude, Staircase Streaming, and other systems-focused papers.

## Benchmark anti-patterns

Do not compare different checkpoints, input lengths, quantization settings, warm-up policies, or batch sizes and attribute the whole difference to a method. Do not report GPU utilization alone as speed. Do not mix paper-reported numbers with locally reproduced numbers without marking the source.


---

## Paper #12 — Rethinking Layer Redundancy in LLMs

**Paper:** [Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning](https://openreview.net/forum?id=QuWWYol2nP)
**Primary module:** `Quantization_and_Pruning/pruning/layer_depth/paper12_layer_redundancy/`
**Implementation status:** `partial-reproduction`
**Backbone:** `meta-llama/Llama-2-7b-hf` (paper used LLaMA3-8B, LLaMA3.1-8B, Qwen3-8B)
**Hardware:** _(fill in your device — e.g. NVIDIA A100 40GB)_
**Seed:** 42 | **Calibration samples:** 128 | **Pruning budget k:** 7

### Paper-reported results (from Table 1 — LLaMA3-8B, TLM objective)

> Source: paper Table 1. Units: accuracy in %, perplexity is dimensionless. Mark = paper-reported.

| experiment_id | method | objective | backbone | k_pruned | hellaswag_acc | boolq_acc | avg_acc | margin_tlm | status |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| paper_llama3_8b_baseline | none | none | LLaMA3-8B | 0 | 79.23 | 82.26 | 74.58 | -0.571 | paper-reported |
| paper_llama3_8b_oneshot_tlm_k7 | one-shot | tlm | LLaMA3-8B | 7 | 59.23 | 62.72 | 57.99 | -0.312 | paper-reported |
| paper_llama3_8b_greedy_tlm_k7 | greedy | tlm | LLaMA3-8B | 7 | 60.47 | 67.49 | 60.69 | -0.408 | paper-reported |
| paper_llama3_8b_oneshot_tlm_k9 | one-shot | tlm | LLaMA3-8B | 9 | 46.63 | 63.18 | 52.51 | 0.007 | paper-reported |
| paper_llama3_8b_greedy_tlm_k9 | greedy | tlm | LLaMA3-8B | 9 | 53.28 | 63.21 | 54.43 | -0.168 | paper-reported |

### Local results (partial reproduction — LLaMA-2-7B, k=7)

> Fill these in after running `python baseline_runner.py` and `python run_experiment.py`.
> Device, latency, and memory should be filled from your actual run environment.

| experiment_id | method | objective | backbone | k_pruned | perplexity_wikitext2 | hellaswag_acc | boolq_acc | peak_memory_mb | latency_ms | status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| local_baseline | none | none | llama-2-7b | 0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| local_oneshot_ppl_k7 | one-shot | perplexity | llama-2-7b | 7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| local_oneshot_tlm_k7 | one-shot | tlm | llama-2-7b | 7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| local_greedy_ppl_k7 | greedy | perplexity | llama-2-7b | 7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| local_greedy_tlm_k7 | greedy | tlm | llama-2-7b | 7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |

### Key observation to verify (main paper claim)

After running experiments, check:
1. `local_oneshot_ppl_k7` and `local_greedy_ppl_k7` → removed layer indices should be **similar** (objective dominates)
2. `local_oneshot_ppl_k7` and `local_oneshot_tlm_k7` → removed layer indices should be **different** (objective matters more than algorithm)
3. Spearman correlation between perplexity rank and accuracy rank should be near zero or negative
