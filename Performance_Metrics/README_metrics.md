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

## Recorded results

Rows appended by contributors, in the schema above. Empty cells are unmeasured, never zero.

### SCPS: Resource-Adaptive Foundation Model Reasoning via Semantic Coverage

Owner card: [`Adaptive_Stopping/methods/budget_policy/scps/PAPER_CARD.md`](../Adaptive_Stopping/methods/budget_policy/scps/PAPER_CARD.md). Paper-reported numbers are kept in the card, not here.

Synthetic smoke run (mock backbone, CPU, 256 synthetic items). It checks controller logic and budget handling only; it is not a task-quality result. `self_consistency` is the unmodified full-compute baseline.

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
scps_synthetic_mock_smoke,openreview.suD1tqogTd,scps,8711f63,mock.mcq.latent_dist,n/a,synthetic.mcq,synthetic,data=47;run=13,cpu,python=3.13.18,n/a,1,478.0,271.93,portfolio_pass@16,0.9336,1.99,3.008,133072.962,516.477,,not reproduced,n=256; mean_total_tokens=276.93 (decode+aux); unique=5.094; dup_tok=0.475; majority_acc=0.4141; latency=per-question wall clock; throughput=decode tok/s; memory=process_peak_rss for whole process (shared across methods); energy not_measured
scps_synthetic_mock_smoke,openreview.suD1tqogTd,self_consistency,8711f63,mock.mcq.latent_dist,n/a,synthetic.mcq,synthetic,data=47;run=13,cpu,python=3.13.18,n/a,1,92.0,400.46,portfolio_pass@16,0.8125,0.269,0.309,1447516.384,516.477,,not reproduced,n=256; mean_total_tokens=400.46 (decode+aux); unique=3.375; dup_tok=0.74; majority_acc=0.4375; latency=per-question wall clock; throughput=decode tok/s; memory=process_peak_rss for whole process (shared across methods); energy not_measured
scps_synthetic_mock_smoke,openreview.suD1tqogTd,tree_prefix,8711f63,mock.mcq.latent_dist,n/a,synthetic.mcq,synthetic,data=47;run=13,cpu,python=3.13.18,n/a,1,736.0,404.02,portfolio_pass@16,0.8242,0.412,0.476,963249.581,516.477,,not reproduced,n=256; mean_total_tokens=409.02 (decode+aux); unique=3.383; dup_tok=0.735; majority_acc=0.4297; latency=per-question wall clock; throughput=decode tok/s; memory=process_peak_rss for whole process (shared across methods); energy not_measured
scps_synthetic_mock_smoke,openreview.suD1tqogTd,semantic_pruning,8711f63,mock.mcq.latent_dist,n/a,synthetic.mcq,synthetic,data=47;run=13,cpu,python=3.13.18,n/a,1,736.0,395.93,portfolio_pass@16,0.8164,0.422,0.476,926268.55,516.477,,not reproduced,n=256; mean_total_tokens=400.93 (decode+aux); unique=3.469; dup_tok=0.733; majority_acc=0.4141; latency=per-question wall clock; throughput=decode tok/s; memory=process_peak_rss for whole process (shared across methods); energy not_measured
```

MMLU-Pro run (paper setting), pending:

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
scps_mmlupro_qwen35_9b_vllm,openreview.suD1tqogTd,scps,,Qwen/Qwen3.5-9B,,text.mmlu_pro.test,test,data=47;run=13,,,bfloat16,1,,,portfolio_pass@16,,,,,,,not reproduced,blocked: run not executed yet (no GPU or Hugging Face access where implemented); command in Adaptive_Stopping/methods/budget_policy/scps/PAPER_CARD.md
scps_mmlupro_qwen35_9b_vllm,openreview.suD1tqogTd,self_consistency,,Qwen/Qwen3.5-9B,,text.mmlu_pro.test,test,data=47;run=13,,,bfloat16,1,,,portfolio_pass@16,,,,,,,not reproduced,unmodified full-compute baseline for the row above; blocked for the same reason
```

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
