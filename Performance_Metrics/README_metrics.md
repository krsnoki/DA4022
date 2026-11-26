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

## Reported results

One row per contributed method plus its unmodified baseline, in the schema above. Full per-configuration
results live next to each implementation.

### KVgrad (Attention, #8)

`status=partial`: Qwen3-0.6B, RULER 4K, 130 examples (10 per task), RTX 5050 Laptop GPU, bfloat16, batch size 1.
All 22 configurations (KVzip, KVzip+, KVgrad at 2:1-8:1; score-variant ablation at 8:1-32:1):
[`Attention/experiments/kvgrad/results_rtx5050_bf16/summary.csv`](../Attention/experiments/kvgrad/results_rtx5050_bf16/summary.csv).
Paper card: [`Attention/experiments/kvgrad/PAPER_CARD.md`](../Attention/experiments/kvgrad/PAPER_CARD.md).

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
kvgrad-ruler4096-no_press-cr0.000,kvgrad-icml2026-adaptfm,no_press,686bd09-dirty,Qwen/Qwen3-0.6B,c1899de289a04d12100db370d81485cdf75e47ca,simonjegou/ruler/4096,"test (10/task, 13 tasks)",42,NVIDIA GeForce RTX 5050 Laptop GPU,torch 2.14.1+cu130; transformers 5.2.0; kvpress 0.5.5,bfloat16,1,3807,15,ruler_string_match_macro,79.32,1864.4,4553.9,0.4129,2194.0,,partial,"compression_ratio=0.0; n=130; KV cache kept 416.4 of 416.4 MB (median context, theoretical: kvpress masks evicted entries, so peak_memory_mb does not shrink); prefill+compress median 763.0 ms; decode median 1056.8 ms; throughput in examples/s; latency = full pipeline call incl. tokenisation; attn=sdpa; energy not measured"
kvgrad-ruler4096-kvgrad-cr0.875,kvgrad-icml2026-adaptfm,kvgrad,686bd09-dirty,Qwen/Qwen3-0.6B,c1899de289a04d12100db370d81485cdf75e47ca,simonjegou/ruler/4096,"test (10/task, 13 tasks)",42,NVIDIA GeForce RTX 5050 Laptop GPU,torch 2.14.1+cu130; transformers 5.2.0; kvpress 0.5.5,bfloat16,1,3807,18,ruler_string_match_macro,75.09,7051.8,10859.5,0.136,4984.0,,partial,"compression_ratio=0.875; n=130; KV cache kept 52.0 of 416.4 MB (median context, theoretical: kvpress masks evicted entries, so peak_memory_mb does not shrink); prefill+compress median 5672.5 ms; decode median 1136.2 ms; throughput in examples/s; latency = full pipeline call incl. tokenisation; attn=sdpa; energy not measured"
```
