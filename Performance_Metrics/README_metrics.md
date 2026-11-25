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

## Results: AAAC (Qwen2.5-0.5B, WikiText-2)

Measured rows for [AAAC](../Quantization_and_Pruning/quantization/aaac/PAPER_CARD.md). All variants use identical evaluation tokens (146 non-overlapping 2,048-token windows). Latency is one prefill forward pass over one window (2 warm-up, 10 timed runs). Peak memory on MPS is `max(driver allocated, process peak RSS)` in MiB and accumulates across variants run in one process, so do not compare it between rows. Throughput and latency come from a dense dequantization reference runtime, not native INT4 kernels.

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
aaac-qwen05b-unmodified,arxiv:2605.08692v2,unmodified,269389d,Qwen/Qwen2.5-0.5B,060db6499f32faf8b98477b0a26969ef7d8b9987,wikitext/wikitext-2-raw-v1,test,42,Apple Silicon MPS,torch 2.15.0 / transformers 4.57.6,float16,1,2048,0,perplexity,13.0702,147.588,148.528,13876.51 tok/s,5712.69,,reproduced,"baseline; matches paper 13.07; checkpoint 988065536 B; energy not_measured"
aaac-qwen05b-rtn-asym-g128,arxiv:2605.08692v2,rtn-int4-zero-point,269389d,Qwen/Qwen2.5-0.5B,060db6499f32faf8b98477b0a26969ef7d8b9987,wikitext/wikitext-2-raw-v1,test,42,Apple Silicon MPS,torch 2.15.0 / transformers 4.57.6,float16 + int4 g128,1,2048,0,perplexity,15.5420,253.434,254.120,8081.01 tok/s,5744.69,,reproduced,"matches paper 15.54; quant 0.3 s; packed 187299840 B; energy not_measured"
aaac-qwen05b-aaac-g128-s128,arxiv:2605.08692v2,aaac,269389d,Qwen/Qwen2.5-0.5B,060db6499f32faf8b98477b0a26969ef7d8b9987,wikitext/wikitext-2-raw-v1,test,42,Apple Silicon MPS,torch 2.15.0 / transformers 4.57.6,float16 + int4 g128 S128,1,2048,0,perplexity,14.7589,320.054,320.823,6398.92 tok/s,6048.69,,approximate,"paper 14.47; Eq. 9-10 init; C4 calib 4x2048 seed 42; fit 974.9 s on CPU; packed 184515072 B; energy not_measured"
aaac-qwen05b-aaac-g128-s16,arxiv:2605.08692v2,aaac,269389d,Qwen/Qwen2.5-0.5B,060db6499f32faf8b98477b0a26969ef7d8b9987,wikitext/wikitext-2-raw-v1,test,42,Apple Silicon MPS,torch 2.15.0 / transformers 4.57.6,float16 + int4 g128 S16,1,2048,0,perplexity,14.3716,327.937,329.187,6245.10 tok/s,6072.69,,approximate,"paper 14.27; Eq. 9-10 init; C4 calib 4x2048 seed 42; fit 1006.3 s on CPU; packed 187310592 B; energy not_measured"
```
