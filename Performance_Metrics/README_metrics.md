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

## Reported results

### ORA-Q ([OpenReview](https://openreview.net/forum?id=oqacjavnry), `Quantization_and_Pruning/`)

Status `partial`: DAv2-Small, NYUv2 Eigen test (654 images), Tesla T4. The delta1 order matches the paper, but ORA-Q reaches 0.6606 against 0.8507 reported. The first row is the unmodified FP32 baseline. Quantized rows use fake quantization.
Details: [paper card](../Quantization_and_Pruning/ORA_Q_PAPER_CARD.md), [results.csv](../Quantization_and_Pruning/experiments/ora_q/results.csv), [role report](../Quantization_and_Pruning/experiments/ora_q/role_report.json), [conv g=16 ablation](../Quantization_and_Pruning/experiments/ora_q/results_ablation_conv_g16.csv).

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
oraq-nyuv2-fp32-s0,ora-q,fp32,26518ab,vision.depth-anything-v2-small,5426e4f0f36572d16453bbda7a8389317b1bef99,vision.nyuv2.test,eigen_test[:654],0,Tesla T4,torch 2.11.0+cu130,float32,1,518x686,480x640,delta1,0.9557,77.28,78.15,12.94,310.0,,partial,unmodified checkpoint; abs_rel=0.0617; throughput in images/s; latency is model forward only; energy not measured
oraq-nyuv2-minmax-w4a4-s0,ora-q,minmax,26518ab,vision.depth-anything-v2-small,5426e4f0f36572d16453bbda7a8389317b1bef99,vision.nyuv2.test,eigen_test[:654],0,Tesla T4,torch 2.11.0+cu130,fake-quant W4A4 (float32 compute),1,518x686,480x640,delta1,0.4573,165.33,166.29,6.05,698.3,,partial,fake quantization; config minmax.yaml; abs_rel=0.2962; calib_seconds=1.9; throughput in images/s; latency is model forward only; energy not measured
oraq-nyuv2-percentile-w4a4-s0,ora-q,percentile,26518ab,vision.depth-anything-v2-small,5426e4f0f36572d16453bbda7a8389317b1bef99,vision.nyuv2.test,eigen_test[:654],0,Tesla T4,torch 2.11.0+cu130,fake-quant W4A4 (float32 compute),1,518x686,480x640,delta1,0.4668,166.58,167.81,6.0,697.0,,partial,fake quantization; config percentile.yaml; abs_rel=0.2915; calib_seconds=27.9; throughput in images/s; latency is model forward only; energy not measured
oraq-nyuv2-ora_q-w4a4-s0,ora-q,ora_q,26518ab,vision.depth-anything-v2-small,5426e4f0f36572d16453bbda7a8389317b1bef99,vision.nyuv2.test,eigen_test[:654],0,Tesla T4,torch 2.11.0+cu130,fake-quant W4A4 (float32 compute),1,518x686,480x640,delta1,0.6606,187.4,189.28,5.34,698.1,,partial,fake quantization; config ora_q.yaml; abs_rel=0.2002; calib_seconds=28.6; throughput in images/s; latency is model forward only; energy not measured
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
