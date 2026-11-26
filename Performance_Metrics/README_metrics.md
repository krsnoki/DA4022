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

### Result files

| Paper / method | File | Rows |
|---|---|---|
| Step-Tagging Early-Stopping (ST-ES), `Adaptive_Stopping/` | [`results/st_es.csv`](results/st_es.csv) | Replay rows (quality and generated tokens on held-out subsets, no runtime) and live rows (measured latency, throughput and peak memory at batch size 1), each with its no-stopping baseline. The paper's prompt-guided baselines are in `results/st_es_prompt_guided.csv`. Details: [`Adaptive_Stopping/methods/step_tagging/PAPER_CARD.md`](../Adaptive_Stopping/methods/step_tagging/PAPER_CARD.md). |

#### ST-ES rows (schema fields; every field of each row is in `results/st_es.csv`)

Backbone `text.deepseek-r1-distill-qwen-1.5b` @ `ad9f0ae`, seed 42, bf16, RTX 4060 Laptop GPU 8 GB, quality = Math-Verify pass@1. ST-ES uses the configuration calibrated for 90% of standard accuracy on the train split. Replay rows have no runtime fields; live rows are measured at batch size 1 on the first 30 (GSM8K) or 10 (MATH-500) test problems. Energy: `not_measured`.

| experiment_id | dataset_id | quality_value | latency_median_ms | latency_p95_ms | throughput | peak_memory_mb | status |
|---|---|---:|---:|---:|---:|---:|---|
| `st_es-text.gsm8k.test-replay-standard` (baseline) | text.gsm8k.test (n=200) | 0.830 | | | | | partial |
| `st_es-text.gsm8k.test-replay-st_es@0.90` | text.gsm8k.test (n=200) | 0.815 | | | | | approximate |
| `st_es-text.gsm8k.test-live-standard` (baseline) | text.gsm8k.test (n=30) | 0.900 | 45229.3 | 123128.0 | 27.6 tok/s | 3599.1 | partial |
| `st_es-text.gsm8k.test-live-st_es@0.90` | text.gsm8k.test (n=30) | 0.700 | 21504.1 | 85181.4 | 27.4 tok/s | 3599.1 | approximate |
| `st_es-text.math500.test-replay-standard` (baseline) | text.math500.test (n=150) | 0.807 | | | | | partial |
| `st_es-text.math500.test-replay-st_es@0.90` | text.math500.test (n=150) | 0.727 | | | | | approximate |
| `st_es-text.math500.test-live-standard` (baseline) | text.math500.test (n=10) | 0.700 | 71286.8 | 375400.3 | 23.8 tok/s | 3798.9 | partial |
| `st_es-text.math500.test-live-st_es@0.90` | text.math500.test (n=10) | 0.800 | 50566.9 | 79347.7 | 27.5 tok/s | 3547.6 | approximate |

The live accuracies come from 30 and 10 problems and are not accuracy estimates; use the replay rows for quality.

Shared quality scorers live in [`quality/`](quality/): `math_answer.py` checks math final answers with Math-Verify.

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
