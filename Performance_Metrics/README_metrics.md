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

## COAT benchmark

From the repository root, the CPU check writes a report from the synthetic fixture. The quantization guide has the CPU setup and exact CUDA command. [Run COAT](../Quantization_and_Pruning/README_qp.md#run-coat)

```bash
COAT_RUN=$(mktemp -d)
python -m Quantization_and_Pruning.experiments.run_coat --dry-run --output-dir "$COAT_RUN"
```

`report.json` is an array of rows. Each `experiment_id` identifies a run, variant, and workload. COAT extends the required fields with dataset and tokenizer revisions, hardware and software details, `parameter_count`, `tensor_storage_bytes`, `checkpoint_bytes` where available, and preparation wall times where applicable. The parameter count includes the logical weights represented by packed tensors. `source_dirty` indicates local changes relative to `commit`. Energy and cost are `null` with status `not_measured`.

| Workload | Reported values |
|---|---|
| `perplexity/<dataset>` | Next-token perplexity on adjacent valid token pairs in complete non-overlapping windows. The 7B profile uses 2048-token windows for WikiText2 test, PTB test, and the first 128 complete English C4 validation windows. |
| `perplexity/mean` | Arithmetic mean of the corpus perplexities. |
| `zero_shot/<task>` | Accuracy in percent for each pinned harness task using `acc,none` and zero demonstrations. MMLU uses the harness's size-weighted group score. |
| `zero_shot/mean` | Arithmetic mean of the nine task scores. |
| `generation/prefill` and `generation/decode` | Batch size 1, a 2048-token prompt for Llama-2 7B or 512 tokens for TinyLlama, and 128 fixed continuation tokens from WikiText2 test. Five warmups and 30 measured iterations. Report wall-clock median, p95, raw milliseconds and tokens/s. Decode timing covers all 128 cached forward steps. |

`quality_delta` is the row's score minus the matched floating-point baseline. Timing excludes loading, tokenization and input transfer. CUDA work is synchronized at timing boundaries. `peak_memory_mb` is peak CUDA allocated memory in decimal MB and is `null` on CPU. The `checkpoint_bytes` field uses the source safetensor shards for the 7B baseline and the saved artifact for W4A4KV4. The transient rotation-only variant reports `tensor_storage_bytes`. `preparation_wall_s` records covariance collection, rotation construction including eigensolve, fusion, and GPTQ for the variants that use them.

### Measured synthetic CPU run

Windows 11, AMD64 Family 25 Model 80, one PyTorch thread, Python 3.13.6, PyTorch 2.15.0+cpu, and Transformers 4.48.3. PPL uses four held-out 32-token windows. Timing uses a 16-token prompt, four fixed decode tokens, one warmup, and three measured iterations. The profile is `synthetic-smoke`. Peak memory, energy, and cost are `not_measured`.

| Variant | PPL | Prefill median / p95 (ms) | Decode median / p95 (ms) | Decode tokens/s | Status |
|---|---:|---:|---:|---:|---|
| Floating-point FP32 | 129.8261 | 2.90 / 2.92 | 11.28 / 12.57 | 348.05 | `partial` |
| COAT rotation-only FP32 | 129.8261 | 5.68 / 5.92 | 18.44 / 19.67 | 213.44 | `partial` |
| COAT W4A4KV4 with FP32 compute | 129.8717 | 14.55 / 15.48 | 55.59 / 73.19 | 66.58 | `partial` |

### Measured TinyLlama GPU run

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6` ran on one NVIDIA Tesla T4 with PyTorch `2.11.0+cu130` and Transformers `4.48.3`. WikiText-2 test perplexity used 330 windows of 1024 tokens and 337590 scored tokens. COAT calibrated on 128 WikiText-2 training windows of 512 tokens. The W4A4KV4 variant uses packed INT4 storage / FP16 compute. Energy and cost were unmeasured. [T4 run command](../Quantization_and_Pruning/README_qp.md#tinyllama-on-a-t4)

| Variant | WikiText-2 test PPL | Prefill median / p95 (ms) | Decode median / p95 (ms) | Prefill tokens/s | Decode tokens/s | Prefill peak allocated (MB) | Decode peak allocated (MB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP16 baseline | 8.918699027193576 | 76.5203805000283 / 77.68567220005025 | 3640.8193019999544 / 3727.6137939000478 | 6697.6467625801415 | 35.06186323060157 | 2302.605824 | 2230.564352 | `partial` |
| COAT rotation-only FP16 | 8.918598526694602 | 178.43607200006772 / 183.51887234980495 | 8903.963900500003 / 8956.405533099996 | 2877.444361822512 | 14.382164145491837 | 2341.226496 | 2264.99072 | `partial` |
| COAT W4A4KV4 | 11.691915222981782 | 353.917150000143 / 355.0558034499999 | 22151.110946000015 / 22241.88096125001 | 1446.0447016543008 | 5.774415324247129 | 937.375744 | 892.555264 | `partial` |

Schema-complete CSV rows for the measured T4 workloads. Blank fields indicate metrics outside the workload or unmeasured energy. The report records local changes relative to the listed commit.

```csv
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes,energy_status
b9ebb32629944f73952c96446fe2f8d7/floating_point/perplexity/wikitext2,openreview:6JGbm48VoG,floating_point,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,1024,,perplexity,8.918699027193576,,,,,,partial,Untouched floating-point backbone.,not_measured
b9ebb32629944f73952c96446fe2f8d7/floating_point/generation/prefill,openreview:6JGbm48VoG,floating_point,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,0,,,76.5203805000283,77.68567220005025,6697.6467625801415,2302.605824,,partial,Untouched floating-point backbone.,not_measured
b9ebb32629944f73952c96446fe2f8d7/floating_point/generation/decode,openreview:6JGbm48VoG,floating_point,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,128,,,3640.8193019999544,3727.6137939000478,35.06186323060157,2230.564352,,partial,Untouched floating-point backbone.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_rotation_only/perplexity/wikitext2,openreview:6JGbm48VoG,coat_rotation_only,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,1024,,perplexity,8.918598526694602,,,,,,partial,COAT rotations with floating-point inference.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_rotation_only/generation/prefill,openreview:6JGbm48VoG,coat_rotation_only,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,0,,,178.43607200006772,183.51887234980495,2877.444361822512,2341.226496,,partial,COAT rotations with floating-point inference.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_rotation_only/generation/decode,openreview:6JGbm48VoG,coat_rotation_only,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,128,,,8903.963900500003,8956.405533099996,14.382164145491837,2264.99072,,partial,COAT rotations with floating-point inference.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_w4a4kv4/perplexity/wikitext2,openreview:6JGbm48VoG,coat_w4a4kv4,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,1024,,perplexity,11.691915222981782,,,,,,partial,Packed INT4 storage with floating-point compute.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_w4a4kv4/generation/prefill,openreview:6JGbm48VoG,coat_w4a4kv4,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,0,,,353.917150000143,355.0558034499999,1446.0447016543008,937.375744,,partial,Packed INT4 storage with floating-point compute.,not_measured
b9ebb32629944f73952c96446fe2f8d7/coat_w4a4kv4/generation/decode,openreview:6JGbm48VoG,coat_w4a4kv4,8aaad7d62ba3cb181c551e1b644b205dff701107,TinyLlama/TinyLlama-1.1B-Chat-v1.0,fe8a4ea1ffedaf415f4da2f062534de366a451e6,wikitext2,test,0,cuda:0,2.11.0+cu130,float16,1,512,128,,,22151.110946000015,22241.88096125001,5.774415324247129,892.555264,,partial,Packed INT4 storage with floating-point compute.,not_measured
```

### Pending Llama-2 7B GPU benchmark

| RTX A6000 / Llama-2 7B | Mean PPL | Mean zero-shot accuracy | Prefill / decode latency | Decode tokens/s | Peak allocated MB | Status |
|---|---:|---:|---|---:|---:|---|
| Floating-point FP16 | — | — | — | — | — | Pending GPU run |
| COAT rotation-only FP16 | — | — | — | — | — | Pending GPU run |
| COAT W4A4KV4, packed INT4 storage with FP16 compute | — | — | — | — | — | Pending GPU run |

After a successful CUDA run, fill these summaries from `report.json` in the specified output directory and retain the per-workload rows and timing samples. Label the first reference benchmark `partial`.

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
