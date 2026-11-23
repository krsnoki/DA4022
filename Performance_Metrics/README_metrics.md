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

## Recorded results

### BASTION (Speculative_Decoding/BASTION), status `partial`

These are locally measured summary rows. The per-dataset rows (one per model × benchmark × method) are in [`results/e1_da4022_results.csv`](../Speculative_Decoding/BASTION/results/e1_da4022_results.csv) and [`results/sweep_qwen3-8b_da4022_results.csv`](../Speculative_Decoding/BASTION/results/sweep_qwen3-8b_da4022_results.csv). The analysis is in [`REPORT.md`](../Speculative_Decoding/BASTION/REPORT.md).

Protocol shared by all rows:
- **Benchmarks:** `bastion-short-8` is GSM8K, MATH500, AIME25, HumanEval, MBPP and LCB release_v2 (test splits), plus MT-Bench and Alpaca (train splits).
  - E1 uses 64 prompts per benchmark, all 30 for AIME25 and 40 MT-Bench questions × 2 turns: n = 494 generations.
  - E4 uses 16 prompts per benchmark (MT-Bench 16 × 2 turns): n = 144.
- **Decoding:** T=0, `max_new_tokens` 2048, non-thinking chat template, one discarded warm-up pass per mode.
- **Latency:** decode time per output token (TPOT, prefill excluded), median and p95 over records.
- **Throughput:** 1 / mean TPOT, in decode tokens/s.
- **Peak memory:** `torch.cuda.max_memory_allocated` in MiB, maximum over records. It includes both target and drafter weights in every mode.
- **Energy:** `not_measured`.
- **Speedup (in notes):** mean TPOT(AR) / mean TPOT(method) per benchmark on the same prompts, then the unweighted mean over the 8 benchmarks (the paper's Table 1 convention).
- **Quality:** boxed-answer accuracy pooled over GSM8K + MATH-500 + AIME25 (158 problems) from `analysis/score_math.py`, a simple normalizer; the E4 rows were not scored.
- **Commits:** `commit` is the DA4022 harness commit. The harness code is identical in cbac596, 3653faa and ddc864e. Upstream BASTION is at 1989b80.

```csv
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
e1:Qwen3-4B:ar,bastion-arxiv-2605.29727,AR,cbac596,Qwen/Qwen3-4B,1cfa9a7,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,178.1,512.7,math acc (158),0.6899,13.810,14.240,72.25,9202,,partial,E1 baseline; n=494
e1:Qwen3-4B:dflash,bastion-arxiv-2605.29727,DFlash,cbac596,Qwen/Qwen3-4B,1cfa9a7,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,178.1,520.2,math acc (158),0.6772,3.297,8.432,247.99,9357,,partial,E1; n=494; speedup 3.95x (avg of 8); AAL 5.69
e1:Qwen3-4B:bastion-adaptive,bastion-arxiv-2605.29727,BASTION (adaptive),cbac596,Qwen/Qwen3-4B,1cfa9a7,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,178.1,525.9,math acc (158),0.6772,2.857,6.627,292.94,10323,,partial,E1; n=494; speedup 4.50x (avg of 8); AAL 7.58; mean tree 60.1; warm profile
e1:Qwen3-8B:ar,bastion-arxiv-2605.29727,AR,cbac596,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,176.5,500.9,math acc (158),0.7025,22.205,22.635,45.03,18051,,partial,E1 baseline; n=494
e1:Qwen3-8B:dflash,bastion-arxiv-2605.29727,DFlash,cbac596,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,176.5,497.3,math acc (158),0.7089,4.962,13.240,160.94,18469,,partial,E1; n=494; speedup 4.17x (avg of 8); AAL 5.76
e1:Qwen3-8B:bastion-adaptive,bastion-arxiv-2605.29727,BASTION (adaptive),cbac596,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,176.5,499.2,math acc (158),0.7025,4.691,10.679,180.28,19203,,partial,E1; n=494; speedup 4.47x (avg of 8); AAL 7.03; mean tree 33.2; warm profile
e4:Qwen3-8B:ar,bastion-arxiv-2605.29727,AR,cbac596,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,568.5,not_scored,,22.224,22.680,44.99,18051,,partial,E4 baseline = E1 AR on sweep prompts; n=144
e4:Qwen3-8B:fixed-32,bastion-arxiv-2605.29727,BASTION-Fixed-32,3653faa,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,577.8,not_scored,,4.778,10.304,183.82,19201,,partial,E4; n=144; speedup 4.51x (avg of 8); AAL 7.03; N=32
e4:Qwen3-8B:fixed-128,bastion-arxiv-2605.29727,BASTION-Fixed-128,3653faa,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,578.0,not_scored,,5.173,10.580,172.23,19382,,partial,E4; n=144; speedup 4.20x (avg of 8); AAL 7.96; N=128
e4:Qwen3-8B:fixed-1024,bastion-arxiv-2605.29727,BASTION-Fixed-1024,3653faa,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,588.4,not_scored,,17.252,35.870,50.31,20165,,partial,E4; n=144; speedup 1.23x (avg of 8); AAL 8.99; N=1024
e4:Qwen3-8B:bastion-adaptive-warm,bastion-arxiv-2605.29727,BASTION (adaptive),3653faa,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,570.7,not_scored,,4.763,10.336,184.44,19203,,partial,E4; n=144; speedup 4.52x (avg of 8); AAL 7.15; mean tree 33.2
e4:Qwen3-8B:bastion-adaptive-cold,bastion-arxiv-2605.29727,"BASTION (adaptive, upstream cold profile)",ddc864e,Qwen/Qwen3-8B,b968826,bastion-short-8,test (mt-bench/alpaca: train),0,NVIDIA RTX PRO 4500 Blackwell 32GB,torch 2.8.0+cu128; transformers 4.57.1; CUDA 12.8; SDPA,bf16,1,195.0,602.9,not_scored,,4.754,10.119,183.69,19297,,partial,E4; n=144; speedup 4.51x (avg of 8); AAL 7.40; mean tree 40.0
```

Paper-reported values, **not measured here**, for comparison: A100, T=0, average over the same 8 benchmarks (Table 1). Speedup / AAL: Qwen3-4B DFlash 4.83× / 6.23, BASTION 6.93× / 8.56; Qwen3-8B DFlash 4.76× / 6.24, BASTION 6.61× / 8.48. The paper reports no latency, throughput, memory or quality values for these runs.
