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

## Reported result rows

| File | Paper | Rows | Hardware | Status |
|---|---|---|---|---|
| [staircase_streaming_rtx4090_full_2026-11-24.csv](staircase_streaming_rtx4090_full_2026-11-24.csv) | Staircase Streaming (arXiv 2510.05059) | full-scale run: baseline MoA + 13 staircase variants (latency, 128 prompts each); baseline + staircase on the whole GSM8K test set (1319 each) | 1× RTX 4090 24 GB, vLLM 0.34.0 | `partial` (local, measured; headline results in `Serving/staircase_streaming/PAPER_CARD.md`) |

Baseline and method rows for Staircase Streaming (unmodified baseline MoA and staircase with the paper's default settings, f8 r2), copied from the CSV above:

```text
experiment_id,paper_id,method,commit,backbone_id,backbone_revision,dataset_id,split,seed,device,framework_version,dtype,batch_size,input_length,output_length,quality_metric,quality_value,latency_median_ms,latency_p95_ms,throughput,peak_memory_mb,energy_j,status,notes
staircase_moa_1gpu_full:baseline:latency,arxiv-2510.05059,baseline_moa,2f5af63,proposers=text.qwen2_5_1_5b_instruct|text.smollm2_1_7b_instruct|text.qwen2_5_0_5b_instruct;aggregator=text.qwen2_5_1_5b_instruct,proposers=989aa7980e4cf806f80c7fef2b1adb7bc71aa306|31b70e2e869a7173562077fd711b654946d38674|7ae557604adf67be50417f59c2c2f167def9a775;aggregator=989aa7980e4cf806f80c7fef2b1adb7bc71aa306,text.alpaca_eval.eval+text.arena_hard_v0_1.eval,eval,42,"NVIDIA GeForce RTX 4090, 24564 MiB",vllm 0.34.0; python 3.11.10,auto,1,54,506,,,4276.0,5998.6,66.12,18725,,partial,"latency=TTFT (request start -> first aggregator token); throughput=paper TPS (aggregator tokens / request start->last token), median; e2e_median_ms=6113.8; e2e_p95_ms=8210.1; agg_prefill_median_ms=33.5; agg_decode_median_ms=2186.8; agg_calls_mean=1.00; agg_prompt_tokens_mean=1221.2; agg_prompt_token_ratio_vs_baseline=1.00; total_prompt_token_ratio_vs_baseline=1.00; agg_cached_tokens_mean=81.5; n=128; failures=0; peak_memory=whole run, sum over GPUs; sampled nvidia-smi memory.used every 0.5s (device-wide MiB, incl. vLLM KV pool); energy not_measured"
staircase_moa_1gpu_full:staircase_f8_r2:latency,arxiv-2510.05059,staircase_f8_r2,2f5af63,proposers=text.qwen2_5_1_5b_instruct|text.smollm2_1_7b_instruct|text.qwen2_5_0_5b_instruct;aggregator=text.qwen2_5_1_5b_instruct,proposers=989aa7980e4cf806f80c7fef2b1adb7bc71aa306|31b70e2e869a7173562077fd711b654946d38674|7ae557604adf67be50417f59c2c2f167def9a775;aggregator=989aa7980e4cf806f80c7fef2b1adb7bc71aa306,text.alpaca_eval.eval+text.arena_hard_v0_1.eval,eval,42,"NVIDIA GeForce RTX 4090, 24564 MiB",vllm 0.34.0; python 3.11.10,auto,1,54,490,,,97.2,101.5,73.73,18725,,partial,"latency=TTFT (request start -> first aggregator token); throughput=paper TPS (aggregator tokens / request start->last token), median; e2e_median_ms=5767.5; e2e_p95_ms=7179.0; agg_prefill_median_ms=172.0; agg_decode_median_ms=2981.7; agg_calls_mean=4.33; agg_prompt_tokens_mean=4601.3; agg_prompt_token_ratio_vs_baseline=3.77; total_prompt_token_ratio_vs_baseline=3.34; agg_cached_tokens_mean=2456.5; n=128; failures=0; peak_memory=whole run, sum over GPUs; sampled nvidia-smi memory.used every 0.5s (device-wide MiB, incl. vLLM KV pool); energy not_measured"
staircase_moa_1gpu_full:baseline:gsm8k,arxiv-2510.05059,baseline_moa,2f5af63,proposers=text.qwen2_5_1_5b_instruct|text.smollm2_1_7b_instruct|text.qwen2_5_0_5b_instruct;aggregator=text.qwen2_5_1_5b_instruct,proposers=989aa7980e4cf806f80c7fef2b1adb7bc71aa306|31b70e2e869a7173562077fd711b654946d38674|7ae557604adf67be50417f59c2c2f167def9a775;aggregator=989aa7980e4cf806f80c7fef2b1adb7bc71aa306,text.gsm8k.test,test,42,"NVIDIA GeForce RTX 4090, 24564 MiB",vllm 0.34.0; python 3.11.10,auto,1,122,290,exact_match,0.6611,3109.5,4805.1,66.74,18725,,partial,"latency=TTFT (request start -> first aggregator token); throughput=paper TPS (aggregator tokens / request start->last token), median; e2e_median_ms=4396.6; e2e_p95_ms=6681.8; agg_prefill_median_ms=27.3; agg_decode_median_ms=1253.4; agg_calls_mean=1.00; agg_prompt_tokens_mean=1055.2; agg_prompt_token_ratio_vs_baseline=1.00; total_prompt_token_ratio_vs_baseline=1.00; agg_cached_tokens_mean=67.7; n=1319; failures=0; peak_memory=whole run, sum over GPUs; sampled nvidia-smi memory.used every 0.5s (device-wide MiB, incl. vLLM KV pool); energy not_measured"
staircase_moa_1gpu_full:staircase_f8_r2:gsm8k,arxiv-2510.05059,staircase_f8_r2,2f5af63,proposers=text.qwen2_5_1_5b_instruct|text.smollm2_1_7b_instruct|text.qwen2_5_0_5b_instruct;aggregator=text.qwen2_5_1_5b_instruct,proposers=989aa7980e4cf806f80c7fef2b1adb7bc71aa306|31b70e2e869a7173562077fd711b654946d38674|7ae557604adf67be50417f59c2c2f167def9a775;aggregator=989aa7980e4cf806f80c7fef2b1adb7bc71aa306,text.gsm8k.test,test,42,"NVIDIA GeForce RTX 4090, 24564 MiB",vllm 0.34.0; python 3.11.10,auto,1,122,294,exact_match,0.6710,98.5,102.4,76.05,18725,,partial,"latency=TTFT (request start -> first aggregator token); throughput=paper TPS (aggregator tokens / request start->last token), median; e2e_median_ms=3948.3; e2e_p95_ms=5940.5; agg_prefill_median_ms=131.3; agg_decode_median_ms=1845.7; agg_calls_mean=3.81; agg_prompt_tokens_mean=3416.6; agg_prompt_token_ratio_vs_baseline=3.24; total_prompt_token_ratio_vs_baseline=2.65; agg_cached_tokens_mean=1554.1; n=1319; failures=0; peak_memory=whole run, sum over GPUs; sampled nvidia-smi memory.used every 0.5s (device-wide MiB, incl. vLLM KV pool); energy not_measured"
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
