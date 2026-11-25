# Paper implementation card: One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training

- **Paper / OpenReview:** https://openreview.net/forum?id=QSv1lTk06x (arXiv: https://arxiv.org/pdf/2609.11936) — Schultheis, Kleinegger, Alistarh. Verify version before merging.
- **Primary module:** Performance_Metrics (Profiling, Systems and Hardware-Aware Evaluation)
- **Secondary modules:** none
- **Owner / reviewer:** <your GitHub handle> / <reviewer>
- **Status:** implementing
- **Upstream code / licence:** not found (paper has no repo link; Listing 1 is the only artifact)
- **Backbone IDs:** none for the micro-benchmark; paper's end-to-end test uses Qwen2-0.5B via LLMQ (not reproduced here)
- **Dataset IDs and split:** none (synthetic random / zero matrices)
- **Hardware assumptions:** paper: DGX Spark and L40S. Here: any CUDA GPU; CPU runs are smoke tests only. NVML optional (`pip install nvidia-ml-py`).

## Research summary

### Problem
Modern GPUs throttle clocks (DVFS) when power or heat limits are hit. Workloads alternating compute-bound matmuls with memory-bound ops (e.g. LM-head + cross-entropy) cause spiky power draw and throttling.

### Method
Split the workload into row-chunks so compute and memory phases interleave finely, smoothing power draw. Applied to the LM-head in training and to a matmul-softmax-matmul micro-benchmark.

### Expected behaviour
Paper-reported (not locally measured): DGX Spark micro-benchmark ~15% more FLOPS at 2 chunks, ~21.6% at 16; end-to-end Qwen2-0.5B training ~20% less time, ~14% less energy (LM-head microbatch 16 vs 64). L40S: ~2.5% time, ~1.3% energy gain. vLLM prefill: <=1.6%.

## Implementation

- **Input contract:** `mm_softmax_mm(a:(M,K), b:(K,N), chunks)`; `chunked_lm_head_ce(hidden:(T,H), weight:(V,H), targets:(T,), chunks)`
- **Output contract:** `(N,K)` accumulator tensor in input dtype; scalar float32 mean loss (differentiable)
- **Configuration:** `benchmark.py` CLI flags; full config (torch/CUDA/device versions, seed, args) stored in each JSONL record
- **Reproduction command:** `python benchmark.py --m 32768 --k 2048 --n 131072 --dtype bfloat16 --chunks 1 2 4 8 16 32 --warmup 10 --iters 100 --compile`
- **Tests:** `python -I -m pytest -q tests` (CPU; chunked == unchunked for outputs and grads, config validation)

## Results

| Source         | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|----------------|--------:|--------:|-----------:|------------:|-------:|-------|
| Paper-reported | n/a | 796 -> 776 ms (L40S, 4 chunks) | +15–21.6% FLOPS (Spark) | not reported | 2747 -> 2712 J (L40S) | M=32768,K=2048,N=131072 |
| Local baseline |  |  |  |  |  | chunks=1, TODO fill from your GPU |
| Local method   |  |  |  |  |  | best chunks, TODO |

## Differences and limitations

- Second matmul form is my reading of Listing 1 (contraction over the chunked dim, accumulated into a shared buffer); verify against the paper.
- NVML energy is an estimate; cross-check with wall-socket power / Nsight where possible.
- End-to-end LLMQ training and vLLM runs are not reproduced.
- Best chunk size is device/workload dependent; no auto-tuning here.

