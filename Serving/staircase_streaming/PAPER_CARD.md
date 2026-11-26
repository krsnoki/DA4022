# Paper implementation card: Staircase Streaming for Low-Latency Multi-Agent Inference

- **Paper / OpenReview:** https://openreview.net/forum?id=9VqvhSZwN6 · arXiv [2510.05059v1](https://arxiv.org/abs/2510.05059) (6 Oct 2025)
- **Authors:** Junlin Wang, Jue Wang, Zhen Xu, Ben Athiwaratkun, Bhuwan Dhingra, Ce Zhang, James Zou
- **Primary module:** `Serving/` (folder name pending maintainer approval; listed under "Profiling, systems and hardware-aware evaluation" / "latency and serving" in `PAPER_INDEX.md`)
- **Secondary modules:** `Performance_Metrics/` (latency protocol)
- **Owner / reviewer:** Nitish Kumar Singh, Ayush Agarwal, Kunj Aggarwal, Mithil Modi, Onkar Kawde, Pawar Snehil Sai
- **Status:** `partial`. Implemented, tested, and measured on 1× RTX 4090 (2026-11-24). See Results and Differences.
- **Upstream code / licence:** We found no code link in the paper (v1 HTML).
- **Backbone IDs:** see `configs/full.yaml` (`text.qwen2_5_1_5b_instruct`, `text.smollm2_1_7b_instruct`, `text.qwen2_5_0_5b_instruct`)
- **Dataset IDs and split:** `text.alpaca_eval.eval`, `text.arena_hard_v0_1.eval` (latency); `text.gsm8k.test` (quality)
- **Hardware assumptions:** We use Linux with NVIDIA GPU (≥ 16 GB recommended), running vLLM OpenAI-compatible servers. The mock backend runs anywhere on CPU.

## Research summary

### Problem
In multi-agent inference such as Mixture-of-Agents (MoA), several proposer models answer a query and an aggregator synthesizes their answers. The aggregator cannot start until every proposer has finished, so time-to-first-token (TTFT) is roughly the slowest proposer's full generation time plus the aggregator's prefill (paper Eq. 1). That is often many seconds.

### Method
Staircase streaming (paper Algorithm 1) divides proposer and aggregator outputs into chunks:
1. The aggregator starts as soon as each proposer has produced a small first chunk (`first_chunk_size` = 8 tokens). It does not wait for the `redundancy` slowest proposers for that first chunk.
2. For each later chunk *j*, it waits for proposer chunk *j* from all proposers and rebuilds its prompt with all proposal text so far (Table 2 template). It then continues its own partial answer for `C_A,j` tokens and streams them to the user.
3. Chunk sizes grow: 8, then 128, then 256 (proposers) / 128 (aggregator).

TTFT becomes the slowest *first chunk* plus prefill (Eq. 2). The cost is repeated aggregator prefills over a growing prompt, i.e. more prompt tokens. An optional prefix-cache variant (Table 6 template) appends new chunks at the end of the prompt so the earlier prefix can be reused from the KV cache.

### Expected behaviour (paper-reported, Table 1)
The paper's setup:
- Proposers: LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 and Qwen-1.5-7B-Chat, via the Together API.
- Aggregator: Gemma-2-9B-IT, on vLLM on 1× H100.
- Efficiency set: 64 Arena-Hard + 64 AlpacaEval prompts.

| Method (paper-reported) | Arena-Hard WR | AlpacaEval LC WR | TTFT (s) | TPS (tok/s) |
|---|---:|---:|---:|---:|
| MoA | 47.5 | 56.8 | 10.6 ± 3.4 | 28.3 ± 8.9 |
| MoA + staircase | 48.3 | 55.1 | 0.47 ± 0.1 | 43.4 ± 6.5 |
| MoA + staircase & prefix-cache | 46.9 | 53.0 | 0.45 ± 0.1 | 45.0 ± 7.0 |

The abstract claims "up to 93%" TTFT reduction; the MoA rows above give ≈ 95.6% (our calculation). The paper does not quantify the prompt-token overhead.

## Implementation

- **Input contract:** a query string. Proposers and the aggregator are model IDs served by a `Backend` (`backends.py`) that streams `TokenEvent(text, t, ...)` objects.
- **Output contract:** `MoAResult` (`timing.py`) containing:
  - the final text;
  - TTFT, end-to-end time, and TPS (aggregator tokens ÷ request→last-token time, as in the paper);
  - per-proposer and per-aggregator-call timings, with prefill vs decode;
  - per-chunk records;
  - prompt-token counts, plus cached-token counts where the server reports them.
- **Configuration:** `configs/full.yaml` (used for all reported results; `configs/default.yaml` has identical settings with a smaller, quicker workload). The staircase hyperparameters follow paper Appendix A: `first_chunk_size` 8, `second_chunk_size` 128, `chunk_size` 256, `aggregator_chunk_size` 128, `redundancy` 2.
- **Reproduction command:** We run the benchmark using the command: `python -m Serving.staircase_streaming.run_benchmark --config Serving/staircase_streaming/configs/full.yaml` (on a GPU machine with servers running; see README).
- **Tests:** `python -m pytest Serving/staircase_streaming/tests -q`

## Results

Headline numbers come from the **full-scale run**: `configs/full.yaml`, 64 AlpacaEval + 64 Arena-Hard v0.1 prompts (the paper's count) and the whole GSM8K test set.

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (MoA) | AH 47.5 / AE-LC 56.8 | TTFT 10.6 s | 28.3 tok/s | not reported | not reported | 7–9B models, Together API + 1× H100 |
| Paper-reported (MoA + staircase) | AH 48.3 / AE-LC 55.1 | TTFT 0.47 s | 43.4 tok/s | not reported | not reported | same setup |
| Local, RTX 4090 (MoA baseline) | GSM8K EM 0.661 (872/1319) | TTFT 4.28 s median / 6.00 s p95; E2E 6.11 s / 8.21 s | 66.1 tok/s | 18 725 MiB | not_measured | 128 prompts, 0 failures |
| Local, RTX 4090 (MoA + staircase, f8 r2) | GSM8K EM 0.671 (885/1319) | TTFT 0.097 s / 0.102 s p95; E2E 5.77 s / 7.18 s | 73.7 tok/s | 18 725 MiB | not_measured | 128 prompts, 0 failures; 3.77× aggregator prompt tokens |
| Local, RTX 4090 (MoA + staircase & prefix-cache, f8 r2) | not measured | TTFT 0.097 s / 0.101 s p95; E2E 5.65 s / 7.17 s | 73.3 tok/s | 18 725 MiB | not_measured | 128 prompts, 0 failures |

All local rows were measured on the same machine with the same prompts (seed 42), after 3 warm-up requests. Raw rows: `Performance_Metrics/staircase_streaming_rtx4090_full_2026-11-24.csv` (commit `2f5af63`; 4 430 requests, 0 failures).

- **Setup:** We use 1× NVIDIA GeForce RTX 4090 24 GB (vast.ai), driver 595.84 (CUDA 13.2), vLLM 0.34.0, PyTorch 2.14.0 (CUDA 13.0 build), bf16, FlashAttention 2. Three vLLM servers share the GPU. Run with `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Definitions:**
  - TTFT = request start → first aggregator token.
  - TPS = aggregator tokens ÷ (request start → last token), median over prompts (paper definition).
  - Peak memory = maximum of `nvidia-smi memory.used` sampled every 0.5 s over the whole run. It is device-wide and includes vLLM's pre-allocated KV pools, so it reflects `gpu_memory_utilization`, not the method.
- **TTFT reduction (default f8 r2):** 97.7 % lower median TTFT than the baseline (our calculation from the rows above; the paper reports ≈ 95.6 %). TPS is +11.5 % (paper: +53 %), median E2E −5.7 %.
- **Cost:** The staircase aggregator re-prefills 4.33 times per request on average. That consumes 3.77× the baseline's aggregator prompt tokens and 3.34× the total prompt tokens (proposers + aggregator). Median aggregator prefill time is 172 ms vs 34 ms for the baseline.
- **Sweep (median TTFT, ms; baseline 4276):**

  | first chunk \ redundancy | r = 0 | r = 1 | r = 2 |
  |---|---:|---:|---:|
  | 4 | 113.2 | 112.0 | 72.5 |
  | 8 | 175.1 | 174.0 | 97.2 |
  | 16 | 297.6 | 297.8 | 132.5 |
  | 32 | 545.8 | 545.8 | 220.7 |

  TTFT grows with the first-chunk size and falls with redundancy 2 (skipping the 2 slowest of 3 proposers for chunk 1). r = 1 barely helps; our interpretation, not separately measured, is that the two slower proposers finish their first chunk at about the same time. TPS ranged from 72.0 to 73.7 tok/s across the sweep.
- **Prefix-cache variant:** no measurable gain here (TTFT 96.6 vs 97.2 ms). vLLM's automatic prefix caching is on for every method, and the Table 2 layout already keeps a long common prefix. So the Table 6 layout adds little at these prompt lengths.
- **Quality (GSM8K exact match, all 1 319 test questions):** 0.661 baseline vs 0.671 staircase (±0.025 95 % CI each).
  - Per question: 755 both right, 317 both wrong, 117 only baseline right, 130 only staircase right.
  - We performed an exact McNemar test on the 247 disagreements: p = 0.445, so no statistically significant difference.
  - Staircase keeps answer quality while cutting TTFT by 97.7 %.
  - We did not run an LLM-as-judge, so the paper's Arena-Hard / AlpacaEval win rates are not reproduced.

Label `partial`: the TTFT reduction is reproduced (larger than the paper's) and quality is preserved on GSM8K. The TPS gain has the same direction but is smaller. The models are smaller (0.5–1.7B vs 7–9B), everything runs on one shared local GPU, and quality is GSM8K EM only.

## Differences and limitations

- **Smaller models:** The proposers are Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct and Qwen2.5-0.5B-Instruct, and the aggregator is Qwen2.5-1.5B-Instruct, instead of 7–9B models. Absolute quality and latency are not comparable to the paper.
- **All local:** All agents run on local vLLM servers on the same machine, often the same GPU. The paper's proposers ran on the Together API. Proposers and the aggregator can contend for one GPU, which may reduce the TPS gain.
- **Continuously streaming proposers:** proposers stream continuously in their own threads into thread-safe buffers instead of being restarted per chunk as in Algorithm 1. Chunk *j* is "ready" when a proposer's buffer reaches the cumulative boundary or the proposer has finished. One consequence: proposers are prefilled once, not once per chunk.
- **Redundancy only for the first chunk:** redundancy applies only to the first chunk, following the paper's Section 3.1 text; Algorithm 1 itself is ambiguous.
- **All text so far in Table 2 mode:** the aggregator prompt contains *all* proposer text received so far, which can exceed the chunk boundary. In prefix-cache mode, chunk blocks are capped at the boundary so earlier blocks never change.
- **Proposers cancelled at EOS:** When the aggregator hits EOS while proposers are still generating, the proposers are cancelled (HTTP request closed), so they don't slow down the next request. The paper doesn't specify this.
- **Table 6 wording:** the prefix-cache template is a meaning-level reconstruction. One sentence is paraphrased; see `prompts.py`.
- **Partial-answer continuation:** We use the chat template with `continue_final_message=True` (vLLM chat endpoint). The partial answer is re-tokenized at each chunk boundary.
- **Prefix caching:** We have enabled vLLM's automatic prefix caching for all methods (server-level setting). "Prefix-cache mode" changes only the prompt layout (Table 6), as in the paper.
- **Token counting:** chunk boundaries count streamed text deltas. vLLM normally streams one token per delta, but exact token counts come from server-reported `usage`.
- **Sampling settings are ours:** the paper does not report sampling settings or max lengths. We use proposer temperature 0.7 / top-p 0.9 with fixed seeds, a greedy aggregator, and a 512-token cap for both.
- **Quality metric:** GSM8K exact match (whole test set in the full run). No LLM-as-judge (Arena-Hard / AlpacaEval win rates are not reproduced).
- **Efficiency set:** `configs/full.yaml` uses the paper's 64 AlpacaEval + 64 Arena-Hard prompts (headline results). `configs/default.yaml` uses 32 + 32 for a quicker run.

## Reproduction checklist

- [x] We have verified the exact paper link and version (arXiv 2510.05059v1).
- [x] We have pinned the base model and dataset revisions (config + registry; used in the 2026-11-24 GPU run).
- [x] Baseline runs on the same setup (same GPU, servers, prompts, seeds)
- [x] We added tests and seed handling (mock backend + live vLLM smoke tests).
- [x] Resource and quality metrics reported (TTFT/E2E/TPS/prompt tokens/peak memory/GSM8K EM; energy not measured)
- [x] Licence/data terms checked (models Apache-2.0; GSM8K MIT; AlpacaEval CC-BY-NC-4.0; Arena-Hard-Auto Apache-2.0, per HF metadata on 2026-11-23)
