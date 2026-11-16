# Paper implementation card: DropKV: Decoupling Residual-Output Perturbation for Near-Optimal KV-Cache Eviction

- **Paper / OpenReview:** https://openreview.net/forum?id=MqfNzH3TVH (AdaptFM, ICML 2026 workshop); extended version on arXiv: https://arxiv.org/abs/2610.22130
- **Primary module:** `Attention/` (`Attention/methods/kv_eviction/`)
- **Secondary modules:** `Dataloaders/synthetic/`, `Backbones/Generative_Pretrained/`, `Performance_Metrics/`
- **Owner / reviewer:** Deval Singhal (@maidevalhoon), Jayant Kumar, Sonal Mangesh Wagh, Sristi Vats, Shourya Veer Singh
- **Status:** partial-reproduction
- **Upstream code / licence:** https://github.com/AozhongZhang/DropKV, Apache-2.0 (also available as `DropKVPress` in NVIDIA/kvpress). Not used here: this implementation is written from Algorithm 1 of the paper in plain PyTorch.
- **Backbone IDs:** `text.qwen2_5_0_5b_instruct.self_supervised` ([model card](../../../Backbones/Generative_Pretrained/qwen2_5_0_5b_instruct.md))
- **Dataset IDs and split:** `text.passkey_retrieval.synthetic`, split `synthetic` ([dataset card](../../../Dataloaders/synthetic/passkey_retrieval.md))
- **Hardware assumptions:** any device PyTorch runs on. Tests run on CPU. Benchmark numbers below are from one NVIDIA Tesla T4 (15 GB, Colab), fp16. No custom kernels.

## Research summary

### Problem

During generation the KV cache grows linearly with the prompt length and, for long prompts, takes more memory than the model weights. Eviction methods keep only a fraction of the cached tokens after prefill. Most of them rank tokens by attention weight alone and ignore the value vectors, so a token with moderate attention but an unusual value can be thrown away even though removing it changes the attention output a lot. Methods that do model the output change have to solve a subset-selection problem that is NP-hard.

### Method

For one query with attention weights `p` and output `a = sum_i p_i v_i`, removing a set `J` of tokens shifts the output by

    Delta(J) = sum_{i in J} p_i (a - v_i) / (1 - sum_{i in J} p_i)        (Lemma 1)

DropKV does not search over sets. It gives every token the shift that removing it alone would cause, and sums the squared shift over the last `w` queries of the prompt (the observation window):

    score_j = sum_t (p_j^(t) / (1 - p_j^(t)))^2 * ||a^(t) - v_j||^2

Scores are max-pooled over neighbouring positions (kernel 11), the last `w` tokens are always kept, and the tokens with the lowest scores are evicted. The paper proves that this greedy choice is within a constant factor of the optimal set when the evicted tokens have small attention weights and their residuals `a - v_i` lie in a common cone.

The difference from SnapKV is only the factor `||a - v_j||^2` (and the square): a token whose value is close to the current output costs nothing to remove, however much attention it gets.

### Expected behaviour

Reported in the paper (A100-80GB, RULER at 128K tokens, 5% cache budget): average accuracy 82.00 on Llama-3.1-8B-Instruct, 83.77 on Qwen2.5-7B-Instruct-1M and 85.40 on MegaBeam-Mistral-7B-512K, which is 2.4 to 7.7 points above SnapKV and 2.35 to 4.03 points above the best other baseline. On LongBench at 5% budget: 45.51, 42.66 and 36.22. The fused Triton scoring kernel is reported 19.8x faster than a PyTorch reference at 128K tokens and gives up to 9.9% faster prefill.

None of these numbers were measured by us. Our local numbers are in the Results table and come from a much smaller model, shorter prompts and a simpler task.

## Implementation

- **Input contract:** queries of the observation window `q [B, H, W, D]`, cached keys and values `k, v [B, H, N, D]`, any float dtype (scoring is done in float32). In the Hugging Face path the queries are recorded by forward hooks on each attention layer (`q_proj` output of the last `W` positions with RoPE applied) and grouped-query heads are folded onto their KV head. Batch size 1.
- **Output contract:** scores `[B, H, N]` float32; kept indices `[B, H, n_keep]` sorted, always containing the last `W` positions; the cache is modified in place to keys/values of shape `[B, H, n_keep, D]`. Decoding continues with the original position ids.
- **Configuration:** [`Attention/configs/dropkv/qwen2_5_0_5b.yaml`](../../configs/dropkv/qwen2_5_0_5b.yaml) (30% and 10% budgets) and [`qwen2_5_0_5b_low_budget.yaml`](../../configs/dropkv/qwen2_5_0_5b_low_budget.yaml) (5%, 2%, 1%). Important fields: `budgets` (fraction of positions kept per head), `window` (8), `kernel_size` (11), `methods`, `context_tokens`, `model_revision`.
- **Reproduction command:** `python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/qwen2_5_0_5b.yaml`
- **Tests:** `python -m pytest Attention/tests/test_dropkv.py -q` (CPU, no downloads)

Code map:

| File | Content |
|---|---|
| `Attention/methods/kv_eviction/scoring.py` | window attention, DropKV / SnapKV / random scores, pooling and top-k selection |
| `Attention/methods/kv_eviction/eviction.py` | query recording hooks, cache compression, prefill and greedy decode on the compressed cache |
| `Attention/methods/kv_eviction/perturbation.py` | Lemma 1, exact objective, brute-force optimum (used for checks) |
| `Attention/tests/test_dropkv.py` | correctness tests |
| `Attention/experiments/dropkv/run_benchmark.py` | benchmark driver |

## Results

Local numbers: one NVIDIA Tesla T4 (15 GB, Colab), fp16, torch 2.14.1+cu130, transformers 5.23.0, batch size 1, 20 prompts per configuration, 1 warm-up request, greedy decoding of up to 12 tokens. Quality is passkey accuracy. Latency is median end-to-end time per request (prefill + scoring + eviction + decoding), throughput is decode tokens/s, peak memory is `torch.cuda.max_memory_allocated` during prefill. Raw rows: [`results/results.csv`](results/results.csv), [`results_low_budget/results.csv`](results_low_budget/results.csv), environment in the `environment.json` next to each.

Prompts of about 4000 tokens (4006 after the chat template):

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | 82.00 / 83.77 / 85.40 avg. RULER score | not given as absolute | 1.75-8.22x decode throughput vs dynamic cache | scoring scratch ~17 MB vs ~9 GB | not reported | Llama-3.1-8B / Qwen2.5-7B-1M / Mistral-7B-512K, 128K tokens, 5% budget, A100-80GB. Different models, task and length: not comparable with the rows below |
| Local baseline (full cache) | 1.00 | 402.3 ms | 36.4 tok/s | 2173.6 MB | not measured | KV cache 46.95 MB |
| Local method, DropKV 30% | 1.00 | 434.4 ms | 32.2 tok/s | 2178.0 MB | not measured | KV cache 14.09 MB |
| Local method, DropKV 10% | 1.00 | 445.4 ms | 33.2 tok/s | 2178.0 MB | not measured | KV cache 4.70 MB |
| Local method, DropKV 5% | 1.00 | 440.9 ms | 32.4 tok/s | 2178.0 MB | not measured | KV cache 2.34 MB |
| Local method, DropKV 2% | 1.00 | 449.3 ms | 30.7 tok/s | 2178.0 MB | not measured | KV cache 0.94 MB |
| Local method, DropKV 1% | 0.45 | 454.8 ms | 29.3 tok/s | 2178.0 MB | not measured | KV cache 0.47 MB |

Comparison with the other eviction rules at the same budgets (accuracy, 4000 tokens):

| Budget | 30% | 10% | 5% | 2% | 1% |
|---|---:|---:|---:|---:|---:|
| DropKV | 1.00 | 1.00 | 1.00 | 1.00 | 0.45 |
| SnapKV | 1.00 | 1.00 | 1.00 | 1.00 | 0.45 |
| Random | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

At about 2000 tokens the picture is the same: full cache, DropKV and SnapKV are at 1.00 for 30% and 10%, random is at 0.05 and 0.00.

Cost of the method at 4000 tokens:

| | Full cache | DropKV 10% | SnapKV 10% |
|---|---:|---:|---:|
| Prefill, median (ms) | 264.5 | 299.2 | 286.2 |
| Decode, median (ms/token) | 27.1 | 28.6 | 29.6 |
| KV cache (MB) | 46.95 | 4.70 | 4.70 |
| Peak memory during decode (MB) | 2167.8 | 2124.8 | 2124.8 |

What these numbers show, and what they do not:

- Eviction guided by the window attention works on this task: the cache can be cut to 2% (46.95 MB to 0.94 MB) with no loss in accuracy, while random eviction at any budget destroys the answer.
- **DropKV and SnapKV cannot be told apart here.** They have the same accuracy at every budget, including the point where both break (1%). The advantage over SnapKV reported in the paper is therefore not reproduced by this experiment. A single passkey that the final question attends to directly is too easy to separate an attention-only score from a value-aware one; that would need tasks such as RULER or LongBench at much longer contexts.
- Scoring is not free: prefill is about 13% slower with DropKV and about 8% slower with SnapKV than with no eviction. The difference between the two is the value-residual term.
- Decode latency did not improve. With a 0.5B model and a 4K prompt on a T4 the KV cache is under 50 MB, so decoding is not limited by cache size. The throughput gains in the paper come from 32K-128K contexts and large batches, which were not tested.
- Peak memory during prefill does not change, because the whole prompt is processed before anything is evicted. Decode-time peak memory drops by 43 MB, which matches the reduction of the cache itself.
- The full-cache baseline was measured in both runs (402.3 ms and 423.1 ms median), so differences of a few percent in latency are within run-to-run variation.

## Differences and limitations

- **Model scale:** Qwen2.5-0.5B-Instruct instead of the 7-8B models of the paper.
- **Context length:** about 2K and 4K tokens instead of 16K-128K.
- **Task:** synthetic passkey retrieval instead of RULER and LongBench, so quality numbers are not comparable with the paper's tables. On this task DropKV and SnapKV give identical accuracy, so the paper's improvement over SnapKV is not reproduced.
- **Kernels:** the fused Triton kernels are not implemented. Our scoring uses the same `||a||^2 + ||v||^2 - 2<a, v>` expansion, which avoids the `[W, N, D]` temporary, but it still materialises the `[W, N]` attention weights. The 19.8x kernel speedup is therefore not reproduced.
- **Baselines:** full cache, SnapKV and random only. AnDPro, CriticalKV, CAKE, PyramidKV, KeyDiff and the adaptive head budget variant (AdaDropKV) are not included.
- **Memory:** eviction happens after the prompt has been processed once, so peak memory during prefill does not go down. The saving is in the resident KV cache and in decode-time memory, which are reported separately.
- **Batching:** batch size 1, no padding support.
- **Supported models:** Hugging Face decoder-only models whose attention layers expose `q_proj` and receive rotary `position_embeddings` (Llama and Qwen2 families).
- **Energy:** not measured.

## Reproduction checklist

- [x] Exact paper link and version verified
- [x] Base model and dataset revisions pinned
- [x] Baseline runs on the same setup
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported
- [x] Licence/data terms checked
