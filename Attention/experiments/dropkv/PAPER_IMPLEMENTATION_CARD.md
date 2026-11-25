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
- **Configuration:** [`Attention/configs/dropkv/qwen2_5_0_5b.yaml`](../../configs/dropkv/qwen2_5_0_5b.yaml) (30% and 10% budgets) and [`qwen2_5_0_5b_low_budget.yaml`](../../configs/dropkv/qwen2_5_0_5b_low_budget.yaml) (5%, 2%, 1%); [`perturbation.yaml`](../../configs/dropkv/perturbation.yaml) for the perturbation run. Important fields: `budgets` (fraction of positions kept per head), `window` (8), `kernel_size` (11), `methods`, `context_tokens`, `model_revision`.
- **Reproduction command:** `python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/qwen2_5_0_5b.yaml`
- **Output-perturbation command:** `python -m Attention.experiments.dropkv.run_perturbation --config Attention/configs/dropkv/perturbation.yaml` (CPU, about 20 minutes on a laptop)
- **Tests:** `python -m pytest Attention/tests/test_dropkv.py -q` (CPU, no downloads, 26 tests)

Code map:

| File | Content |
|---|---|
| `Attention/methods/kv_eviction/scoring.py` | window attention, DropKV / SnapKV / random scores, pooling and top-k selection |
| `Attention/methods/kv_eviction/eviction.py` | query recording hooks, cache compression, prefill and greedy decode on the compressed cache |
| `Attention/methods/kv_eviction/perturbation.py` | Lemma 1, exact objective, brute-force optimum (used for checks) |
| `Attention/tests/test_dropkv.py` | correctness tests |
| `Attention/experiments/dropkv/run_benchmark.py` | benchmark driver (accuracy, latency, memory) |
| `Attention/experiments/dropkv/run_perturbation.py` | output-perturbation driver (attention-output shift and next-token KL) |

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

### Output perturbation

Measured with [`run_perturbation.py`](run_perturbation.py) on a MacBook CPU (Apple M1, 8 GB), float32, torch 2.15.0, transformers 5.23.0, 20 prompts per row. For every prompt the cache is built once and each method is applied to the same cache. `attn_shift` is the relative change of the attention output of the 8 window queries, `||a_kept - a_full|| / ||a_full||`, averaged over queries, heads and layers. `next_token_kl` is the KL divergence of the next-token distribution at the first decode step against the full cache. Lower is better for both. Raw rows: [`results_perturbation/perturbation.csv`](results_perturbation/perturbation.csv) and [`perturbation_per_layer.csv`](results_perturbation/perturbation_per_layer.csv).

Prompts of about 4000 tokens (4006 after the chat template):

| Budget | DropKV shift | SnapKV shift | Random shift | DropKV KL | SnapKV KL | Random KL | DropKV lower than SnapKV on (shift / KL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.0302 | 0.0291 | 1.62 | 2e-05 | 1.8e-05 | 2.67 | 0% / 35% of prompts |
| 10% | 0.0603 | 0.0588 | 2.38 | 5.8e-05 | 5.2e-05 | 8.15 | 0% / 20% of prompts |
| 5% | 0.0802 | 0.0785 | 2.65 | 8.3e-05 | 8.2e-05 | 9.06 | 0% / 50% of prompts |
| 2% | 0.121 | 0.116 | 2.92 | 0.000333 | 0.000407 | 11.5 | 0% / 45% of prompts |
| 1% | 0.207 | 0.197 | 3.05 | 0.031 | 0.073 | 12.3 | 0% / 80% of prompts |

Prompts of about 2000 tokens (2014 after the chat template):

| Budget | DropKV shift | SnapKV shift | Random shift | DropKV KL | SnapKV KL | Random KL | DropKV lower than SnapKV on (shift / KL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.0278 | 0.0269 | 1.59 | 9e-06 | 7e-06 | 2.3 | 0% / 30% of prompts |
| 10% | 0.059 | 0.0571 | 2.4 | 3.2e-05 | 2.9e-05 | 8.22 | 0% / 35% of prompts |
| 5% | 0.091 | 0.0883 | 2.7 | 0.000193 | 0.000216 | 11.2 | 0% / 55% of prompts |
| 2% | 0.212 | 0.199 | 2.98 | 0.0106 | 0.13 | 12.7 | 0% / 70% of prompts |
| 1% | 0.39 | 0.388 | 3.1 | 6.97 | 7.44 | 12.4 | 20% / 50% of prompts |

Caveats: `attn_shift` is measured on the same window queries that were used for scoring, `next_token_kl` is not. Each row averages 20 synthetic prompts, so a difference of a few percent is small compared with the spread between prompts. KL values at 30% to 5% are of the order of 1e-5 to 1e-4 and are close to numerical noise.

What the perturbation numbers show:

- Both attention-guided rules change the output far less than random eviction, for example at 10% the shift is about 0.06 against 2.4 for random.
- On the quantity that the paper derives its score from, the attention-output shift, DropKV is slightly **higher** than SnapKV in all 10 settings (2 context lengths x 5 budgets), by 0.5% to 6.5%. DropKV is lower on 0% of the prompts in 9 of the 10 settings and on 20% in the remaining one.
- On next-token KL, DropKV has the lower mean in 5 of the 10 settings, mostly at the smallest budgets (for example 0.031 against 0.073 at 4000 tokens and 1%, and 0.011 against 0.130 at 2000 tokens and 2%). The share of prompts on which it is lower ranges from 20% to 80% with an average of 47%, which is no consistent difference.
- Taken together, this experiment does not show the advantage of DropKV over SnapKV that the paper reports. We did not investigate why. Possible factors are the small model, the short contexts, the shared max-pooling and protected window, and the fact that the guarantee in the paper is for the exact single-query objective while our measurement averages over heads, layers and a multi-query window.

What these numbers show, and what they do not:

- Eviction guided by the window attention works on this task: the cache can be cut to 2% (46.95 MB to 0.94 MB) with no loss in accuracy, while random eviction at any budget destroys the answer.
- **DropKV and SnapKV cannot be told apart by accuracy.** They have the same accuracy at every budget, including the point where both break (1%). A single passkey that the final question attends to directly is too easy to separate an attention-only score from a value-aware one. The output-perturbation measurement above looks at the effect of eviction directly and does not show a consistent advantage of DropKV either, so the advantage over SnapKV reported in the paper is not reproduced by these experiments.
- Scoring is not free: prefill is about 13% slower with DropKV and about 8% slower with SnapKV than with no eviction. The difference between the two is the value-residual term.
- Decode latency did not improve. With a 0.5B model and a 4K prompt on a T4 the KV cache is under 50 MB, so decoding is not limited by cache size. The throughput gains in the paper come from 32K-128K contexts and large batches, which were not tested.
- Peak memory during prefill does not change, because the whole prompt is processed before anything is evicted. Decode-time peak memory drops by 43 MB, which matches the reduction of the cache itself.
- The full-cache baseline was measured in both runs (402.3 ms and 423.1 ms median), so differences of a few percent in latency are within run-to-run variation.

## Differences and limitations

- **Model scale:** Qwen2.5-0.5B-Instruct instead of the 7-8B models of the paper.
- **Context length:** about 2K and 4K tokens instead of 16K-128K.
- **Task:** synthetic passkey retrieval instead of RULER and LongBench, so quality numbers are not comparable with the paper's tables. On this task DropKV and SnapKV give identical accuracy, and the output-perturbation measurement shows no consistent advantage for DropKV, so the paper's improvement over SnapKV is not reproduced.
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
