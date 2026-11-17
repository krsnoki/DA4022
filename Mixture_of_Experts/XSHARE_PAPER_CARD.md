# Paper implementation card: XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference

- **Paper / OpenReview:** https://openreview.net/forum?id=KmY5XUgEYe
- **Authors:** Daniil Vankov, Nikita Ivkin, Jaime Campos Salas, Kyle R. Ulrich, Xiang Song, Ashish Khetan, George Karypis
- **Version implemented:** the AdaptFM workshop version (ICML 2026, poster), as linked from the [official accepted-paper list](https://adaptfm.gitlab.io/accepted-papers). PDF SHA-256 `61bd00193ceafc8a3944634884526b18e2ebb9b88aebcac0e2fc1c71391b0103`. An earlier version, [arXiv 2602.07265v1](https://arxiv.org/abs/2602.07265) (February 2026), reports different headline numbers ("up to 30%" fewer activated experts and "up to 14%" higher throughput with speculative decoding); the numbers below refer to the workshop version.
- **Primary module:** `Mixture_of_Experts/`
- **Secondary modules:** `Backbones/` (wrapper, registry, model card), `Dataloaders/` (GSM8K loader, registry, dataset card), `Performance_Metrics/` (result schema)
- **Owner / reviewer:** @AyushWani11 / not assigned
- **Issue:** https://github.com/krsnoki/DA4022/issues/5
- **Status:** partial-reproduction
- **Upstream code / licence:** none found. Checked the paper text, the arXiv page, the ICML page and GitHub on 2026-11-17. Reimplemented from the paper.
- **Backbone IDs:** `text.olmoe-1b-7b-0125-instruct` (64 experts, 8 per token, 16 layers; `Backbones/registry.yaml`)
- **Dataset IDs and split:** `text.gsm8k.test`, three disjoint subsets of 200 test questions (600 questions; `Dataloaders/registry.yaml`)
- **Hardware assumptions:** one CUDA GPU with about 16 GB free for the experiments. The core algorithms and all unit tests run on a CPU. Results were measured on one NVIDIA RTX PRO 4500 Blackwell (32 GB), driver 570.211.

## Research summary

### Problem
In a Mixture-of-Experts (MoE) model each token uses only `k` experts. A serving
batch holds many tokens, and together they use most of the experts. During
decoding the cost is dominated by reading expert weights from memory, so a
batch that touches most experts loses much of the benefit of sparsity.
Speculative decoding makes this worse, because each request contributes several
tokens per step.

### Method
At each MoE layer and each decode step, XShare picks one expert set `S` for the
whole batch. Each token is then routed to its top-`k` experts inside `S`. There
is no retraining and no calibration data.

- **Batch-aware selection (Alg. 1-2).** Warm-up: keep every token's top-`k0`
  experts. Greedy step: add the `b` experts with the highest gating score summed
  over the batch. The objective is a plain sum over experts, so this greedy
  choice is optimal for it. With `b = 0` the method is OEA (Oncescu et al.,
  2025), the paper's main comparison.
- **Speculative-decoding variant (Alg. 3-4).** Run the selection per request
  over that request's tokens, take the union, then run the global greedy step.
- **Expert-parallel variant (Alg. 5-6).** Experts are spread over GPUs. Select
  round-robin, one expert per GPU per pass, to limit the load on the busiest GPU.

### Expected behaviour
Paper-reported, GPT-OSS-120B (128 experts, 4 per token) on H200 GPUs with vLLM:

- No speculative decoding: +14-23% output tokens/s at concurrency 8-32, within
  1 point of baseline accuracy.
- With Eagle3 speculative decoding: +25-37% at concurrency 4-16.
- Mixed workloads: the fixed-budget greedy keeps accuracy where OEA with
  `k0 = 1` collapses.
- Expert parallel on DeepSeek-R1: 73% fewer active experts and 3x lower peak
  GPU load.

These numbers come from a different model, serving stack and hardware. They
are not reproduced here; see "Results" for what we measured.

## Implementation

| Part | File |
|---|---|
| Core algorithms (Alg. 1-6), routing, diagnostics | `routing/xshare/selection.py` |
| Hook for Hugging Face MoE models (decode steps only) | `execution_layouts/xshare_hf.py` |
| Backbone wrapper (load, preprocess, forward, metadata) | `Backbones/Generative_Pretrained/hf_causal_lm.py` |
| GSM8K loader (subsets, batching, scoring) | `Dataloaders/evaluation/gsm8k.py` |
| Sweep configuration | `configs/xshare/gsm8k_sweep.yaml` |
| Sweep runner (all measurements) | `experiments/xshare/run_sweep.py` |
| Summary table and paired statistics | `experiments/xshare/summarize.py` |
| Plots | `experiments/xshare/plot_results.py` |
| GPU smoke test | `experiments/xshare/smoke_test.py` |
| Activated experts: formula, simulation, measured | `experiments/xshare/expected_activation.py` |
| Simulations of Alg. 3-6 on captured router scores | `experiments/xshare/simulate_variants.py` |
| Tests | `tests/test_xshare_selection.py`, `tests/test_xshare_hf.py`, `Dataloaders/tests/` |

- **Input contract:** router logits `[n_tokens, n_experts]` (float) for one MoE
  layer at one decode step. Optional: `valid_mask [n_tokens]` (bool, false for
  padding or finished sequences), `request_ids [n_tokens]` (int),
  `expert_groups [n_experts]` (int, GPU of each expert). Selection runs in
  float32 whatever the model dtype.
- **Output contract:** `keep_mask [n_experts]` (bool), `indices [n_tokens, k]`
  (int), `weights [n_tokens, k]` (float), and diagnostics: kept experts,
  activated experts, per-expert load, share of score kept, per-group load.
  Every token is always routed to exactly `k` experts (the kept set is topped
  up to at least `k`), so no token is ever dropped.
- **Configuration:** `configs/xshare/gsm8k_sweep.yaml` lists the backbone,
  dataset, subsets, seed, generation settings, energy settings and every
  method. `calibration_dataset: null`.
- **Reproduction command** (from the repository root; resumes if interrupted):

  ```bash
  python -m Mixture_of_Experts.experiments.xshare.run_sweep \
      --config Mixture_of_Experts/configs/xshare/gsm8k_sweep.yaml \
      --out-dir Mixture_of_Experts/experiments/xshare/results/gsm8k
  ```

  Then, from `Mixture_of_Experts/experiments/xshare/`, the tables, plots and
  simulations:

  ```bash
  python -m Mixture_of_Experts.experiments.xshare.summarize results/gsm8k
  python -m Mixture_of_Experts.experiments.xshare.plot_results results/gsm8k/summary.csv --out-dir results/plots
  python -m Mixture_of_Experts.experiments.xshare.simulate_variants --logits results/gsm8k/router_logits_c8.pt results/gsm8k/router_logits_c32.pt --out-dir results/simulations
  ```

- **Install:** see `requirements-xshare.txt` (exact versions).
- **Tests:** `python -m pytest Mixture_of_Experts/tests Dataloaders/tests -q` (595 tests, all passing at commit `6054781`;
  CPU only; the dataset and tokenizer tests skip unless those are cached locally).
- **Environment used:** Ubuntu 24.04, Python 3.12.3, torch 2.13.0+cu129,
  transformers 5.19.0 (attention `sdpa`, experts `grouped_mm`), code tree `c984e3decbd2dfa3217c5d65f2aaa4026df04fce`, commit
  `60547812a18a626202ce040b18d3dedf0620ec1c`.

## Results

Locally measured unless marked "paper-reported". OLMoE-1B-7B-0125-Instruct,
bfloat16, greedy decoding, up to 320 new tokens, GSM8K test. Every
configuration answered the same 600 questions (three disjoint subsets of 200,
one run each). Files: `experiments/xshare/results/gsm8k/` (one JSON per run
with every generation and every decode-step time, `results.csv` in the
`Performance_Metrics` schema, `summary.csv` with the statistics below) and
`experiments/xshare/results/plots/`.

How to read the columns:

- **Quality:** GSM8K exact match (flexible answer extraction), pooled over 600
  questions, with a 95% Wilson interval.
- **p:** exact McNemar test against the baseline on the same questions. It
  counts only the questions where the two disagree. Above 0.05, the accuracy
  change is indistinguishable from noise.
- **Latency:** median wall time of one decode step for the whole batch.
- **Throughput:** decode tokens per second, mean ± standard deviation over the
  three subsets. Speed-up is computed per subset against the baseline.
- **Energy:** joules per generated token from whole-board GPU power
  (`nvidia-smi`, 100 ms samples, prefill and decode). In brackets: above the
  idle power, which was 34.5 W (batch 8) and 38.5 W (batch 32).
- **Experts:** mean experts activated per MoE layer per decode step, of 64.

| Source | Batch | Method | Quality | p | Latency | Throughput | Peak memory | Energy (J/token) | Experts |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Paper-reported | 16-32 | OEA `k0=2` (= k/2), GPT-OSS-120B, H200, vLLM | within 0.5 pt of baseline | n/a | not reported | +20-23% | not reported | not reported | n/a |
| Local baseline | 8 | unpruned | 69.8% [66.0, 73.4] | | 24.0 ms | 214 ± 3 | 13.5 GiB | 0.474 (0.312) | 25.1 |
| Local method | 8 | XShare keep-all (overhead check) | 69.8%, identical outputs | 1.00 | 28.0 ms | 182 ± 3 (−14.6%) | 13.5 GiB | 0.509 (0.320) | 25.1 |
| Local method | 8 | OEA `k0=1` | 42.3% (−27.5 pt) | <0.001 | 21.2 ms | 221 ± 4 (+3.2%) | 13.4 GiB | 0.299 (0.142) | 8.0 |
| Local method | 8 | OEA `k0=2` | 50.8% (−19.0 pt) | <0.001 | 21.6 ms | 223 ± 3 (+4.4%) | 13.4 GiB | 0.301 (0.145) | 9.6 |
| Local method | 8 | OEA `k0=4` | 69.5% (−0.3 pt) | 0.92 | 23.7 ms | 211 ± 1 (−1.2%) | 13.4 GiB | 0.348 (0.184) | 14.6 |
| Local method | 8 | XShare `k0=1, b=8` | 65.3% (−4.5 pt) | 0.017 | 23.2 ms | 207 ± 2 (−3.0%) | 13.4 GiB | 0.337 (0.170) | 12.0 |
| Local method | 8 | XShare `k0=2, b=8` | 67.8% (−2.0 pt) | 0.26 | 24.6 ms | 205 ± 4 (−4.2%) | 13.4 GiB | 0.361 (0.192) | 15.7 |
| Local method | 8 | XShare `k0=2, b=16` | 69.3% (−0.5 pt) | 0.82 | 26.6 ms | 190 ± 2 (−11.3%) | 13.5 GiB | 0.428 (0.246) | 20.6 |
| Local method | 8 | XShare `k0=4, b=8` | 69.3% (−0.5 pt) | 0.82 | 26.5 ms | 193 ± 5 (−9.6%) | 13.4 GiB | 0.417 (0.238) | 21.1 |
| Local baseline | 32 | unpruned | 69.0% [65.2, 72.6] | | 34.0 ms | 532 ± 16 | 15.1 GiB | 0.232 (0.159) | 41.7 |
| Local method | 32 | XShare keep-all (overhead check) | 69.0%, identical outputs | 1.00 | 38.0 ms | 474 ± 14 (−10.9%) | 15.1 GiB | 0.239 (0.157) | 41.7 |
| Local method | 32 | OEA `k0=1` | 34.7% (−34.3 pt) | <0.001 | 24.5 ms | 647 ± 28 (+21.7%) | 15.1 GiB | 0.149 (0.089) | 11.5 |
| Local method | 32 | OEA `k0=2` | 61.3% (−7.7 pt) | <0.001 | 26.9 ms | 637 ± 22 (+19.9%) | 15.1 GiB | 0.155 (0.093) | 17.8 |
| **Local method** | **32** | **OEA `k0=4`** | **69.3% (+0.3 pt)** | **0.91** | **30.9 ms** | **584 ± 10 (+10.0%)** | **15.1 GiB** | **0.178 (0.112)** | **27.9** |
| Local method | 32 | XShare `k0=1, b=8` | 66.0% (−3.0 pt) | 0.095 | 27.7 ms | 597 ± 9 (+12.4%) | 15.1 GiB | 0.164 (0.098) | 18.2 |
| Local method | 32 | XShare `k0=2, b=8` | 67.5% (−1.5 pt) | 0.41 | 30.5 ms | 559 ± 12 (+5.3%) | 15.1 GiB | 0.179 (0.110) | 25.4 |
| Local method | 32 | XShare `k0=2, b=16` | 70.0% (+1.0 pt) | 0.59 | 33.1 ms | 529 ± 16 (−0.6%) | 15.1 GiB | 0.197 (0.123) | 31.9 |
| Local method | 32 | XShare `k0=4, b=8` | 67.5% (−1.5 pt) | 0.36 | 33.9 ms | 522 ± 14 (−1.8%) | 15.1 GiB | 0.202 (0.128) | 34.7 |

Request latency (time from a batch's start to a question's final token) and
prefill time are in `summary.csv` and the JSON files. Request latency also
depends on how long each answer is, which changes when pruning changes the
answer, so it is not a clean speed measure.

What the measurements support:

1. **The paper's main claim holds in direction, at about half the size.** At
   batch 32, warm-up only with half of each token's experts (OEA `k0=4`, the
   analogue of the paper's `k0=2` with `k=4`) is 10.0% faster, uses 23% less
   energy per token, and leaves accuracy unchanged (+0.3 pt, p = 0.91). The
   paper reports +20-23% for the equivalent setting.
2. **The batch-level budget rescues aggressive warm-ups.** With `k0=1`, OEA
   loses 27-34 points. Adding `b=8` brings the loss to 3.0-4.5 points (66.0%
   against 34.7% at batch 32). This matches the paper's claim that OEA with a
   small `k0` collapses and the greedy budget prevents it.
3. **With a generous warm-up, the greedy budget adds cost but no accuracy.**
   OEA `k0=4` already preserves accuracy, and adding `b=8` to it only makes
   it slower. In our setup the best configuration is warm-up only.
4. **No configuration is faster at batch 8.** With about 5 live sequences per
   step the baseline activates only 25 of 64 experts, so there is little left
   to prune. The 4.0 ms selection overhead outweighs the saving.
5. **Pruning always lowers energy per token,** at both batch sizes, even where
   throughput falls: fewer expert weights are read per step.
6. **Selection overhead: 4.0 ms per decode step** at both batch sizes (keep-all
   XShare against the baseline). Keep-all XShare produced exactly the
   baseline's answer on all 600 questions at both batch sizes, which confirms
   the routing replacement is exact.
7. **Peak memory does not change:** all expert weights stay loaded. The saving
   is in weights read per step, not weights stored.

### Other experiments

- **Activated experts, formula vs measured**
  (`results/expected_activation.csv`). With uniform random routing the
  simulation matches the paper's formula `N(1-(1-k/N)^B)` to within 0.1 expert
  for 64 experts and `k = 8`. The real model activates fewer, because tokens
  in a batch share expert preferences: 33.4 for 8 tokens (formula 42.0) and
  53.6 for 32 tokens (formula 63.1), measured on unpruned decode steps.
- **Speculative-decoding variant, simulated (approximate).** No draft model was
  run: each request contributes 4 consecutive true tokens from the captured
  router scores. Consecutive tokens of one request share 3.0 of their top-8
  experts, against 2.0 for tokens of different requests (1.5x; the paper
  reports 2-3x). At 128 tokens per step, per-request selection keeps more of
  each token's top-8 gate weight than global selection with the same number of
  experts: 95.7% against 94.9% with about 40 experts kept, and 98.4% with 46
  kept against 97.2% with 48 kept. At 32 tokens per step it is not better
  (86.5% with 22 kept against 88.2% with 24 kept). This matches the paper,
  where the per-request variant pays off at large effective batches. No
  throughput or accuracy was measured for this variant.
- **Expert-parallel variant, simulated (approximate).** Experts are assigned to
  2, 4 or 8 virtual GPUs by `expert index mod G`; there is one physical GPU.
  With 8 groups and 32 sequences, the busiest group's active experts fall from
  7.75 (unpruned) to 4.10 with the GPU-aware selection at 2 experts per group
  (18 kept), against 6.11 for global selection with 33 kept. The GPU-aware sets
  are smaller, so they retain less gate weight (76% and 85% against 93%). No
  latency or communication cost was measured.

## Differences and limitations

- **Different model.** OLMoE-1B-7B (64 experts, 8 per token) instead of
  GPT-OSS-120B (128 experts, 4 per token). The paper's budgets do not transfer
  directly: `k0` must be read relative to `k`.
- **Different serving stack.** Hugging Face transformers in plain PyTorch with
  static batches, instead of vLLM with a fused kernel and continuous batching.
  We first tried `openai/gpt-oss-20b` in vLLM. vLLM release builds from 0.11.1
  need a CUDA 12.9 or newer toolchain, the NVIDIA driver on the machine we used
  (570.211) accepts only CUDA 12.8, and loading failed with
  `cudaErrorUnsupportedPtxVersion` (see `Backbones/Generative_Pretrained/gpt_oss_20b.md`).
- **Selection overhead.** The paper builds the expert mask inside the MoE
  kernel. Our selection runs as separate PyTorch operations and costs 4.0 ms
  per decode step (measured). Every pruned configuration pays it.
- **Expert kernel.** transformers ran the experts with its `grouped_mm`
  implementation. We did not measure how much of the weight-read saving this
  kernel turns into time.
- **Static batching.** Finished sequences stay in the batch as padding and do
  not vote in the selection. On average 5.1 of 8 and 17.4 of 32 sequences were
  live per step, so the effective batch is smaller than the nominal one.
- **Evaluation scope.** One dataset (GSM8K), 600 questions, one run per subset.
  Throughput spread over subsets mixes run-to-run noise with differences
  between question sets. No AIME, GPQA, ShareGPT or HumanEval, and no
  mixed-workload scenario.
- **Energy.** Whole-board power from `nvidia-smi` every 100 ms, integrated over
  prefill and decode. It includes the board's idle draw, which is also
  reported. Other processes on the machine (desktop sessions) draw a small
  amount of GPU power and are inside the idle figure. Sampling at 100 ms over
  runs of minutes is coarse but adequate for per-run totals.
- **Peak memory** is `torch.cuda.max_memory_allocated`. It does not change with
  pruning because all expert weights stay loaded.
- **Communication cost** (required by the MoE README for distributed variants):
  not applicable. There is one GPU; the expert-parallel variant is only
  simulated.
- **Speculative decoding** was not run. The per-request variant is evaluated
  only by simulation on true consecutive tokens, which a real draft model may
  not produce.

### Choices where the paper is ambiguous

| Question | Choice | Reason |
|---|---|---|
| Is `b` the total set size or added on top of warm-up? | On top (`budget_mode="additional"`); "total" is also implemented. | Table 2 reports 9.1 activated experts at `b=8, k0=1`, concurrency 4. A total of 8 could not give that. |
| Which score is summed over the batch? | Softmax of the router logits over all experts. | The paper says "gating scores". Raw logits can be negative, which makes a sum meaningless. |
| Gate weights after masking? | The backbone's own rule. OLMoE: softmax over all experts, take top-k, no renormalisation. Dropping an expert does not change the weights of the others. | Keeps the unpruned case identical to the model's router. Checked: keep-all XShare gives the same tokens as the baseline. |
| What if the set has fewer than `k` experts? | Top up to `k` with the best remaining experts (`min_keep`). | Table 2 reports exactly 4.00 activated experts for `k0=1, b=0` at concurrency 4 with `k = 4`. |
| When to prune? | Decode steps only (one new token per sequence). Prefill is unpruned. | The paper formulates the problem "at the decode phase". |
| Per-request budget (`g`): total or additional? | Additional, as for `b`. | Table 3 shows activated experts rising with `g` beyond what a total cap of 4-6 would allow. |

## Reproduction checklist

- [x] Exact paper link and version verified (accepted-paper list, PDF checksum, arXiv version noted)
- [x] Base model and dataset revisions pinned (model `b89a7c4b...`, dataset `740312ad...`)
- [x] Baseline runs on the same setup
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported
- [x] Licence/data terms checked (model Apache-2.0, GSM8K MIT)
