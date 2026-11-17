# Paper implementation card: BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting

- **Paper / OpenReview:** <https://openreview.net/forum?id=uqeOxztSIS> (AdaptFM accepted paper). arXiv: <https://arxiv.org/abs/2605.29727>. arXiv lists v1 (28 May 2026) to v4 (28 Oct 2026). The text this card quotes from carries an ICML 2026 (PMLR 306) header. Which arXiv version that text matches was not verified.
- **Primary module:** `Speculative_Decoding/`
- **Secondary modules:** `Backbones/Generative_Pretrained/` (target models), `Dataloaders/evaluation/` (benchmark prompts), `Performance_Metrics/` (result rows, pending)
- **Owner / reviewer:** `<owner GitHub handle>` / `<reviewer GitHub handle>`
- **Status:** `partial-reproduction`. E1 (Table 1 at T=0, Qwen3-4B and Qwen3-8B), E3 (exact match against AR) and E5 (latency model) are measured. The E4 budget sweep and the fp32/bf16 precision check are still running (see "Pending"). The full write-up is in [`REPORT.md`](REPORT.md).
- **Upstream code / licence:** <https://github.com/kaist-ai-osi-lab/BASTION>, MIT ("Copyright (c) 2026 Z Lab"), pinned as the git submodule `upstream/` at commit `1989b80b304f01710946d7fb68af330c03d97461` and not modified
- **Backbone IDs:** `text.qwen3-4b.post_trained.mixed` + drafter `text.qwen3-4b-dflash-b16.drafter.unknown`, and `text.qwen3-8b.post_trained.mixed` + drafter `text.qwen3-8b-dflash-b16.drafter.unknown` ([`Backbones/registry.yaml`](../../Backbones/registry.yaml))
- **Dataset IDs and split:** `text.gsm8k.main.test`, `text.math500.test`, `text.aime25.test`, `text.humaneval.test`, `text.mbpp.sanitized.test`, `text.livecodebench.release_v2.test`, `text.mt_bench_prompts.train`, `text.alpaca.train` ([`Dataloaders/registry.yaml`](../../Dataloaders/registry.yaml), cards in [`Dataloaders/evaluation/bastion_short_benchmarks.md`](../../Dataloaders/evaluation/bastion_short_benchmarks.md))
- **Hardware assumptions:** one NVIDIA GPU, CUDA, bf16, batch size 1. Local: NVIDIA RTX PRO 4500 Blackwell, 32 GB (32,623 MiB per `nvidia-smi`), compute capability 12.0, driver 620.70.06, with torch 2.8.0+cu128 and transformers 4.57.1. The paper used A100 80 GB (Table 1), plus H100, RTX A6000 and RTX PRO 6000 Blackwell. BASTION's latency model needs the GPU's peak bf16 throughput and memory bandwidth, and needs exclusive use of the GPU during calibration.

## Research summary

### Problem
Block-diffusion drafters such as DFlash propose a whole block of future tokens in one parallel forward pass. They produce a separate marginal distribution for each position, not a jointly conditioned sequence. Verifying only the greedy top-1 chain therefore often misses the path the target model would have taken, and the accepted length stays short. A draft tree can fix this, but a bigger tree costs more verification time. The best size depends on the prompt, the context length, the model and the GPU. Static tree shapes and fixed budgets leave speed on the table.

### Method
At each decoding step BASTION builds a query-dependent draft tree from the drafter's position-wise distributions and chooses its size online:

1. **Acceptance surrogate from drafter marginals.** Each tree node gets a path score: the product of the drafter's marginal probabilities along its root-to-node path. Suppose the drafter acted as its own verifier, drawing one independent sample per position. The expected committed length is then exactly the sum of path scores over the tree. BASTION uses this sum as a cheap proxy for the expected number of tokens the target will accept. The paper reports a Pearson correlation of 0.79–0.82 between surrogate and measured accepted length.
2. **Best-first, prefix-closed tree construction.** Candidates come from the top-K tokens per position. Nodes are added in decreasing path-score order from a max-heap, and a child is pushed only after its parent is in the tree, so every intermediate tree is prefix-closed. Path scores never increase with depth. Greedy best-first insertion therefore gives, for every size N, the prefix-closed tree with the maximum surrogate value. The trees are nested, and the marginal gain per added node does not increase.
3. **Calibrated roofline verification-latency model.** The cost of verifying an N-node tree at the current context length is predicted analytically. A roofline model takes the larger of memory time and compute time, using the GPU's bandwidth and peak FLOPS and the model's dimensions. This is followed by a per-(GPU, model) linear calibration fitted once on measured verification latencies. Draft time and non-verification overhead are profiled once and treated as constants.
4. **Online budget controller.** While the tree grows, the controller tracks estimated speedup = surrogate accepted length × AR step latency ÷ predicted cycle cost. It stops at the first N where adding a node lowers this estimate. The paper shows the estimate is unimodal when the surrogate is concave and the cost is convex, so the first drop marks the global maximum. The selected tree is verified in one target forward pass with a tree attention mask, and standard speculative acceptance keeps the output distribution unchanged (lossless).

BASTION needs no training and no per-setting tuning beyond the one-off latency calibration.

### Expected behaviour
Paper-reported numbers: on A100 at T=0, BASTION raises the 8-benchmark average speedup over AR from 4.83× (DFlash) to 6.93× on Qwen3-4B and from 4.76× to 6.61× on Qwen3-8B. Average accepted length rises from about 6.2 to about 8.5 (Table 1, quoted below). The paper's mean realized tree sizes on A100 are N̄ = 202.0 (Qwen3-4B) and 139.3 (Qwen3-8B) (Fig. 9b, "Static"). The largest gains are on math and code. MT-Bench and Alpaca gain less. Output should equal AR output at T=0, except for rare bf16 numerical divergences. Absolute speedups depend on the hardware. The adaptive budget should approach the best fixed budget per dataset (paper Fig. 4) and avoid the slowdown that comes with over-budgeting. The locally measured values are in "Results".

## Implementation

The code lives in this folder. See [`README.md`](README.md), section "How to run", for details.

- **Input contract:** a chat prompt (user turns only) rendered with the target's chat template in **non-thinking mode** (`enable_thinking=False`) and tokenized to `input_ids` `int64[1, L]` on the GPU. Batch size 1, no padding. The target model (`AutoModelForCausalLM`, bf16) and the DFlash drafter (`DFlashDraftModel`, bf16, block size 16) are loaded once and shared by all modes. Generation stops at EOS `<|im_end|>` or after `max_new_tokens` = 2048. Temperature is 0.
- **Output contract:** upstream `bastion_generate` / `dflash_generate` return a namespace with `output_ids` `int64[1, L + n]`, `num_input_tokens`, `num_output_tokens`, `time_to_first_token`, `time_per_output_token` (decode only), per-step `acceptance_lengths`, and, for BASTION, per-step draft, verification and tree-build latencies. The harness writes one JSONL record per (dataset, `orig_idx`, turn, mode, budget) to `runs/<run-name>.jsonl`, plus the realized per-step tree sizes. `analysis/aggregate.py` reduces these to speedup, AAL and tree size per (model, dataset, method), a DA4022 CSV row, and a Table-1-style markdown file under `results/`.
- **Configuration:** the scripts in [`scripts/`](scripts/) set all flags. `run_e1.sh` is configured through environment variables: `MAX_SAMPLES`=64, `CAP_AIME25`=30, `CAP_MTBENCH`=40, `MAX_NEW_TOKENS`=2048, `WARMUP`=1 and `PROFILE_MODE`=warm. `run_sweep.sh` takes `SWEEP_SAMPLES`=16 and `BUDGETS="32 64 128 256 512 1024"`, plus adaptive with both profile modes. Method defaults come from upstream: adaptive minimum tree size 32, maximum 8192, top-K ≤ 128, `fixed_interval` return buckets, `BASTION_SIGMA_SCALE`=30. A fixed budget N is passed as `min_tree_size = max_tree_size = N`. Calibration caches: `upstream/cache/calibration_b4500_<model>.json`, `.profile.json` (upstream cold profile), `.profile.warm.json` (warm profile used by E1) and `calib_probe_b4500_<model>.sigma_sequence_v1.jsonl` (git-ignored).
- **Reproduction command** (E1 on Qwen3-8B, run from this folder):
  ```bash
  scripts/run_e1.sh qwen3-8b
  ```
  Full sequence on an otherwise idle GPU:
  ```bash
  .venv/bin/python tools/prepare_lcb.py                                     # once: LCB prompt cache
  tools/setup_smoke.sh Qwen/Qwen3-4B z-lab/Qwen3-4B-DFlash-b16              # calibration + smoke, refuses if GPU busy
  tools/setup_smoke.sh Qwen/Qwen3-8B z-lab/Qwen3-8B-DFlash-b16
  scripts/run_pilot.sh qwen3-4b                                             # 2 samples/dataset pipeline check
  scripts/run_e1.sh qwen3-4b && scripts/run_e1.sh qwen3-8b                  # E1 + E3 data
  scripts/run_sweep.sh qwen3-8b                                             # E4
  scripts/run_analysis.sh                                                   # E1/E3/E4/E5 tables, math accuracy, figures (CPU)
  ```
- **Tests:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q`. CPU only, synthetic data and tiny random models, no downloads. On 2026-11-22: 114 passed.

## Results

Experiments in this reproduction:

- **E1:** AR vs DFlash vs BASTION, Table 1 setting at T=0. Measured.
- **E3:** lossless check, exact match against AR at T=0. Measured.
- **E4:** fixed vs adaptive budget, as in Fig. 4. **Pending** (run in progress).
- **E5:** accuracy of the latency model, as in Fig. 9a. Measured.
- **Math accuracy:** a quality check on GSM8K, MATH-500 and AIME25. Measured.
- **Precision check:** fp32 vs bf16 decoding on GSM8K prompts, to test whether the E3 mismatches are a bf16 effect. **Pending**.

All local numbers come from `runs/e1_qwen3-4b.jsonl` and `runs/e1_qwen3-8b.jsonl` (494 prompts/turns × 3 modes per model) through the scripts in `analysis/`. The output files are in [`results/`](results/). Paper numbers are quoted from the paper and were not measured here.

### Reproduction verdict

| Claim (paper) | Verdict here | Evidence |
|---|---|---|
| BASTION is faster than DFlash on every benchmark (Table 1) | **Reproduced as a direction, much smaller margin.** BASTION's point estimate is higher on all 16 (model, benchmark) cells. On Qwen3-8B MATH500 and AIME25 the difference is within noise: 95% bootstrap intervals of the ratio are [0.992, 1.034] and [0.948, 1.058]. | E1 |
| Average speedup 6.93× / 6.61× (A100) | **Not reproduced** on this GPU: 4.50× / 4.47×, which is 0.650 / 0.676 of the paper's values. Absolute speedups are hardware-dependent. | E1 |
| BASTION/DFlash average ratio 1.43 / 1.39 | **Not reproduced**: 1.14 / 1.07. | E1 |
| BASTION AAL > DFlash AAL | **Reproduced** on all 16 cells. The gain is smaller: +1.88 / +1.27 tokens vs the paper's +2.33 / +2.24. | E1 |
| Mean realized tree N̄ = 202.0 / 139.3 (A100, Fig. 9b) | **Different**: 60.1 / 33.2 (pooled over steps). On Qwen3-8B, 96.3% of steps stop at the minimum size of 32. | E1 |
| Lossless at T=0 (output equals AR) | **Not confirmed.** Only 30–35% of DFlash and BASTION outputs match AR exactly, at similar rates for both methods. The cause is not yet established; the precision check is pending. | E3 |
| Calibration cuts the latency-model RMSE by 87–92% | **Partially reproduced**: 83.0% / 84.9% on the paper's contexts, with a higher calibrated residual (3.28 / 4.34 ms vs 2.3 / 3.1 ms). | E5 |
| Adaptive budget tracks the best fixed budget (Fig. 4) | Pending (E4). | E4 |

### Template summary

Local rows: mean over all 494 records per model (all 8 benchmarks pooled). Latency is the decode time per output token (TPOT, prefill excluded). Throughput is 1/mean TPOT. Peak memory is `torch.cuda.max_memory_allocated`, the maximum over records. All modes run in one process with both target and drafter loaded, so the AR value includes the drafter weights. Quality is boxed-answer accuracy summed over GSM8K + MATH-500 + AIME25 (158 problems).

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (A100, T=0, avg of 8 benchmarks) | not reported (lossless by construction) | not reported as absolute values | — | not reported | not reported | Speedup vs AR / AAL τ: Qwen3-4B DFlash 4.83× / 6.23, BASTION 6.93× / 8.56. Qwen3-8B DFlash 4.76× / 6.24, BASTION 6.61× / 8.48 (Table 1) |
| Local baseline AR, Qwen3-4B / Qwen3-8B (RTX PRO 4500, T=0) | 69.0% (109/158) / 70.3% (111/158) | 13.84 / 22.21 ms | 72.3 / 45.0 tok/s | 9,202 / 18,051 MiB | not_measured | |
| Local baseline DFlash, Qwen3-4B / Qwen3-8B | 67.7% (107/158) / 70.9% (112/158) | 4.03 / 6.21 ms | 248.0 / 160.9 tok/s | 9,357 / 18,469 MiB | not_measured | pooled speedup 3.43× / 3.57× |
| Local method BASTION adaptive (warm profile), Qwen3-4B / Qwen3-8B | 67.7% (107/158) / 70.3% (111/158) | 3.41 / 5.55 ms | 292.9 / 180.3 tok/s | 10,323 / 19,203 MiB | not_measured | pooled speedup 4.05× / 4.00× |

### E1: speedup and AAL (T=0, unweighted mean over the 8 short-context benchmarks)

Paper rows are quoted exactly from Table 1 (NVIDIA A100, T=0, "Average" column; EAGLE-3 tree budget 60, DFlash block size 16). Local intervals: 95% percentile bootstrap over prompts within each dataset (2000 resamples). They cover prompt-sampling variability only, not run-to-run timing noise.

| Source | GPU | Model | Method | Speedup | τ (AAL) | Mean tree size |
|---|---|---|---|---:|---:|---:|
| Paper | A100 | Qwen3-4B | EAGLE-3 (paper only, not run locally) | 2.80× | 2.45 | — |
| Paper | A100 | Qwen3-4B | DFlash | 4.83× | 6.23 | — |
| Paper | A100 | Qwen3-4B | BASTION | 6.93× | 8.56 | 202.0 (Fig. 9b) |
| Paper | A100 | Qwen3-8B | EAGLE-3 (paper only, not run locally) | 2.70× | 2.32 | — |
| Paper | A100 | Qwen3-8B | DFlash | 4.76× | 6.24 | — |
| Paper | A100 | Qwen3-8B | BASTION | 6.61× | 8.48 | 139.3 (Fig. 9b) |
| Local | RTX PRO 4500 Blackwell | Qwen3-4B | DFlash | 3.95× [3.88, 4.04] | 5.69 | — |
| Local | RTX PRO 4500 Blackwell | Qwen3-4B | BASTION (adaptive) | 4.50× [4.43, 4.59] | 7.58 | 60.1 |
| Local | RTX PRO 4500 Blackwell | Qwen3-8B | DFlash | 4.17× [4.08, 4.27] | 5.76 | — |
| Local | RTX PRO 4500 Blackwell | Qwen3-8B | BASTION (adaptive) | 4.47× [4.39, 4.55] | 7.03 | 33.2 |

BASTION/DFlash ratio of the averages: local 1.139 [1.126, 1.151] (Qwen3-4B) and 1.071 [1.058, 1.083] (Qwen3-8B). Paper: 1.435 and 1.389. Local mean tree sizes are pooled over all steps (36,048 and 37,311 steps). The unweighted means of the per-dataset values are 59.6 and 33.5.

Per benchmark (speedup ×, local / paper):

| Model | Method | GSM8K | MATH500 | AIME25 | HumanEval | MBPP | LCB | MT-Bench | Alpaca |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | DFlash | 4.62 / 5.23 | 5.57 / 6.22 | 4.49 / 6.16 | 4.52 / 5.34 | 4.14 / 5.09 | 3.97 / 5.71 | 2.26 / 2.74 | 2.07 / 2.12 |
| Qwen3-4B | BASTION | 5.26 / 7.44 | 5.88 / 8.57 | 4.88 / 8.23 | 5.15 / 7.91 | 4.93 / 7.41 | 4.53 / 7.81 | 2.79 / 4.53 | 2.60 / 3.53 |
| Qwen3-8B | DFlash | 4.83 / 4.96 | 5.71 / 6.06 | 4.97 / 5.91 | 4.76 / 5.25 | 4.27 / 4.91 | 4.39 / 5.90 | 2.28 / 2.81 | 2.14 / 2.28 |
| Qwen3-8B | BASTION | 5.17 / 6.92 | 5.78 / 8.02 | 4.98 / 7.69 | 5.17 / 7.64 | 4.65 / 7.10 | 4.67 / 7.64 | 2.71 / 4.31 | 2.60 / 3.54 |

τ (AAL), local / paper:

| Model | Method | GSM8K | MATH500 | AIME25 | HumanEval | MBPP | LCB | MT-Bench | Alpaca |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | DFlash | 6.23 / 6.50 | 8.19 / 8.12 | 7.02 / 7.48 | 6.52 / 6.66 | 5.50 / 6.31 | 5.89 / 7.23 | 3.25 / 4.63 | 2.96 / 2.90 |
| Qwen3-4B | BASTION | 8.26 / 9.07 | 10.01 / 10.60 | 8.96 / 9.70 | 8.44 / 9.38 | 7.67 / 8.99 | 8.12 / 9.48 | 4.79 / 6.58 | 4.36 / 4.64 |
| Qwen3-8B | DFlash | 6.28 / 6.51 | 8.04 / 8.25 | 7.15 / 7.20 | 6.42 / 6.64 | 5.55 / 6.04 | 6.62 / 7.64 | 3.14 / 4.46 | 2.90 / 3.15 |
| Qwen3-8B | BASTION | 7.85 / 8.85 | 9.31 / 10.53 | 8.34 / 9.46 | 8.06 / 9.44 | 7.01 / 8.65 | 7.33 / 9.91 | 4.33 / 6.28 | 4.02 / 4.68 |

Files: [`results/e1_table1.md`](results/e1_table1.md) (with per-group TPOT, token counts and truncation rates), [`results/e1_paper_comparison.md`](results/e1_paper_comparison.md) (ratios, bootstrap intervals, tree-size distribution, resume check), [`results/e1_summary.csv`](results/e1_summary.csv), and figures [`results/figures/speedup_bars_Qwen3-4B.png`](results/figures/speedup_bars_Qwen3-4B.png) and [`results/figures/speedup_bars_Qwen3-8B.png`](results/figures/speedup_bars_Qwen3-8B.png).

### E3: losslessness (T=0)

Exact match means the generated token ids are identical to AR's for the same prompt. A pure length difference counts as a divergence. n = 494 per row (6 × 64 + 30 + 80 MT-Bench turns).

| Model | Method | Exact-match rate vs AR | Records with divergence | Median first-divergence position (tokens) | Notes |
|---|---|---:|---:|---:|---|
| Qwen3-4B | DFlash | 34.8% (172/494) | 322 | 96 | per-dataset range 3.3% (AIME25) to 56.2% (LCB) |
| Qwen3-4B | BASTION | 33.4% (165/494) | 329 | 93 | per-dataset range 0.0% (AIME25) to 56.2% (LCB) |
| Qwen3-8B | DFlash | 30.2% (149/494) | 345 | 90 | per-dataset range 0.0% (AIME25) to 65.6% (LCB) |
| Qwen3-8B | BASTION | 31.0% (153/494) | 341 | 90 | per-dataset range 3.3% (AIME25) to 65.6% (LCB) |

Both speculative methods diverge from AR at about the same rate, and long generations (AIME25, MATH500) almost never match. Greedy speculative decoding is lossless in exact arithmetic. The working hypothesis is that bf16 numerics differ with the shape of the target forward pass (1 token for AR, a 16-token block for DFlash, an N-node tree for BASTION), so near-ties in the argmax flip. **This is not established yet**: the fp32-vs-bf16 precision check is pending. Speedups are still computed on matched prompts, but the continuations differ, so the decoded token sequences are not identical across modes. Mean output lengths stay within 2.6% of AR's (Qwen3-4B: AR 512.7, DFlash 520.2, BASTION 525.9 tokens; Qwen3-8B: AR 500.9, DFlash 497.3, BASTION 499.2). Files: [`results/e1_lossless.md`](results/e1_lossless.md), [`results/e1_lossless_per_record.csv`](results/e1_lossless_per_record.csv).

### E4: fixed vs adaptive budget (Qwen3-8B, 16 samples per benchmark)

**Pending: the sweep is still running (`runs/sweep_qwen3-8b.jsonl`).** It covers fixed N ∈ {32, 64, 128, 256, 512, 1024}, adaptive with the warm profile (E1 configuration), and adaptive with upstream's cold profile.

The paper shows this only as a figure (Fig. 4: Qwen3-8B and Llama-3.1-8B on A100, RTX PRO 6000 and A6000). It reports qualitatively that adaptive BASTION tracks the per-dataset best fixed N ("Oracle") on most panels. No numbers are quoted here.

| N | 32 | 64 | 128 | 256 | 512 | 1024 | adaptive, warm (mean realized N) | adaptive, upstream cold (mean realized N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean speedup (local) | pending | pending | pending | pending | pending | pending | pending | pending |

Pilot evidence (Qwen3-4B, 2 samples per dataset, 256 new tokens, upstream cold profile; `runs/pilot_qwen3-4b.jsonl`, not a result): pooled speedup was 3.62× for DFlash, 3.45× for adaptive BASTION (mean tree 181.2) and 4.21× for fixed N=64.

### E5: verification-latency model (Fig. 9a)

Probe grid from upstream's calibration: s = 16…512 (step 16) × c ∈ {64, 128, 256, 512, 1024, 2048}, 192 points per model, mean of 5 timed repeats each. The calibrated RMSE is in-sample. LOCO is leave-one-context-out, refit with upstream's fitting function, averaged over held-out contexts. "In-situ" compares the calibrated prediction with the verification latency measured at every BASTION step of E1 (tree mask, real contexts).

| Model | RMSE uncalibrated roofline | RMSE calibrated fit | Reduction | Paper (A100) |
|---|---:|---:|---:|---|
| Qwen3-4B, c ∈ {64, 256, 1024} | 19.28 ms | 3.28 ms | 83.0% | 29.2 → 2.3 ms (92%) |
| Qwen3-8B, c ∈ {64, 256, 1024} | 28.74 ms | 4.34 ms | 84.9% | 30.6 → 3.1 ms (90%) |
| Qwen3-4B, all 192 points | 20.46 ms | 3.82 ms (LOCO 4.13) | 81.4% | — |
| Qwen3-8B, all 192 points | 29.92 ms | 5.08 ms (LOCO 5.31) | 83.0% | — |
| Qwen3-4B, in-situ (36,048 steps) | 8.68 ms | 1.68 ms | — | — |
| Qwen3-8B, in-situ (37,311 steps) | 10.40 ms | 1.91 ms | — | — |

The uncalibrated roofline underestimates latency throughout (mean bias −17.8 / −26.3 ms on the paper's contexts), as in the paper. Its absolute error depends on the **assumed** peak throughput of the RTX PRO 4500 (202 TFLOPS; see below). The fitted memory-term slopes are α_m = 1.75 (Qwen3-4B) and 1.35 (Qwen3-8B). Measured latency jumps between s = 256 and 272 (+12.0 to +15.6 ms, depending on model and context) and again between s = 384 and 400. A linear calibration cannot follow these steps. The paper attributes such steps to CUDA tile allocation. Files: [`results/latency_model.md`](results/latency_model.md), [`results/latency_model.json`](results/latency_model.json), figures [`results/figures/latency_model_b4500_qwen3-4b.png`](results/figures/latency_model_b4500_qwen3-4b.png) and [`results/figures/latency_model_b4500_qwen3-8b.png`](results/figures/latency_model_b4500_qwen3-8b.png).

### Math accuracy (quality check, not in the paper)

Boxed-answer accuracy from `analysis/score_math.py`. This is a simple normalizer, not a standard evaluation harness, so compare the numbers across modes, not against published accuracies. With `max_new_tokens` = 2048 in non-thinking mode, 50–63% of AIME25 generations hit the token limit, so AIME25 accuracy mainly measures truncation.

| Model | Dataset (n) | AR | DFlash | BASTION |
|---|---|---:|---:|---:|
| Qwen3-4B | GSM8K (64) | 87.5% (56) | 87.5% (56) | 89.1% (57) |
| Qwen3-4B | MATH-500 (64) | 75.0% (48) | 73.4% (47) | 71.9% (46) |
| Qwen3-4B | AIME25 (30) | 16.7% (5) | 13.3% (4) | 13.3% (4) |
| Qwen3-8B | GSM8K (64) | 92.2% (59) | 92.2% (59) | 90.6% (58) |
| Qwen3-8B | MATH-500 (64) | 73.4% (47) | 75.0% (48) | 75.0% (48) |
| Qwen3-8B | AIME25 (30) | 16.7% (5) | 16.7% (5) | 16.7% (5) |

Across the 158 problems the speculative methods differ from AR by at most 2 correct answers per model (Qwen3-4B: AR 109, DFlash 107, BASTION 107; Qwen3-8B: AR 111, DFlash 112, BASTION 111). These differences come from the output divergences in E3. At these sample sizes they are not distinguishable from zero. Files: [`results/e1_math_scores.md`](results/e1_math_scores.md).

### Precision check (fp32 vs bf16)

**Pending** (`results/precision_check_qwen3-4b/`, run queued after the sweep).

## Differences and limitations

Scope decisions:

- **Targets:** only Qwen3-4B and Qwen3-8B, paired with `z-lab/Qwen3-4B-DFlash-b16` and `z-lab/Qwen3-8B-DFlash-b16`. Llama-3.1-8B-Instruct and its drafter were skipped: the model is gated and no Hugging Face token was available. The harness still accepts the `llama3.1-8b` key.
- **Temperature 0 only.** The T=1 half of Table 1 was skipped for time. The harness passes T > 0 through, but that path is untested.
- **Batch size 1**, as in the paper.
- **Hardware:** one NVIDIA RTX PRO 4500 Blackwell (32 GB) in a shared machine. The paper used A100 (Table 1), H100, RTX A6000 and RTX PRO 6000 Blackwell. Absolute speedups are hardware-dependent and are not expected to match the A100 numbers.
- **Attention backend:** SDPA for every mode, because flash-attn is not installed. Upstream uses `flash_attention_2` for AR and DFlash when it is available, and always SDPA for BASTION verification. The paper's AR and DFlash baselines may have used FA2, which shifts every speedup ratio by an unknown amount in an unknown direction.
- **Sample counts:** 64 per benchmark (seed-0 shuffle, first 64); all 30 for AIME25 (original order); 40 MT-Bench questions × 2 turns. The paper does not state its per-benchmark counts; Alpaca's full split has 52,002 items. `max_new_tokens` = 2048 (upstream default). One discarded warm-up pass per mode; upstream has none.
- **Budget sweep (E4):** Qwen3-8B only, 16 samples per benchmark, N ∈ {32, 64, 128, 256, 512, 1024} plus adaptive. The paper's sweep also covers Llama-3.1-8B and three GPUs.
- **Not run:**
  - The EAGLE-3 baseline. The paper used the separate AngelSlim codebase, so its numbers are quoted from the paper only.
  - LongBench (7 English subsets).
  - The beam-search ablation (Fig. 8).
  - The H100 results (Table 5).
  - The online-calibration variants (Fig. 9b).
- **Execution:** a single process on one GPU, without torchrun sharding. Several datasets run in one process, with sample selection reproduced exactly (seed-0 shuffle, first k; see the dataset card).

Workarounds and environment differences:

- **GPU not in upstream's cost-model table.** Unmodified upstream raises `Unsupported gpu_type: 'NVIDIA RTX PRO 4500 Blackwell'`. [`tools/bastion_local.py`](tools/bastion_local.py) registers it at run time as `b4500`, and the harness reuses that registration. The memory bandwidth is 896 GB/s, from NVIDIA's product page. The peak dense bf16 throughput is **assumed** to be about 202 TFLOPS: 1617 sparse FP4 TOPS divided by 8, the same ratio that maps the RTX PRO 6000 to upstream's 504 TFLOPS. A 4096² bf16 matmul measured 177 TFLOPS locally. These priors feed straight into the uncalibrated roofline (E5). The per-(GPU, model) linear calibration absorbs a constant scale error.
- **Calibration memory fix.** Upstream's calibration OOMed on Qwen3-8B with 32 GB (`runs/setup_smoke_Qwen3-8B.oom_attempt1.log`): its untimed context prefill runs with autograd enabled and keeps activations alive. `tools/bastion_local.py` wraps `collect_calibration_data` in `torch.inference_mode()`. The timed forwards were already in inference mode, so the measured latencies are unaffected.
- **Controller profile constants (warm profile).** Upstream profiles draft latency and non-verification overhead once, cold (3 trials, no warm-up). On this machine that gave 15.1 ms / 151.3 ms for Qwen3-4B and 5.4 ms / 17.4 ms for Qwen3-8B. Repeating the same profiler call and keeping the second result gave 3.2 ms / 0.85 ms and 4.9 ms / 0.99 ms. During E1, the logged per-step draft latency averaged 3.3 ms (Qwen3-4B) and 5.0 ms (Qwen3-8B). Inflated constants make the controller choose oversized trees: in the pilot, with the cold profile, adaptive was slower than both DFlash and fixed N=64 (see E4). E1 therefore uses `--profile-mode warm`. This is a deliberate deviation from upstream's default, and E4 runs both variants.
- **LiveCodeBench loading.** `livecodebench/code_generation_lite` is script-only, and `datasets>=4` rejects scripts. [`tools/prepare_lcb.py`](tools/prepare_lcb.py) rebuilds the `release_v2` prompt cache (511 items) from the raw `test.jsonl` and `test2.jsonl` at a pinned revision, using upstream's own prompt format.
- **Software versions:** `datasets` 5.0.1, where upstream's `uv.lock` pins 4.8.5. The DFlash model cards list torch 2.9 and transformers 4.57.3 for their evaluation; this setup has torch 2.8.0+cu128 and transformers 4.57.1. The full freeze is in `runs/setup_pip_freeze.txt` (git-ignored).
- **Dataset revisions:** upstream does not pin them. The revisions cached on 2026-11-21 and the sha256 of each prompt file are recorded in `Dataloaders/registry.yaml`.
- **Interrupted run.** The first Qwen3-4B E1 session (started 2026-11-21 20:02 IST) was killed at 20:48 IST, when its controlling session ended, during HumanEval. It was resumed on 2026-11-22 at 14:05 IST from the JSONL, keeping its 541 records. HumanEval #22 has its AR record from the first session and its DFlash/BASTION records from the second. For HumanEval, mean TPOT per session is AR 13.89 / 13.81 ms, DFlash 3.05 / 3.07 ms and BASTION 2.68 / 2.69 ms (`results/e1_paper_comparison.md`).
- **Metric definitions:**
  - Speedup is mean per-record TPOT(AR) divided by mean per-record TPOT(method), as upstream prints it. It is computed over records matched on (model, dataset, `orig_idx`, turn) with the same prompt hash. All 1,976 non-AR records matched. TPOT covers decode only.
  - AAL pools all steps.
  - The "Average" row is the unweighted mean over datasets, as in the paper's Table 1.
- **Math accuracy** is our own check and not part of the paper. It has no symbolic equivalence checking, so it is a lower bound.
- **Calibration and GPU sharing.** Calibration ran on 2026-11-21 between 19:49 and 19:58 IST through `tools/setup_smoke.sh`, which refuses to start while any other compute process is on the GPU. The E1 runs were queued one after another. Other users' jobs were not logged during the runs. The AR TPOT spread was small (Qwen3-4B 13.84 ± 0.23 ms, Qwen3-8B 22.21 ± 0.29 ms, mean ± s.d. over 494 records), so no large interference shows in the data.
- **Peak memory** is reported as PyTorch peak allocated memory per record. Energy is not measured.

## Reproduction checklist

- [ ] Exact paper link and version verified. The OpenReview and arXiv links resolve, but which arXiv version (v1–v4) matches the quoted text was not verified.
- [x] Base model and dataset revisions pinned. Model revisions are pinned. Dataset revisions and prompt-file hashes are recorded, but upstream loads `main`.
- [x] Baseline runs on the same setup. AR and DFlash run in the same process, on the same prompts and GPU, as BASTION.
- [x] Tests and seed handling added (114 CPU tests; selection equivalence with upstream tested)
- [ ] Resource and quality metrics reported. Latency, throughput, peak memory and math accuracy are reported for E1. Energy is not measured. E4 and the precision check are pending.
- [x] Licence/data terms checked. Upstream code: MIT. Qwen3: Apache-2.0. DFlash drafters: MIT. For the datasets, see the dataset card; the MATH-500 and LiveCodeBench licences are not verified, and Alpaca is CC-BY-NC-4.0.
