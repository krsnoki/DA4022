# Paper implementation card: BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting

- **Paper / OpenReview:** <https://openreview.net/forum?id=uqeOxztSIS> (AdaptFM accepted paper). arXiv: <https://arxiv.org/abs/2605.29727>. arXiv lists v1 (28 May 2026) to v4 (28 Oct 2026). The text this card quotes from carries an ICML 2026 (PMLR 306) header. Which arXiv version that text matches was not verified.
- **Primary module:** `Speculative_Decoding/`
- **Secondary modules:** `Backbones/Generative_Pretrained/` (target models), `Dataloaders/evaluation/` (benchmark prompts), `Performance_Metrics/` (result rows, pending)
- **Owner / reviewer:** `<owner GitHub handle>` / `<reviewer GitHub handle>`
- **Status:** implementing. The harness, analysis and CPU tests are done. The GPU runs have not been executed, so no local results exist yet.
- **Upstream code / licence:** <https://github.com/kaist-ai-osi-lab/BASTION>, MIT ("Copyright (c) 2026 Z Lab"), pinned as the git submodule `upstream/` at commit `1989b80b304f01710946d7fb68af330c03d97461` and not modified
- **Backbone IDs:** `text.qwen3-4b.post_trained.mixed` + drafter `text.qwen3-4b-dflash-b16.drafter.unknown`, and `text.qwen3-8b.post_trained.mixed` + drafter `text.qwen3-8b-dflash-b16.drafter.unknown` ([`Backbones/registry.yaml`](../../Backbones/registry.yaml))
- **Dataset IDs and split:** `text.gsm8k.main.test`, `text.math500.test`, `text.aime25.test`, `text.humaneval.test`, `text.mbpp.sanitized.test`, `text.livecodebench.release_v2.test`, `text.mt_bench_prompts.train`, `text.alpaca.train` ([`Dataloaders/registry.yaml`](../../Dataloaders/registry.yaml), cards in [`Dataloaders/evaluation/bastion_short_benchmarks.md`](../../Dataloaders/evaluation/bastion_short_benchmarks.md))
- **Hardware assumptions:** one NVIDIA GPU, CUDA, bf16, batch size 1. Local: NVIDIA RTX PRO 4500 Blackwell, 32 GB, compute capability 12.0, with torch 2.8.0+cu128 and transformers 4.57.1. The paper used A100 80 GB (Table 1), plus H100, RTX A6000 and RTX PRO 6000 Blackwell. BASTION's latency model needs the GPU's peak bf16 throughput and memory bandwidth, and needs exclusive use of the GPU during calibration.

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
Paper-reported numbers: on A100 at T=0, BASTION raises the 8-benchmark average speedup over AR from 4.83× (DFlash) to 6.93× on Qwen3-4B and from 4.76× to 6.61× on Qwen3-8B. Average accepted length rises from about 6.2 to about 8.5 (Table 1, quoted below). The largest gains are on math and code. MT-Bench and Alpaca gain less. Output should equal AR output at T=0, except for rare bf16 numerical divergences. Absolute speedups depend on the hardware. The adaptive budget should approach the best fixed budget per dataset (paper Fig. 4) and avoid the slowdown that comes with over-budgeting. No locally measured values exist yet.

## Implementation

The code lives in this folder. See [`README.md`](README.md), section "How to run", for details.

- **Input contract:** a chat prompt (user turns only) rendered with the target's chat template in **non-thinking mode** (`enable_thinking=False`) and tokenized to `input_ids` `int64[1, L]` on the GPU. Batch size 1, no padding. The target model (`AutoModelForCausalLM`, bf16) and the DFlash drafter (`DFlashDraftModel`, bf16, block size 16) are loaded once and shared by all modes. Generation stops at EOS `<|im_end|>` or after `max_new_tokens` = 2048. Temperature is 0.
- **Output contract:** upstream `bastion_generate` / `dflash_generate` return a namespace with `output_ids` `int64[1, L + n]`, `num_input_tokens`, `num_output_tokens`, `time_to_first_token`, `time_per_output_token` (decode only), per-step `acceptance_lengths`, and, for BASTION, per-step draft, verification and tree-build latencies. The harness writes one JSONL record per (dataset, `orig_idx`, turn, mode, budget) to `runs/<run-name>.jsonl`, plus the realized per-step tree sizes. `analysis/aggregate.py` reduces these to speedup, AAL and tree size per (model, dataset, method), a DA4022 CSV row, and a Table-1-style markdown file under `results/`.
- **Configuration:** the scripts in [`scripts/`](scripts/) set all flags. `run_e1.sh` is configured through environment variables: `MAX_SAMPLES`=64, `CAP_AIME25`=30, `CAP_MTBENCH`=40, `MAX_NEW_TOKENS`=2048 and `WARMUP`=1. `run_sweep.sh` takes `SWEEP_SAMPLES`=16 and `BUDGETS="adaptive 32 64 128 256 512 1024"`. Method defaults come from upstream: adaptive minimum tree size 32, maximum 8192, top-K ≤ 128, `fixed_interval` return buckets, `BASTION_SIGMA_SCALE`=30. A fixed budget N is passed as `min_tree_size = max_tree_size = N`. Calibration caches: `upstream/cache/calibration_b4500_<model>.json`, `.profile.json` and `calib_probe_b4500_<model>.sigma_sequence_v1.jsonl` (git-ignored).
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
- **Tests:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q`. CPU only, synthetic data and tiny random models, no downloads. On 2026-11-21: 102 passed in about 3 s.

## Results

Experiments in this reproduction:

- **E1:** AR vs DFlash vs BASTION, Table 1 setting at T=0.
- **E3:** lossless check, exact match against AR at T=0.
- **E4:** fixed vs adaptive budget, as in Fig. 4.
- **E5:** accuracy of the latency model, as in Fig. 9a.
- **Math accuracy:** a quality check on GSM8K, MATH-500 and AIME25.

**All local cells are `TBD`: no GPU run has been executed.** Do not read them as results.

### Template summary

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (A100, T=0, avg of 8 benchmarks) | not reported (lossless by construction) | not reported as absolute values | — | not reported | not reported | Speedup vs AR / AAL τ: Qwen3-4B DFlash 4.83× / 6.23, BASTION 6.93× / 8.56. Qwen3-8B DFlash 4.76× / 6.24, BASTION 6.61× / 8.48 (Table 1) |
| Local baseline (AR, RTX PRO 4500, T=0) | TBD (math acc.) | TBD (TPOT ms) | TBD (tok/s) | TBD (MiB, allocated) | not_measured | DFlash is also run as a second baseline |
| Local method (BASTION adaptive, RTX PRO 4500, T=0) | TBD (math acc.) | TBD (TPOT ms) | TBD (tok/s) | TBD (MiB, allocated) | not_measured | |

### E1: speedup and AAL (T=0, unweighted mean over the 8 short-context benchmarks)

Paper rows are quoted exactly from Table 1 (NVIDIA A100, T=0, "Average" column; EAGLE-3 tree budget 60, DFlash block size 16).

| Source | GPU | Model | Method | Speedup | τ (AAL) |
|---|---|---|---|---:|---:|
| Paper | A100 | Qwen3-4B | EAGLE-3 (paper only, not run locally) | 2.80× | 2.45 |
| Paper | A100 | Qwen3-4B | DFlash | 4.83× | 6.23 |
| Paper | A100 | Qwen3-4B | BASTION | 6.93× | 8.56 |
| Paper | A100 | Qwen3-8B | EAGLE-3 (paper only, not run locally) | 2.70× | 2.32 |
| Paper | A100 | Qwen3-8B | DFlash | 4.76× | 6.24 |
| Paper | A100 | Qwen3-8B | BASTION | 6.61× | 8.48 |
| Local | RTX PRO 4500 Blackwell | Qwen3-4B | DFlash | TBD | TBD |
| Local | RTX PRO 4500 Blackwell | Qwen3-4B | BASTION (adaptive) | TBD | TBD |
| Local | RTX PRO 4500 Blackwell | Qwen3-8B | DFlash | TBD | TBD |
| Local | RTX PRO 4500 Blackwell | Qwen3-8B | BASTION (adaptive) | TBD | TBD |

Per-benchmark local results: TBD (`results/e1_*.md` after `scripts/run_analysis.sh`).

### E3: losslessness (T=0)

| Model | Method | Exact-match rate vs AR | Records with divergence | Notes |
|---|---|---:|---:|---|
| Qwen3-4B | DFlash | TBD | TBD | |
| Qwen3-4B | BASTION | TBD | TBD | |
| Qwen3-8B | DFlash | TBD | TBD | |
| Qwen3-8B | BASTION | TBD | TBD | |

### E4: fixed vs adaptive budget (Qwen3-8B, 16 samples per benchmark)

The paper shows this only as a figure (Fig. 4: Qwen3-8B and Llama-3.1-8B on A100, RTX PRO 6000 and A6000). It reports qualitatively that adaptive BASTION tracks the per-dataset best fixed N ("Oracle") on most panels. No numbers are quoted here.

| N | 32 | 64 | 128 | 256 | 512 | 1024 | adaptive (mean realized N) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean speedup (local) | TBD | TBD | TBD | TBD | TBD | TBD | TBD (TBD) |

### E5: verification-latency model (Fig. 9a)

Paper (A100, three targets, contexts 64/256/1024): calibration cuts RMSE by 87–92% compared with the uncalibrated roofline.

| Model | RMSE uncalibrated roofline | RMSE calibrated fit | Reduction |
|---|---:|---:|---:|
| Qwen3-4B (local) | TBD | TBD | TBD |
| Qwen3-8B (local) | TBD | TBD | TBD |

### Math accuracy (quality check, not in the paper)

Boxed-answer accuracy from `analysis/score_math.py`. This is a simple normalizer, not a standard evaluation harness, so compare the numbers across modes, not against published accuracies.

| Model | Dataset | AR | DFlash | BASTION |
|---|---|---:|---:|---:|
| Qwen3-4B / Qwen3-8B | GSM8K, MATH-500, AIME25 | TBD | TBD | TBD |

## Differences and limitations

Scope decisions:

- **Targets:** only Qwen3-4B and Qwen3-8B, paired with `z-lab/Qwen3-4B-DFlash-b16` and `z-lab/Qwen3-8B-DFlash-b16`. Llama-3.1-8B-Instruct and its drafter were skipped: the model is gated and no Hugging Face token was available. The harness still accepts the `llama3.1-8b` key.
- **Temperature 0 only.** The T=1 half of Table 1 was skipped for time. The harness passes T > 0 through, but that path is untested.
- **Batch size 1**, as in the paper.
- **Hardware:** one NVIDIA RTX PRO 4500 Blackwell (32 GB). The paper used A100 (Table 1), H100, RTX A6000 and RTX PRO 6000 Blackwell. Absolute speedups are hardware-dependent and are not expected to match the A100 numbers.
- **Attention backend:** SDPA for every mode, because flash-attn is not installed. Upstream uses `flash_attention_2` for AR and DFlash when it is available, and always SDPA for BASTION verification. The local AR and DFlash baselines may therefore be slower or faster than the paper's, which shifts every speedup ratio.
- **Sample counts:** 64 per benchmark; all 30 for AIME25 (original order); 40 MT-Bench questions × 2 turns. The paper does not state its per-benchmark counts. `max_new_tokens` = 2048 (upstream default). One discarded warm-up pass per mode; upstream has none.
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
- **LiveCodeBench loading.** `livecodebench/code_generation_lite` is script-only, and `datasets>=4` rejects scripts. [`tools/prepare_lcb.py`](tools/prepare_lcb.py) rebuilds the `release_v2` prompt cache (511 items) from the raw `test.jsonl` and `test2.jsonl` at a pinned revision, using upstream's own prompt format.
- **Software versions:** `datasets` 5.0.1, where upstream's `uv.lock` pins 4.8.5. The DFlash model cards list torch 2.9 and transformers 4.57.3 for their evaluation; this setup has torch 2.8.0+cu128 and transformers 4.57.1. The full freeze is in `runs/setup_pip_freeze.txt` (git-ignored).
- **Dataset revisions:** upstream does not pin them. The revisions cached on 2026-11-21 and the sha256 of each prompt file are recorded in `Dataloaders/registry.yaml`.
- **Metric definitions:**
  - Speedup is mean per-record TPOT(AR) divided by mean per-record TPOT(method), as upstream prints it. It is computed over records matched on (model, dataset, `orig_idx`, turn) with the same prompt hash. TPOT covers decode only.
  - AAL pools all steps.
  - The "Average" row is the unweighted mean over datasets, as in the paper's Table 1.
- **Math accuracy** is our own check and not part of the paper. It has no symbolic equivalence checking, so it is a lower bound.
- **Calibration requires an idle GPU.** At the time of writing, the GPU was occupied by an unrelated long-running process that held about 30 of 32 GB. Calibration and all GPU experiments are pending. Any concurrent GPU work during calibration corrupts the latency fit; delete the `upstream/cache/calibration_b4500_*` and `calib_probe_b4500_*` files to recalibrate.
- **Peak memory** will be reported as PyTorch peak allocated memory per record. Energy is not measured.

## Reproduction checklist

- [ ] Exact paper link and version verified. The OpenReview and arXiv links resolve, but which arXiv version (v1–v4) matches the quoted text was not verified.
- [x] Base model and dataset revisions pinned. Model revisions are pinned. Dataset revisions and prompt-file hashes are recorded, but upstream loads `main`.
- [ ] Baseline runs on the same setup. AR and DFlash run in the same process as BASTION. GPU runs are pending.
- [x] Tests and seed handling added (102 CPU tests; selection equivalence with upstream tested)
- [ ] Resource and quality metrics reported (pending GPU runs)
- [x] Licence/data terms checked. Upstream code: MIT. Qwen3: Apache-2.0. DFlash drafters: MIT. For the datasets, see the dataset card; the MATH-500 and LiveCodeBench licences are not verified, and Alpaca is CC-BY-NC-4.0.
