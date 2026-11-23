# Paper implementation card: DIPA: Difficulty-Informed Probabilistic Allocation of Test-Time Compute via Training-Free Proxies

- **Paper / OpenReview:** https://openreview.net/forum?id=9Joja0tx5c (AdaptFM Workshop at ICML 2026)
- **Primary module:** `Adaptive_Stopping`
- **Secondary modules:** `Backbones`, `Dataloaders`, `Performance_Metrics`
- **Owner / reviewer:** `@Victorvkk38596` (Vasudev Krishna)
- **Team:** Vasudev Krishna, G Jayagiri, Uditya Shekhawat, Pranav Deshmukh, Arkaprava Paul, Harshit Goel, and T Chethan
- **Status:** Proposed
- **Upstream code / licence:** https://github.com/allen4747/DIPA (placeholder repository with `README.md` only; no source code or licence published as of 2026-11-22)
- **Backbone IDs:** `Qwen/Qwen2.5-Math-1.5B-Instruct` (primary target; exact revision and registry ID to be pinned in `Backbones/Generative_Pretrained/`). Out of initial PR scope: `Qwen2.5-1.5B-Instruct` (non-verifiable GPQA-Diamond), `Llama3.1-8B` (gated and exceeds single 16 GB GPU in FP16 for multi-sample inference).
- **Dataset IDs and split:** `MATH500` official test split (primary target; `GSM8K` official test split as optional stretch; exact revisions and registry IDs to be pinned in `Dataloaders/evaluation/`). Out of initial PR scope: `LiveCodeBench` (requires sandboxed code execution verifier), `GPQADiamond` (requires 4B reward model `Skywork-Reward-V2-Qwen3-4B`).
- **Hardware assumptions:** Local CPU for unit/smoke tests and gradient-proxy verification; Kaggle Tesla T4 (16 GB VRAM) in FP16/BF16 for `Qwen2.5-Math-1.5B-Instruct` inference benchmarks.

## Research summary

### Problem
Test-time compute scaling methods (e.g., Best-of-N, Self-Consistency) typically allocate a uniform number of generation attempts across all queries. Uniform allocation wastes compute on easy problems and under-allocates on hard problems under a fixed total budget `T`. Prior adaptive allocators require training auxiliary difficulty probes (e.g., ~54 GPU-hours in Damani et al., 2025), adding training data and inference overhead.

### Method
DIPA estimates instance difficulty using training-free signals derived directly from the LLM, then allocates generation attempts via a Multi-Armed Bandit (MAB) policy with arm elimination upon success (Algorithm 1):

1. **Input-based proxies** (computed before generation to initialize `M_i`):
   - `MQL(x) = |x|` (query token length; zero extra forward pass)
   - `MEnt(x) = CE(x)` (input cross-entropy; single forward pass)
   - `MGN(x)` and `MVoG(x)` (mean and variance of input-embedding gradient norms; backward pass, implemented and unit-tested, excluded from full Kaggle sweeps due to compute cost)
2. **Generation-based proxies** (updated dynamically from accumulated generations `O_i` on failed attempts):
   - `MGL(O|x)` (mean generation length; paper's main proposal and top proxy on MATH500; zero extra forward pass)
   - `MGC(O|x)` (generation consensus / majority-answer fraction; top proxy on GSM8K; zero extra forward pass)
   - `MEnt(O|x)` (sequence cross-entropy; forward pass only)
   - `MGN(O|x)` and `MVoG(O|x)` (implemented and unit-tested; excluded from full Kaggle sweeps)
3. **Probabilistic allocation (Algorithm 1):**
   - At each compute step `t in [T]`, active unsolved instances `x_k in X` are sampled with probability `P_k = (1 / M_k^lambda) / sum_{x_l in X} (1 / M_l^lambda)`, where higher `M_k` indicates greater difficulty.
   - Proxies negatively correlated with difficulty (e.g., `MGC`, where higher consensus implies an easier instance) are inverted before applying Eq. 4 (Appendix D.3).
   - Sampling temperature `lambda` supports both fixed values (e.g., `1`, `10` as ablated in Appendix E.7) and a configurable dynamic schedule `lambda proportional to |X|`.
   - On verifiable math tasks, an attempt is checked against the ground-truth answer extracted from `\boxed{}` (Appendix D.2 prompt template). Solved instances are removed from `X`; unsolved instances update `M_j <- M_gen(O_j | x_j)`.

### Expected behaviour
DIPA (`DIPA(MGL)`) is expected to achieve higher coverage (fraction of solved instances) than uniform budget allocation, greedy `Easy2Hard`, and `Hard2Easy` at equal total compute budget `T`, especially in tight-to-moderate budget regimes (`T = 0.5N` to `8N`).

## Implementation

- **Input contract:** Instance set `X = {x_i}` (`N` prompts), integer total compute budget `T >= 0`, backbone generator interface (sampling hyperparameters per Appendix D.1: `temperature=0.7`, `top_p=0.8`, `repetition_penalty=1.05`, `max_new_tokens=512`), verifiable math answer checker `1(o | x_i)`, proxy selectors (`M_input`, `M_gen`), temperature schedule `lambda`, and random seed.
- **Output contract:** Batch-level `BudgetAllocator` interface returning per-instance allocations `T_i` (`sum T_i <= T`), candidate generations `O_i`, solved flags `s_i`, extracted answers, and an allocation summary dict, plus a `StoppingPolicy`-compatible adapter (`reset`, `should_stop`, `report`) matching `Adaptive_Stopping/README_adaptiveStopping.md`.
- **Configuration:** `Adaptive_Stopping/configs/dipa.yaml` (to be added)
- **Reproduction command:** `<to be added once implemented>`
- **Tests:** `pytest Adaptive_Stopping/tests/test_dipa.py` (to be added)

## Results

Paper-reported values below are from Table 5 (`MATH500`, `Qwen2.5-Math-1.5B-Instruct`, 3-seed average, run on 8x H100 PCIe with 500 pre-sampled responses per question).

| Source | Quality (Coverage on MATH500) | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (`DIPA(MGL)`) | `0.333` (`0.5N`), `0.565` (`2N`), `0.692` (`8N`), `0.765` (`32N`) | `1.62 s` (allocation overhead, Table 6) | not_measured | not_measured | not_measured | Table 5 & 6; 8x H100 PCIe |
| Local baseline (Uniform) | | | | | | Kaggle T4 16 GB; identical split, prompts, seeds |
| Local baseline (Easy2Hard) | | | | | | Kaggle T4 16 GB; identical split, prompts, seeds |
| Local method (`DIPA(MGL)`) | | | | | | Kaggle T4 16 GB; identical split, prompts, seeds |

## Differences and limitations

- **No upstream implementation:** `https://github.com/allen4747/DIPA` contains only a `README.md` with no code or licence; this implementation is written from the paper specification.
- **`DIPA(MGC)` input prior (`const`):** `MGC(O|x)` has no input-based proxy counterpart in Table 1, and its inverted score `1 / MGC >= 1.0` starts at `1.0` on `m = 1` attempt. Pairing `mgc` with `mql` (`20..100` tokens) drops a failed arm's difficulty by ~50x on the first pull and locks onto one problem, so `dipa_mgc` uses `input_proxy="const"` (`1.0`).
- **Unspecified dynamic `lambda` proportionality constant:** Section 4.2 and Appendix E.7 specify `lambda proportional to |X|` without stating the exact proportionality constant. Setting `lambda = 1.0 * |X|` directly collapses DIPA into deterministic `Easy2Hard` at `N >= 100` (`lambda_t >= 57`). We implement scale-invariant `lambda_t = lambda_value * (|X| / N)` (default `lambda_value = 5.0` in `dipa.yaml`, matching Appendix E.7's `[1, 10]` scale) plus fixed `lambda` overrides via `--lambda-mode` / `--lambda-value`.
- **Hardware and budget scope:** Paper experiments ran on 8x H100 (80 GB) with 500 pre-sampled responses per problem (and up to 48 GPU-hours for gradient proxies). Kaggle T4 (16 GB) benchmarks will use `Qwen2.5-Math-1.5B-Instruct` with lightweight zero-overhead proxies (`MQL`, `MGL`, `MGC`, plus `MEnt`) and lower total budgets (e.g., up to `T = 8N` or a pinned subset), so full benchmark results will be labelled `partial` reproduction unless the full `32N` sweep completes.
- **Out-of-scope models and tasks:** `Llama3.1-8B` (gated, FP16 VRAM limit on single T4), `LiveCodeBench` (requires sandboxed code execution), and `GPQADiamond` (requires 4B reward model `Skywork-Reward-V2-Qwen3-4B`) are deferred to follow-up work.
- **Batch-level interface:** DIPA allocates budget across `N` instances jointly rather than making independent single-request early-exit decisions; the `StoppingPolicy` methods act as a wrapper around the batch allocator state.
- **Energy:** Will be reported as `not_measured` if Kaggle GPU power telemetry is unavailable.

## Reproduction checklist

- [x] Exact paper link and version verified
- [ ] Base model and dataset revisions pinned
- [ ] Baseline runs on the same setup
- [ ] Tests and seed handling added
- [ ] Resource and quality metrics reported
- [ ] Licence/data terms checked
