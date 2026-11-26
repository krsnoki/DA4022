# Paper implementation card: DIPA: Difficulty-Informed Probabilistic Allocation of Test-Time Compute via Training-Free Proxies

- **Paper / OpenReview:** https://openreview.net/forum?id=9Joja0tx5c (AdaptFM Workshop at ICML 2026)
- **Primary module:** `Adaptive_Stopping`
- **Secondary modules:** `Backbones`, `Dataloaders`, `Performance_Metrics`
- **Team:** Vasudev Krishna (`@Victorvkk38596`) - 230102113, G Jayagiri (`@jayagiri14`) - 230108067, Uditya Shekhawat, Pranav Deshmukh, Arkaprava Paul, Harshit Goel, and Thirunagari Chethan (`@chethanthirunagari`) - 230102099
- **Status:** partial-reproduction
- **Upstream code / licence:** https://github.com/allen4747/DIPA (placeholder repository with `README.md` only; no source code or licence published as of 2026-11-26)
- **Backbone IDs:** `llm.qwen2_5_math_1_5b_instruct.generative` (`Qwen/Qwen2.5-Math-1.5B-Instruct`, Apache-2.0, `Backbones/registry.yaml`). Deferred out of initial PR scope: `Qwen2.5-1.5B-Instruct` (non-verifiable GPQA-Diamond), `Llama3.1-8B` (gated).
- **Dataset IDs and split:** `math.math500.test` (`HuggingFaceH4/MATH-500`, MIT, `test` split, 500 examples, `Dataloaders/registry.yaml`). Deferred out of initial PR scope: `GSM8K`, `LiveCodeBench` (sandboxed code execution), `GPQADiamond` (4B reward model `Skywork-Reward-V2-Qwen3-4B`).
- **Hardware assumptions:** Local CPU for unit/smoke tests and tiny-model gradient-proxy verification; NVIDIA L4 (24 GB VRAM, `bfloat16`, `batch_size = 16`, peak allocated VRAM `4147.35 MB`) for full 500-problem `MATH-500` evaluation.

## Research summary

### Problem
Test-time compute scaling methods (e.g., Best-of-N, Self-Consistency) typically allocate a uniform number of generation attempts across all queries. Uniform allocation wastes compute on easy problems and under-allocates on hard problems under a fixed total budget `T`. Prior adaptive allocators require training auxiliary difficulty probes (e.g., ~54 GPU-hours in Damani et al., 2025), adding training data and inference overhead.

### Method
DIPA estimates instance difficulty using training-free signals derived directly from the LLM, then allocates generation attempts via a Multi-Armed Bandit (MAB) policy with arm elimination upon success (Algorithm 1):

1. **Input-based proxies** (computed before generation to initialize `M_i`):
   - `MQL(x) = |x|` (query token length; zero extra forward pass)
   - `MEnt(x) = CE(x)` (input cross-entropy; single forward pass)
   - `MGN(x)` and `MVoG(x)` (mean and variance of input-embedding gradient norms; backward pass, implemented and unit-tested on CPU, excluded from full GPU sweeps due to compute cost)
2. **Generation-based proxies** (updated dynamically from accumulated generations `O_i` on failed attempts):
   - `MGL(O|x)` (mean generation length; paper's main proposal and top proxy on MATH-500; zero extra forward pass)
   - `MGC(O|x)` (generation consensus / majority-answer fraction; zero extra forward pass; paired with `input_proxy="const"` because `MGC` has no input-based counterpart in Table 1)
   - `MEnt(O|x)` (sequence cross-entropy; forward pass only)
   - `MGN(O|x)` and `MVoG(O|x)` (implemented and unit-tested on CPU)
3. **Probabilistic allocation (Algorithm 1):**
   - At each compute step `t in [T]`, active unsolved instances `x_k in X` are sampled with probability `P_k = (1 / M_k^lambda_t) / sum_{x_l in X} (1 / M_l^lambda_t)` computed in stabilized log-space, where higher `M_k` indicates greater difficulty.
   - Proxies negatively correlated with difficulty (e.g., `MGC`) are inverted (`1 / M_k`) before applying Eq. 4 (Appendix D.3).
   - Sampling temperature `lambda_t` supports both fixed values (`1`, `10` as ablated in Appendix E.7) and a batch-size normalized dynamic schedule `lambda_t = lambda_value * (|X| / N)` (default `lambda_value = 5.0`).
   - On verifiable math tasks, attempts are checked against ground truth extracted from `\boxed{}` (Appendix D.2 prompt template) using normalized LaTeX, numeric fractions/decimals, tuple comparison, and lazy `sympy` algebraic equivalence. Solved instances are removed from `X`; unsolved instances update `M_j <- M_gen(O_j | x_j)`.

### Expected behaviour
`DIPA(MGL)` is expected to achieve higher coverage (fraction of solved instances) than uniform budget allocation, greedy `Easy2Hard`, and `Hard2Easy` at equal total compute budget `T`, especially in tight (`b = 0.5`) and moderate-to-high (`b = 2..16`) budget regimes.

## Implementation

- **Input contract:** Instance set `X = {x_i}` (`N` prompts), integer total compute budget `T >= 0`, backbone generator interface (`temperature=0.7`, `top_p=0.8`, `repetition_penalty=1.05`, `max_new_tokens=512` per Appendix D.1), verifiable math checker `verify_math_answer`, proxy selectors (`input_proxy`, `gen_proxy`), temperature schedule (`lambda_mode`, `lambda_value`), and random seed.
- **Output contract:** Batch-level `BudgetAllocator` interface returning `AllocationReport` (per-instance allocations `T_i` summing to `<= T`, coverage, solved/unsolved IDs, total tokens, total latency) plus a `StoppingPolicy`-compatible adapter (`reset`, `should_stop`, `report`) matching `Adaptive_Stopping/README_adaptiveStopping.md`.
- **Determinism:** `DIPABudgetAllocator` is **stochastic** (probabilistic arm sampling plus `temperature=0.7` decoding) and reproducible under fixed random seeds (`42, 43, 44`). Baselines (`UniformAllocator`, `Easy2HardAllocator`, `Hard2EasyAllocator`) are deterministic given a fixed pool/seed.
- **Configuration:** `Adaptive_Stopping/configs/dipa.yaml`
- **Dependencies:** `python -m pip install -r Adaptive_Stopping/requirements.txt` (install PyTorch first via the official selector for your CUDA setup)
- **Reproduction command:**
  ```bash
  python3 -m Adaptive_Stopping.experiments.run_dipa_benchmark \
    --mode hf \
    --pool-size 32 \
    --gen-batch-size 16 \
    --dtype bfloat16 \
    --budget-multipliers 0.5,1,2,4,8,16 \
    --pool-path math500_L4_pool32.jsonl \
    --output-csv Performance_Metrics/results/dipa_math500_L4.csv \
    --overwrite-csv \
    --status partial
  ```
- **Tests:** `pytest -v Adaptive_Stopping/tests/ Backbones/tests/ Dataloaders/tests/ Performance_Metrics/tests/`

## Results

Paper-reported values below are from Table 5 (`MATH-500`, `Qwen2.5-Math-1.5B-Instruct`, 3-seed average, 8x H100 PCIe, `pool_size = 500`). Local values are 3-seed means (`seeds = [42, 43, 44]`) across all `N = 500` `MATH-500` test problems on a single **NVIDIA L4** (`bfloat16`, `batch_size = 16`, `pool_size = 32`, `lambda_mode = dynamic`, `lambda_value = 5.0`), logged in `Performance_Metrics/results/dipa_math500_L4.csv` (`90` rows, commit `46af136`).

| Source | Quality (Coverage on MATH-500) | Latency (median / p95) | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported (`DIPA(MGL)`) | `0.333` (`0.5N`), `0.565` (`2N`), `0.692` (`8N`), `0.765` (`32N`) | `1.62 s` (allocation overhead, Table 6) | not_measured | not_measured | not_measured | Table 5 & 6; 8x H100 PCIe, `pool_size = 500` |
| Paper-reported (`DIPA(MGC)`) | `0.241` (`0.5N`), `0.573` (`2N`), `0.689` (`8N`), `0.759` (`32N`) | `0.81 s` (allocation overhead, Table 6) | not_measured | not_measured | not_measured | Table 5 & 6; 8x H100 PCIe, `pool_size = 500` |
| Local baseline (`Uniform`) | `0.245` (`0.5N`), `0.479` (`1N`), `0.533` (`2N`), `0.576` (`4N`), `0.608` (`8N`), `0.637` (`16N`) | `1175.9 ms` / `1199.9 ms` | `433.0 tok/s` | `4147.35 MB` | not_measured | NVIDIA L4, `bfloat16`, `pool_size = 32` |
| Local baseline (`Easy2Hard`) | `0.294` (`0.5N`), `0.479` (`1N`–`16N` plateau) | `680.2 ms` / `1189.2 ms` | `435.7 tok/s` | `4147.35 MB` | not_measured | Locks onto unsolvable short arms past `1N` |
| Local baseline (`Hard2Easy`) | `0.000` (`0.5N`–`16N`) | `1221.4 ms` / `1223.7 ms` | `419.2 tok/s` | `4147.35 MB` | not_measured | Locks onto hardest unsolvable arm |
| Local method (`DIPA(MGL)`) | `0.292` (`0.5N`), `0.478` (`1N`), `0.548` (`2N`), `0.605` (`4N`), `0.642` (`8N`), `0.665` (`16N`) | `1180.9 ms` / `1202.5 ms` | `432.4 tok/s` | `4147.35 MB` | not_measured | Beats `Uniform` at `0.5N` (`+4.7%`), `2N`–`16N` (`+1.5%` to `+3.4%`) |
| Local method (`DIPA(MGC)`) | `0.222` (`0.5N`), `0.387` (`1N`), `0.553` (`2N`), `0.599` (`4N`), `0.622` (`8N`), `0.633` (`16N`) | `1178.1 ms` / `1204.1 ms` | `432.3 tok/s` | `4147.35 MB` | not_measured | Reproduces `2N` crossover (`0.553 > 0.548 > 0.533`) |

## Differences and limitations

- **No upstream implementation:** `https://github.com/allen4747/DIPA` contains only a `README.md` with no code or licence; this implementation is written from the paper specification.
- **`DIPA(MGC)` input prior (`const`):** `MGC(O|x)` has no input-based proxy counterpart in Table 1, and its inverted score `1 / MGC >= 1.0` starts at `1.0` on `m = 1` attempt. Pairing `mgc` with `mql` (`20..100` tokens) drops a failed arm's difficulty by ~50x on the first pull and locks onto one problem, so `dipa_mgc` uses `input_proxy="const"` (`1.0`).
- **Unspecified dynamic `lambda` proportionality constant:** Section 4.2 and Appendix E.7 specify `lambda proportional to |X|` without stating the exact proportionality constant. Setting `lambda = 1.0 * |X|` directly collapses DIPA into deterministic `Easy2Hard` at `N >= 100` (`lambda_t >= 57`). We implement scale-invariant `lambda_t = lambda_value * (|X| / N)` (default `lambda_value = 5.0` in `dipa.yaml`, matching Appendix E.7's `[1, 10]` scale) plus fixed `lambda` overrides via `--lambda-mode` / `--lambda-value`.
- **Hardware, pool size, and wraparound:** Paper experiments ran on 8x H100 (80 GB) with `pool_size = 500` and budgets up to `32N`. Our single-L4 reproduction uses `pool_size = 32` up to `16N` (`status: partial`). For `dipa_mgl`, pool wraparound is `0.00%` through `4N`, `1.47%` at `8N`, and `21.08%` at `16N` (making `0.665` a conservative lower bound). For `dipa_mgc`, problems that repeatedly generate the same wrong answer produce high consensus (`MGC ~ 1.0`), causing heavier wraparound at `4N` (`7.37%`), `8N` (`33.35%`), and `16N` (`56.23%`) and a slight dip below `Uniform` at `16N` (`0.633` vs `0.637`).
- **Out-of-scope models and tasks:** `Llama3.1-8B` (gated), `LiveCodeBench` (requires sandboxed code execution), and `GPQADiamond` (requires 4B reward model `Skywork-Reward-V2-Qwen3-4B`) are deferred to follow-up work.
- **Batch-level interface:** DIPA allocates budget across `N` instances jointly rather than making independent single-request early-exit decisions; the `StoppingPolicy` methods act as a wrapper around the batch allocator state.
- **Energy:** Reported as `not_measured` because Colab L4 power-sensor telemetry was not sampled.

## Reproduction checklist

- [x] Exact paper link and version verified
- [x] Base model and dataset revisions pinned (`Qwen/Qwen2.5-Math-1.5B-Instruct@aafeb0fc6f22`, `HuggingFaceH4/MATH-500@6e4ed1a2a79a` in `Backbones/registry.yaml`, `Dataloaders/registry.yaml`, and `configs/dipa.yaml`; both equal what `main` resolved to during the benchmark runs)
- [x] Baseline runs on the same setup
- [x] Tests and seed handling added (`157` CPU tests passing)
- [x] Resource and quality metrics reported (`Performance_Metrics/results/dipa_math500_L4.csv`)
- [x] Licence/data terms checked (`Qwen/Qwen2.5-Math-1.5B-Instruct`: Apache-2.0; `HuggingFaceH4/MATH-500`: MIT)
