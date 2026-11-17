# BASTION reproduction (Speculative_Decoding)

Reproduction harness for *BASTION: Budget-Aware Speculative Decoding with
Tree-structured Block Diffusion Drafting* (arXiv 2605.29727). The official code
(MIT, https://github.com/kaist-ai-osi-lab/BASTION) is pinned as the submodule
`upstream/` at commit 1989b80 and is never modified. The scripts here import the
upstream algorithm code and add multi-dataset runs, fixed budgets, logging,
analysis and tests. Results and the comparison with the paper are in
[`REPORT.md`](REPORT.md) and [`PAPER_CARD.md`](PAPER_CARD.md).

| Path | Purpose |
|---|---|
| `run_benchmark.py` | Harness derived from `upstream/bastion/benchmark.py::_run_transformers` (AR / DFlash / BASTION, T=0, batch 1) |
| `repro_lib.py` | GPU registration, tree-size recorder, upstream-identical sample selection, JSONL I/O |
| `analysis/aggregate.py` | Speedup / AAL / tree size per (model, dataset, method), DA4022 CSV, Table-1 markdown |
| `analysis/check_lossless.py` | E3: exact-match and first-divergence statistics vs AR |
| `analysis/e1_extras.py` | E1 vs paper Table 1 per dataset, tree-size distribution, bootstrap intervals, resume check |
| `analysis/score_math.py` | Boxed-answer accuracy on gsm8k / math500 / aime25 (quality sanity check) |
| `analysis/latency_model.py` | E5: uncalibrated roofline vs calibrated fit vs measured (RMSE) |
| `analysis/plots.py` | Speedup bars, budget sweep, latency-model figures |
| `scripts/` | `run_pilot.sh`, `run_e1.sh`, `run_sweep.sh`, `run_analysis.sh` |
| `tests/` | CPU-only pytest suite (synthetic data, tiny random models) |

## How to run

All commands are run from this directory and use `.venv/bin/python`.
Logs and raw records go to `runs/` (git-ignored). Analysis outputs go to `results/`.

```bash
# 0. CPU tests (no GPU, no downloads; takes a few seconds)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q

# 1. Pipeline check: 2 samples/dataset, 256 new tokens, AR + DFlash + BASTION (adaptive and N=64)
scripts/run_pilot.sh qwen3-4b

# 2. E1/E3: Table-1-style run per target (defaults: 64 samples/dataset, aime25=30, mt-bench=40,
#    max_new_tokens 2048, warm-up 1). Rerunning resumes from runs/e1_<key>.jsonl.
scripts/run_e1.sh qwen3-4b
scripts/run_e1.sh qwen3-8b
#    e.g. smaller caps: MAX_SAMPLES=32 CAP_AIME25=30 CAP_MTBENCH=20 scripts/run_e1.sh qwen3-8b

# 3. E4: budget sweep (fixed N in {32..1024} + adaptive with warm profile, then adaptive with
#    upstream's cold profile; BASTION only, 16 samples/dataset).
#    Needs runs/e1_<key>.jsonl for the AR reference and the MT-Bench turn-2 history.
scripts/run_sweep.sh qwen3-8b

# 4. Analysis (CPU): aggregate, lossless check, math scores, latency model (E5), figures
scripts/run_analysis.sh
```

Direct harness use (same flags the scripts pass):

```bash
.venv/bin/python run_benchmark.py --model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --datasets gsm8k math500 --max-samples 64 --cap aime25=30 \
  --modes ar dflash bastion --budgets adaptive 64 256 \
  --max-new-tokens 2048 --temperature 0 --seed 0 --warmup 1 \
  --out-dir runs --run-name my_run          # add --dry-run to print the sample selection only
```

`llama3.1-8b` is still a valid model key, but it is not in the current scope. The model is gated.

## Behaviour that matters for interpreting results

- **Sample selection** matches upstream `_limit_dataset` exactly: `random.seed(0)`, shuffle, keep the first k. The shuffle only happens when k is smaller than the dataset size. The tests check that loading models and tokenizers does not consume Python `random` state, which this equivalence depends on. Each record stores `orig_idx` (the problem's index in upstream's dataset order). Cross-run matching uses `orig_idx`, because aime25 keeps its original order at cap 30 but is shuffled at cap 16.
- **Fixed budget N** is passed to upstream through `bastion_generate(min_tree_size=N, max_tree_size=N)`, which is part of upstream's public signature. With min = max = N the cost-based stopping rule never fires, the builder terminates at exactly N nodes, and the return-size bucket (step = min_tree_size = N) never truncates. The harness refuses to run if `ADAPTIVE_MIN_TREE_SIZE` or `ADAPTIVE_BUCKET_STEP` is set, because these would silently change the result. Every BASTION step logs its realized size (`tree_sizes`) through a wrapper around the upstream builder. Fixed-budget records also store `tree_size_mismatch_steps`.
- **Controller profile constants (`--profile-mode`)**: the adaptive controller's stopping rule uses a profiled draft latency and non-verification overhead. Upstream profiles them once, cold (3 trials, no warm-up), and caches the result in `upstream/cache/calibration_<gpu>_<model>.profile.json`. In the GPU pilot this cold profile overstated the overhead by roughly 180x, which led to oversized trees. `--profile-mode upstream` (the harness default) keeps upstream's behaviour. `--profile-mode warm` (used by the scripts) runs upstream's `profile_tree_adaptive_constants` once and discards the result, runs it again with the same arguments, and keeps the second result. That result is cached separately in `calibration_<gpu>_<model>.profile.warm.json`, and upstream's file is never written in this mode. The roofline calibration is the same in both modes. Every record stores `profile_mode` and the constants it used. For adaptive BASTION the profile mode is part of the resume key and the aggregation key, so the two variants never mix. They are labelled "BASTION (adaptive)" (warm) and "BASTION (adaptive, upstream cold profile)". Records written before this option existed count as upstream. Fixed-budget and AR/DFlash records are keyed without the profile mode, because their output does not depend on it. `run_sweep.sh` runs fixed budgets plus warm adaptive first, then upstream adaptive, both into the same JSONL.
- **Adaptive trees**: with upstream defaults (min 32, max 8192, `fixed_interval` buckets), the returned tree has `max(32, 32*floor(N*/32))` nodes. Here N* is the first N at which the estimated speedup stops increasing. Realized adaptive sizes are therefore multiples of 32 (tested in `tests/test_cost_model.py`).
- **Speedup** = mean per-record TPOT(AR) / mean per-record TPOT(method), as upstream prints it. Both means are taken over records with an AR counterpart for the same (model, dataset, orig_idx, turn) and the same prompt hash. **AAL** pools all steps. The **Average** row is the unweighted mean over datasets, which matches the paper's Table 1 average. The **Pooled** row pools all records.
- **GPU**: the RTX PRO 4500 Blackwell is not in upstream's cost-model GPU table. It is registered at runtime as `b4500` by `tools/bastion_local.py` (assumed ~202 TFLOPS dense bf16 = 1617 sparse FP4 TOPS / 8, the same ratio that maps the RTX PRO 6000's 4000 FP4 TOPS to upstream's 504 TFLOPS; 896 GB/s from NVIDIA's product page). The harness reuses that registration, so cache filenames match. These assumed specs affect the uncalibrated roofline directly. The per-(GPU, model) linear calibration absorbs a constant scale error.
- **Differences from upstream's script**: single process and single GPU (no torchrun sharding); several datasets per process; optional warm-up (`--warmup`, scripts use 1, upstream has none); if the profile cache is missing it is created from the first sample of the first dataset in the run; `--profile-mode warm` (see above) is a deliberate deviation used by the scripts; `--history-jsonl` lets runs without AR reuse AR outputs from E1 as MT-Bench turn-2 history.
