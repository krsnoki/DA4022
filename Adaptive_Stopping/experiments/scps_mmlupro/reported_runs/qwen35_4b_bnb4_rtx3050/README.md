# Reported run: SCPS and baselines, Qwen3.5-4B NF4, MMLU-Pro (256 questions), RTX 3050 Laptop 4 GB

Outputs of `Adaptive_Stopping/configs/scps_mmlupro_qwen35_4b_bnb4.yaml`, kept so the reported numbers can be re-derived and re-tested without rerunning the model. Per-question files hold question IDs, gold letters and answer classes only (no question text).

| File | Content |
|---|---|
| `per_question_<method>.jsonl` | one line per question: answers, Pass@16, distinct answers, majority answer, tokens, rollouts, rounds, stop reason, wall time; `option_topk` also stores the root option probabilities |
| `run_meta_<method>.json` | environment, backbone metadata, model load time, peak CUDA memory, commit |
| `summary.json`, `results.csv` | aggregate metrics, paired bootstrap, top-K control, per-category counts; rows in the `Performance_Metrics` schema |
| `scps_metadata_rerun/` | 32-question SCPS rerun used for SCPS peak memory; its portfolios match the main run on all 32 questions |
| `env_*.txt`, `pip_freeze.txt` | GPU, driver and package versions (converted from UTF-16 PowerShell output to UTF-8) |

## Provenance

| Method | Session (local time, IST) | Code | Notes |
|---|---|---|---|
| `scps` | 2026-11-26 20:25 to 2026-11-26 22:43 | branch commit 531a0f6 | before per-method metadata existed; metadata from `scps_metadata_rerun` (commit 52e2f49) |
| `option_topk`, `self_consistency`, `tree_prefix`, `semantic_pruning` | 2026-11-27 01:45 to 04:20 | 52e2f49 | one process, model loaded once |
| `scps_metadata_rerun` | 2026-11-27 04:21 to 04:30 | 52e2f49 | 32 questions |

Not included: a first self-consistency attempt that drew all 16 samples in one batch and ran out of GPU memory at question 104. It was discarded and rerun with 4 samples per call, so every reported self-consistency result uses one setting.

`summary.json` and `results.csv` were rebuilt from the per-question files with `run.summarize` after copying in the SCPS metadata; no per-question value was changed. Rebuild the tables with:

```bash
python -m Adaptive_Stopping.experiments.scps_mmlupro.report Adaptive_Stopping/experiments/scps_mmlupro/reported_runs/qwen35_4b_bnb4_rtx3050 --latency-subset Adaptive_Stopping/experiments/scps_mmlupro/reported_runs/qwen35_4b_bnb4_rtx3050/scps_metadata_rerun
```

## Model and data versions

The configs used `revision: null`, which resolves to the Hub's `main`. Both repositories were last modified long before the run, so the downloaded versions are `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (last modified 2026-03-02) and `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be` (last modified 2026-05-02). The config now pins both.

## Commit IDs

The commit IDs recorded in `run_meta_*.json` and `results.csv` (`531a0f6`, `52e2f49`) are the IDs the branch had when the runs were made. The branch's commit messages were edited afterwards (trailer lines removed), which changes commit IDs but not file contents. They correspond to these commits:

| Recorded ID | Commit subject |
|---|---|
| `531a0f6` | SCPS run script: --resume, per-question progress output |
| `52e2f49` | SCPS experiment: option top-K control, per-method run metadata, multi-run summary |
