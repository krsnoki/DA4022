# Staircase streaming (MoA) — reproduction

We have re-implemented [Staircase Streaming for Low-Latency Multi-Agent Inference](https://openreview.net/forum?id=9VqvhSZwN6) (arXiv 2510.05059). It is compared against a normal Mixture-of-Agents baseline on the same hardware. See [PAPER_CARD.md](PAPER_CARD.md) for the method, the paper's numbers and the differences.

**Status:** `partial`. Implemented, tested, and benchmarked on 1× RTX 4090 with vLLM 0.34.0 (2026-11-24, paper-sized prompt set + whole GSM8K test set). Median TTFT is 4.28 s for baseline MoA vs 0.097 s for staircase (f8 r2), with GSM8K accuracy unchanged (0.661 vs 0.671, p = 0.45). Results are in [PAPER_CARD.md](PAPER_CARD.md#results).

## Run the tests (any machine, CPU only)

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r Serving/staircase_streaming/requirements.txt
python -m pytest Serving/staircase_streaming/tests -q
```

## Dry-run the whole benchmark on the mock backend (CPU, ~1 min)

```bash
python -m Serving.staircase_streaming.run_benchmark --config Serving/staircase_streaming/configs/mock.yaml
```

## Full GPU run

See [remote/RUNBOOK.md](remote/RUNBOOK.md). The one-command version, on a Linux GPU machine from the repo root:

```bash
bash Serving/staircase_streaming/remote/run_all.sh Serving/staircase_streaming/configs/full.yaml
```

## Files

| File | Purpose |
|---|---|
| `backends.py` | `Backend` protocol, `TokenEvent`, `MockBackend`, `OpenAICompatBackend` (vLLM) |
| `prompts.py` | Aggregator template (paper Table 2) and prefix-cache template (Table 6) |
| `proposers.py` | Thread-based streaming proposers with a shared condition variable |
| `moa_baseline.py` | Normal MoA: wait for all proposers, then aggregate |
| `staircase.py` | Staircase streaming (Algorithm 1, Appendix A hyperparameters) |
| `timing.py` | Call/request records, percentiles, sampled GPU memory |
| `quality.py` | GSM8K answer extraction and exact match |
| `config.py` | YAML config loading; builds backend / MoA / staircase objects |
| `plots.py` | TTFT box plot, TTFT vs first-chunk size, aggregator prompt tokens |
| `remote/` | `run_all.sh` (one-command GPU run) and `RUNBOOK.md` |
| `launch_vllm.py` | Start/stop one vLLM server per model from the config |
| `run_benchmark.py` | Single reproduction command: warm-up, baseline vs staircase, sweep, CSV, plots |
| `env_check.py` | Hardware/software report |
| `configs/` | `full.yaml` (reported results: 64 + 64 prompts, whole GSM8K test set; 1 GPU ≥ 24 GB), `default.yaml` (same settings, smaller workload), `kaggle_2xT4.yaml`, `mock.yaml` |
