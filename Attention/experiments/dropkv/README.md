# DropKV experiments

Code and results for our reproduction of DropKV (KV-cache eviction). The method is explained in [PAPER_IMPLEMENTATION_CARD.md](PAPER_IMPLEMENTATION_CARD.md); this file is only about running things.

## What is where

| Path | What it is |
|---|---|
| `../../methods/kv_eviction/` | the method: scoring, eviction on the Hugging Face cache, Lemma 1 helpers |
| `../../tests/test_dropkv.py` | CPU tests |
| `../../configs/dropkv/` | configs for the three runs below |
| `../../../Dataloaders/synthetic/passkey_retrieval.py` | prompt generator for the passkey task |
| `run_benchmark.py` | accuracy, latency and memory for full cache, DropKV, SnapKV and random |
| `run_perturbation.py` | how much each method changes the attention output and the next-token distribution |
| `make_plots.py` | draws the two figures used in the card |
| `results/`, `results_low_budget/`, `results_perturbation/` | the numbers we got (CSV) and the environment of each run (JSON) |

## Setup

Run everything from the repository root. Install PyTorch for your machine from <https://pytorch.org/get-started/locally/>, then:

```bash
pip install -r Attention/experiments/dropkv/requirements.txt
```

The model (Qwen2.5-0.5B-Instruct, about 1 GB) is downloaded from Hugging Face on the first run and is not part of the repository.

## Run

Tests (CPU, no download, about 10 seconds):

```bash
python -m pytest Attention/tests/test_dropkv.py -q
```

Accuracy, latency and memory. We ran this on a free Colab T4 (about 3 minutes for each config). The same script runs on a laptop CPU, but we did not run the full configs there (the prompt pass alone is more than 30 times slower than on the T4), and the GPU memory columns stay empty:

```bash
python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/qwen2_5_0_5b.yaml
python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/qwen2_5_0_5b_low_budget.yaml
```

Output perturbation. We ran this on a MacBook CPU (Apple M1, 8 GB), where it took about 7 minutes. Add `--num-samples 5 --context-tokens 1000` for a quick check that takes about a minute:

```bash
python -m Attention.experiments.dropkv.run_perturbation --config Attention/configs/dropkv/perturbation.yaml
```

A very small pipeline check (3 prompts, 500 tokens):

```bash
python -m Attention.experiments.dropkv.run_benchmark --config Attention/configs/dropkv/smoke_cpu.yaml
```

Figures (needs matplotlib):

```bash
python Attention/experiments/dropkv/make_plots.py
```

Each run writes its CSV and an `environment.json` (device, library versions, config) into the `output_dir` of its config. Running again overwrites those files, so use `--out some/other/folder` if you want to keep the committed ones.

## Things to know before reading the numbers

- The model, prompts and task are much smaller than in the paper. See the limitations section of the card.
- Timing covers the prompt pass, scoring, eviction and decoding. It does not include model loading or tokenisation.
- Peak memory is measured during the prompt pass, where eviction cannot help yet. The saving shows up in the cache size and in the decode-time peak.
- Energy was not measured.
