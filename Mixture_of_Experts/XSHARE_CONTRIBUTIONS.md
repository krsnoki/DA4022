# XShare implementation: contributions

Team responsible for the XShare implementation in this repository
(paper card: [XSHARE_PAPER_CARD.md](XSHARE_PAPER_CARD.md); issue
[#5](https://github.com/krsnoki/DA4022/issues/5)).

| Name | Roll number | Area |
|---|---|---|
| Shravya Lanka | 230101095 | Paper study and design |
| Freya Shingala | 230101094 | Core algorithms |
| Sharanya M | 230108048 | Model integration |
| Sanjeev | 230101089 | Data pipeline |
| Aishwarya | 230102086 | Experiments and measurement |
| Ayush Wani | 230101022 | Analysis and reporting |

## Areas

### Paper study and design (Shravya Lanka)
Reading the paper and its appendices, deciding how to resolve the points the
paper leaves open, and writing the research summary.
- `XSHARE_PAPER_CARD.md`: research summary, "Choices where the paper is ambiguous"

### Core algorithms (Freya Shingala)
The six selection algorithms of the paper (batch-aware, speculative-decoding-aware
and expert-parallel), routing inside the selected set, and their unit tests.
- `routing/xshare/selection.py`
- `tests/test_xshare_selection.py`

### Model integration (Sharanya M)
Applying XShare to a Hugging Face MoE model at decode steps, the backbone
wrapper and registry, the model cards, and the earlier gpt-oss-20b / vLLM attempt.
- `execution_layouts/xshare_hf.py`, `tests/test_xshare_hf.py`
- `Backbones/Generative_Pretrained/hf_causal_lm.py`, `Backbones/registry.yaml`
- `Backbones/Generative_Pretrained/olmoe_1b_7b_0125_instruct.md`, `Backbones/Generative_Pretrained/gpt_oss_20b.md`

### Data pipeline (Sanjeev)
The GSM8K loader (disjoint question subsets, batching, answer scoring), its
registry entry, dataset card and tests.
- `Dataloaders/evaluation/gsm8k.py`, `Dataloaders/registry.yaml`
- `Dataloaders/evaluation/gsm8k_test.md`, `Dataloaders/tests/`

### Experiments and measurement (Aishwarya)
The sweep runner and configurations, timing, request latency, memory and
energy measurement, the GPU runs and the overhead control runs.
- `experiments/xshare/run_sweep.py`, `experiments/xshare/smoke_test.py`
- `configs/xshare/gsm8k_sweep.yaml`, `configs/xshare/gsm8k_overhead.yaml`
- `tests/test_xshare_run_sweep.py`, `requirements-xshare.txt`

### Analysis and reporting (Ayush Wani)
Statistics, plots and simulations, the results and limitations in the paper
card, the results index, and the issue and pull request.
- `experiments/xshare/summarize.py`, `experiments/xshare/plot_results.py`,
  `experiments/xshare/simulate_variants.py`, `experiments/xshare/expected_activation.py`
- `tests/test_xshare_summarize.py`, `experiments/xshare/results/`
- `XSHARE_PAPER_CARD.md`: results, other experiments, differences and limitations
- `Performance_Metrics/README_metrics.md`: "Result files" index
