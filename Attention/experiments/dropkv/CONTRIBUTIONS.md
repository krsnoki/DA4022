# Division of work

Team for the DropKV contribution (issue #6). The work was split by part of the project, so that each of us owns one piece and can explain it.

| Member | Roll no. | Part | Main files |
|---|---|---|---|
| Deval Singhal | 230101035 | Method core and project coordination. Wrote the DropKV, SnapKV and random scoring and the token selection, the Lemma 1 helpers, and handled the issue, the fork, the branch and the pull request. | `Attention/methods/kv_eviction/scoring.py`, `Attention/methods/kv_eviction/perturbation.py` |
| Jayant Kumar | 230101050 | Hugging Face integration. Recording the window queries with hooks, grouped-query attention, shortening the cache in place, and decoding on the shortened cache with the original position ids. End-to-end tests on a tiny model. | `Attention/methods/kv_eviction/eviction.py`, end-to-end tests in `Attention/tests/test_dropkv.py` |
| Sonal Mangesh Wagh | 230108054 | Benchmark and metrics. Accuracy, latency, throughput and memory runs on the Colab T4, the configs, and the result rows in the shared schema. | `Attention/experiments/dropkv/run_benchmark.py`, `Attention/configs/dropkv/qwen2_5_0_5b*.yaml`, `Performance_Metrics/results/dropkv_qwen2_5_0_5b.csv` |
| Sristi Vats | 230122057 | Output-perturbation experiment. The attention-output shift and next-token KL measurement, the laptop run, the figures, and the analysis of why DropKV and SnapKV did not separate. | `Attention/experiments/dropkv/run_perturbation.py`, `Attention/experiments/dropkv/make_plots.py`, `Attention/configs/dropkv/perturbation.yaml`, `Attention/experiments/dropkv/results_perturbation/` |
| Shourya Veer Singh | 230106069 | Data and documentation. The passkey prompt generator, the model and dataset cards, the paper card, the run README, and the tests for the scoring and the paper's theory (Lemma 1, Theorem 1 on small cases). | `Dataloaders/synthetic/passkey_retrieval.*`, `Backbones/Generative_Pretrained/qwen2_5_0_5b_instruct.md`, `Attention/experiments/dropkv/PAPER_IMPLEMENTATION_CARD.md`, `Attention/experiments/dropkv/README.md` |

## Shared by all five

- Reading the DropKV paper and agreeing on the scope (small model, synthetic task, no Triton kernels).
- Running the test suite and reading the final paper card before the pull request.
- Being able to explain the whole method, not only our own part.

## A note on the commit history

All commits come from one GitHub account because the branch was pushed from a single fork. The table above is the actual split of the work, not the commit authorship.
