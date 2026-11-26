# Adaptive Stopping and Variable-Compute Inference

## Purpose

Methods that decide **whether to continue computation**: early exit, learned stopping, adaptive depth/recursion, or allocation of a finite reasoning/decoding budget. The module should consume a common model interface and return the prediction plus stopping metadata.

## Scope

In scope: early-exit heads, loop stopping, confidence/proxy-based stop policies, dynamic depth, token/step budgets, budget-aware inference policies.

Out of scope: general benchmark code (see `Performance_Metrics/`), attention kernel implementations (see `Attention/`), or model weights themselves (see `Backbones/`).

## Suggested structure

```text
Adaptive_Stopping/
├── README_adaptiveStopping.md
├── baselines/
├── methods/
│   ├── confidence_threshold/
│   ├── learned_exit/
│   └── budget_policy/
├── configs/
├── tests/
└── experiments/
```

Use one directory per method only once there is an owner. Avoid paper-specific copies of shared stopping utilities.

## Paper shortlist

- [TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning](https://openreview.net/forum?id=GwTFovD6SM)
- [Step-Tagging Early-Stopping: Toward controlling the generation of Language Reasoning Models through black-box step monitoring](https://openreview.net/forum?id=bNn5OusO22)
- [Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping](https://openreview.net/forum?id=xExWu1Bhuw)
- [Learning Adaptive Reasoning Budgets via Constraint-Rectified Training](https://openreview.net/forum?id=1inovdFo6Z)
- [DIPA: Difficulty-Informed Probabilistic Allocation of Test-Time Compute via Training-Free Proxies](https://openreview.net/forum?id=9Joja0tx5c)
- [Block-Level Recursion: Adaptive Test-Time Routing in Large Language Models](https://openreview.net/forum?id=hLFRcxXKxY)
- [Learning Adaptive LLM Decoding](https://openreview.net/forum?id=14yhYy9m7S) — adaptive decoding policy.

Browse all accepted work in the [official paper index](https://adaptfm.gitlab.io/accepted-papers). Paper assignment is based on subject matter and may be cross-listed.

## Shared method contract

Every method should expose a comparable interface, for example:

```python
class StoppingPolicy:
    def reset(self, request_metadata=None): ...
    def should_stop(self, state, metrics, budget) -> bool: ...
    def report(self) -> dict: ...
```

The exact implementation can differ by model type; the contract must specify which state/metrics are required and whether they are available without an extra forward pass.

## Required evaluation

Report task quality against a no-stopping baseline; average and tail latency where appropriate; average depth/steps/tokens; peak memory; stop rate; and quality vs. compute-budget curves. Use identical prompt/input sets and seeds for baseline/method runs. If the policy adds a classifier or probe, include its compute and memory cost.

## Implemented methods

### Step-Tagging Early-Stopping (ST-ES)

- **Paper:** [OpenReview `bNn5OusO22`](https://openreview.net/forum?id=bNn5OusO22), [arXiv 2512.14332](https://arxiv.org/abs/2512.14332).
- **Paper card:** [`methods/step_tagging/PAPER_CARD.md`](methods/step_tagging/PAPER_CARD.md). Tracking issue: krsnoki/DA4022#24.
- **Stop signal:** a black-box rule evaluated once per completed reasoning step. A step ends at the token that closes a `"\n\n"` delimiter. The rule tags the step's text with a ReasonType step type and stops when the count of steps of the target type τ\* exceeds δ.
- **Cost:** no extra forward pass of the backbone. The policy only reads generated text, and the tagger runs on CPU. After a stop, the model gets an exit prompt and up to 100 greedy answer tokens.
- **Determinism:** the policy is deterministic given the tagger.

| Path | What it holds |
|---|---|
| `common/policy.py` | `StoppingPolicy` contract, `StepState`, shared `max_steps` / `max_tokens` budget caps |
| `common/steps.py` | Online step segmentation (identical to `str.split("\n\n")` for any chunking) and an incremental detokeniser |
| `common/replay.py` | Offline replay of step-level policies over cached traces; fast paths for threshold sweeps |
| `common/cached_generation.py` | Batched sampling of full traces for replay (per-row prefill, continuation rounds) |
| `common/live_generation.py` | Live generation with a policy as stopping criterion, followed by answer forcing |
| `common/forced_answer.py` | Exit prompt and 100-token answer forcing, batched or with shared-prefix KV reuse |
| `methods/step_tagging/` | ReasonType taxonomy, rule-based step tagger, ST-ES policy, calibration at 85/90/95% |
| `baselines/` | `NoStopping` (full generation) and `StepCountStopping` (the paper's count baseline) |
| `experiments/st_es/` | `generate`, `replay_sweep`, `live_benchmark`, `validate_tagger`, `run_all`, `make_report` (tables, plots, metrics rows) |
| `configs/st_es/` | GSM8K and MATH-500 experiment configs |
| `tests/` | CPU tests: canned traces plus a tiny random-model end-to-end run |

The step-level framework in `common/` is documented in [`common/README.md`](common/README.md).

```bash
pip install -r Adaptive_Stopping/requirements-st_es.txt   # after installing PyTorch
python -m pytest Adaptive_Stopping/tests Dataloaders/tests Performance_Metrics/tests -q
python -m Adaptive_Stopping.experiments.st_es.run_all --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
python -m Adaptive_Stopping.experiments.st_es.make_report --config Adaptive_Stopping/configs/st_es/gsm8k.yaml
```

## Implementation checklist

- [ ] Describe the stop signal and the point in the forward/generation loop where it is evaluated.
- [ ] Identify any extra trained weights and the data used to fit them.
- [ ] Include a no-op/full-compute baseline and a simple threshold baseline where meaningful.
- [ ] Test budget edge cases (zero/minimum/maximum budget and missing telemetry).
- [ ] Record whether stopping is deterministic.
- [ ] Add quality-vs-latency plots or a machine-readable table in `Performance_Metrics/`.
