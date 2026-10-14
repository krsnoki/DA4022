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

## Implementation checklist

- [ ] Describe the stop signal and the point in the forward/generation loop where it is evaluated.
- [ ] Identify any extra trained weights and the data used to fit them.
- [ ] Include a no-op/full-compute baseline and a simple threshold baseline where meaningful.
- [ ] Test budget edge cases (zero/minimum/maximum budget and missing telemetry).
- [ ] Record whether stopping is deterministic.
- [ ] Add quality-vs-latency plots or a machine-readable table in `Performance_Metrics/`.
