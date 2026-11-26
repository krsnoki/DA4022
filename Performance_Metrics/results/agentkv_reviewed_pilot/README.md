# Reviewed-scoring AgentKV pilot

All 35 measured episodes completed without recorded runtime failures or truncated
turns. The export matches the archived reviewed source bundle and passes the
sample-coverage, scoring, summary and cache-bound audit.

## Pretrained integration gate

[verification.json](verification.json) records a passing check on the pinned
Qwen3-0.6B checkpoint, float16 on CUDA. For the tested identical input chunks,
full/recent/AgentKV with eviction disabled each had maximum absolute logit error
0.0 versus unmodified Hugging Face execution; final K/V tensors matched within
the declared tolerances. Next-token argmax checks also passed. Forced compression
processed 61 logical tokens, retained eight entries, and completed four compression
events with finite continuation and active phase buffers.

This is evidence for the tested integration paths, not a proof of all model
configurations, all sequences, or exact upstream serving-system parity.

## Setup

- Tesla T4; torch 2.14.1+cu130; transformers 4.57.1; Python 3.13.16.
- Qwen/Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`.
- Five synthetic v1 episodes, seed 42, 48 distractors, batch one, greedy decoding.
- Output cap 512, query capacity 32, decode compression interval 128.
- `upstream_mean` profile, sink16/recent8, pooling kernel 1.
- Source SHA-256: `935778a060bf2620ca3df4d6c846b86b3461bb44f726cb36693e1d80dd199144`.
- The recorded commit is the original base commit; the source hash identifies
  the actual archived implementation, which preceded the first contribution commit.

Scoring and protection settings differ from the two original pilots. Do not pool
their records or attribute between-profile differences solely to phase awareness.

## Results

| Method | Budget | Correct | Decode tokens/s | Median episode seconds | Max final KV MB | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|---:|
| Full cache | unlimited | 4/5 | 25.19 | 9.71 | 159.88 | 1374.86 |
| Recent | 256 | 3/5 | 20.67 | 13.69 | 41.40 | 1373.71 |
| AgentKV | 256 | 1/5 | 21.04 | 11.84 | 42.55 | 1373.71 |
| Recent | 512 | 4/5 | 21.15 | 11.04 | 71.79 | 1373.71 |
| AgentKV | 512 | 5/5 | 20.71 | 11.36 | 72.48 | 1373.71 |
| Recent | 1024 | 4/5 | 20.85 | 12.33 | 130.97 | 1373.71 |
| AgentKV | 1024 | 5/5 | 20.85 | 12.09 | 130.97 | 1373.71 |

MB is decimal. Final KV is not peak memory and excludes query-buffer storage.
Episode latency spans both turns and includes compression; decode throughput
uses actual generated tokens. Output lengths differ, and order/clock effects
were not controlled by repeated timing trials. Energy is unmeasured.

## Interpretation

The revised profile shows a one-example AgentKV accuracy advantage over recent
at 512/1024, and a two-example disadvantage at 256. Five examples and one seed
do not support a general superiority claim. Full cache's one failure is missing
the final-answer delimiter. Both compressed methods remain slower than full
cache, and peak allocated GPU memory remains essentially unchanged.

Every first live response contains the expected code (including reasoning) in
every configuration. Therefore later retrieval can use those repeated assistant
tokens as well as the original tool observation. This experiment cannot isolate
retention of the original tool-response keys. The new diagnostic exposes that
limitation; it does not alter the strict primary score.

## Decision and completed follow-up

The operational/integration gates pass, and no pilot truncation remains. The
previously prepared evaluation completed without tuning: seeds 101/202/303, 10
examples each, 24/48/72 distractors respectively, and all seven method/budget
configurations. This is 210 measured episodes plus warm-ups. See the
[fixed evaluation report](../agentkv_evaluation/README.md). Keep per-length
strata and report the narrow synthetic task family honestly. Do not treat these
three data seeds as repeated stochastic decoding trials.

The evaluation notebook embedded the exact reviewed source archive, reran the
pretrained integration gate, and invoked its existing `run_suite` entrypoint.
No algorithm, prompt or scoring changes were introduced between this pilot and
the larger evaluation. Previous results and source archives remain unchanged.

## Artifacts

- [summary.json](summary.json): configuration-level measurements.
- [samples.jsonl](samples.jsonl): original answers, phase counts and diagnostics.
- [metadata.json](metadata.json): recorded config, runtime and source identity.
- [verification.json](verification.json): pretrained numerical gate.
- [audit.json](audit.json): integrity and metric checks.

```bash
python -m Performance_Metrics.audit_agentkv \
  --results artifacts/agentkv-reviewed-results.zip \
  --source artifacts/agentkv-reviewed-source.zip \
  --out Performance_Metrics/results/agentkv_reviewed_pilot
```

The audit does not rerun inference. The original ZIP also retains the full
environment package listing. The separate verification file was copied without
modification from that ZIP and matched to the manifest source/model revision.
