# AgentKV fixed evaluation

This report evaluates the reviewed AgentKV implementation after the five-example
pilot. The experiment was frozen before execution: three seeded, length-stratified
synthetic retrieval groups, ten episodes per group, and one full-cache plus six
compressed-policy configurations per episode. That produces 210 measured episodes
and 21 unmeasured warm-ups. There were no recorded runtime failures or truncated
turns.

The uploaded export passed the per-stratum audit: source hash, configuration,
coverage, strict-answer score, summaries, and cache bounds all match. The embedded
pretrained numerical gate also passed: its no-eviction adapters had zero maximum
tested logit difference from the unmodified Qwen3 path and matched cached K/V
tensors for the tested chunks. These checks establish the tested integration path;
they do not establish parity with every upstream serving path or model.

## Frozen setup

- Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  float16, batch one, greedy decoding, Tesla T4, CUDA 13.0, torch 2.14.1+cu130,
  transformers 4.57.1, Python 3.13.16.
- `upstream_mean` score profile; query capacity 32, sink 16, recent protection 8,
  pooling disabled, compression interval 128, output cap 512.
- Seeds 101/202/303 use 24/48/72 distractors respectively. Each has ten examples
  and budgets 256, 512, and 1024. They are context-length strata, not independent
  stochastic decoding repetitions.
- Source SHA-256: `935778a060bf2620ca3df4d6c846b86b3461bb44f726cb36693e1d80dd199144`.

## Aggregate results

| Method | KV budget | Correct | Decode tokens/s | Median episode seconds | Max final KV MB | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|---:|
| Full cache | unlimited | 20/30 | 26.35 | 8.92 | 212.98 | 1421.75 |
| Recent-query | 256 | 9/30 | 21.22 | 11.99 | 43.70 | 1422.50 |
| AgentKV | 256 | 2/30 | 20.88 | 11.42 | 43.58 | 1422.50 |
| Recent-query | 512 | 17/30 | 21.26 | 11.34 | 73.17 | 1422.50 |
| AgentKV | 512 | 15/30 | 21.28 | 11.37 | 73.29 | 1422.50 |
| Recent-query | 1024 | 22/30 | 20.90 | 11.18 | 131.55 | 1422.50 |
| AgentKV | 1024 | 20/30 | 21.36 | 11.29 | 131.55 | 1422.50 |

The full-cache baseline is fastest and has lower median latency in this reference
implementation. Cache compression reduces final KV storage, but peak allocated GPU
memory is dominated by model/runtime allocations and remains similar. Energy was
not measured. Final KV is not peak memory and excludes query-buffer storage.

## Per-stratum strict accuracy

| Distractors / seed | Full | Recent 256 | AgentKV 256 | Recent 512 | AgentKV 512 | Recent 1024 | AgentKV 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 / 101 | 6/10 | 2/10 | 0/10 | 2/10 | 4/10 | 6/10 | 6/10 |
| 48 / 202 | 9/10 | 4/10 | 1/10 | 9/10 | 8/10 | 9/10 | 9/10 |
| 72 / 303 | 5/10 | 3/10 | 1/10 | 6/10 | 3/10 | 7/10 | 5/10 |

The larger fixed evaluation does not reproduce the five-example pilot's apparent
AgentKV advantage at 512 or 1024. Recent-query eviction scores higher at every
budget here; full cache ties AgentKV at 1024 and exceeds it elsewhere. This is a
narrow synthetic task, not a general claim that AgentKV is ineffective.

Every initial live response mentioned the expected code in all 210 measured runs.
Later retrieval may therefore use a repeated assistant token as well as the
original tool observation. The task cannot isolate original tool-response-key
retention. Strict failures include wrong final codes and missing `</think>` final
delimiters; the latter receive no primary-score credit by design.

At budget 1024, all seed-101 samples fit without compression. The audit permits
zero eviction only when the logical cache fits the configured budget and the cache
is unchanged; other compressed samples recorded eviction events.

## Files and reproduction

- [aggregate.json](aggregate.json) contains the combined result table.
- `seed_101`, `seed_202`, and `seed_303` contain the audited raw samples,
  configuration metadata, and per-configuration summaries.
- [verification.json](verification.json) is the pretrained numerical gate copied
  byte-for-byte from the cloud export.

```bash
python -m Performance_Metrics.audit_agentkv \
  --results artifacts/agentkv-evaluation-results.zip \
  --source artifacts/agentkv-reviewed-source.zip \
  --prefix seed_101/ \
  --out Performance_Metrics/results/agentkv_evaluation/seed_101
```

Repeat with `seed_202/` and `seed_303/`. The audit validates the recorded export;
it does not rerun GPU inference.
