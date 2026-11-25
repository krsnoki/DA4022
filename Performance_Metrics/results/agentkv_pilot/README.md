# AgentKV cloud pilot: five synthetic episodes

This is an exploratory partial reproduction, not a claim that AgentKV outperforms
recent-query eviction. The uploaded export contains **35 successful execution
records across seven configurations**, five examples each, with zero recorded
runtime failures. Incorrect answers are counted separately from runtime failures.

## Provenance and checks

- Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Tesla T4, float16, batch one; Python 3.13.16, torch 2.14.1+cu130,
  transformers 4.57.1, CUDA runtime 13.0 (as recorded by the cloud runtime).
- Synthetic agent memory v1; seed 42; five identical starting episodes per configuration.
- 192 generated tokens maximum per turn; 48 distractor records; greedy decoding.
- Source SHA-256: `f73578c1db67e9e4acfbd853bd77c572aa02e9c03d5ec7ca7f16f1027c2317c3`.
- Recorded commit is the base of an uncommitted working tree, not a commit
  containing the implementation. The source hash identifies the executed files.

The audit matched the cloud source hash against the original uploaded source ZIP,
checked sample coverage and duplicates, recomputed exact-match scores and key
summary metrics, and verified reported full-cache and compression-budget bounds.
This audits the export; it does not independently rerun inference or prove full
paper fidelity. The result archive contains no cloud pytest output.

## Measured results

| Method | KV budget | Correct | Decode tokens/s | Median episode time (s) | Max final KV (MB) | Peak allocated GPU memory (MB) | Final turns capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full cache | unlimited | 4/5 | 24.50 | 10.03 | 159.42 | 1373.79 | 1/5 |
| Recent | 256 | 0/5 | 20.51 | 12.65 | 43.58 | 1373.71 | 0/5 |
| AgentKV | 256 | 0/5 | 20.59 | 13.02 | 42.55 | 1373.71 | 1/5 |
| Recent | 512 | 4/5 | 20.66 | 12.55 | 73.17 | 1373.71 | 0/5 |
| AgentKV | 512 | 4/5 | 20.73 | 11.95 | 72.25 | 1373.71 | 1/5 |
| Recent | 1024 | 4/5 | 20.77 | 12.17 | 130.97 | 1373.71 | 1/5 |
| AgentKV | 1024 | 4/5 | 21.02 | 12.16 | 130.97 | 1373.71 | 1/5 |

MB means decimal bytes / 1,000,000. Episode time includes both prompt phases and
both generated turns; it excludes tokenization and model loading. Decode time
includes periodic eviction and caching the last generated token. Different
policies can generate different text and output lengths, so this is not a fixed
token-workload speed comparison. All variants ran in a fixed order in one run;
repeatability and order effects have not been evaluated. No p95 is reported for
five examples. Energy was not measured.

## Interpretation

- Both compressed methods tie in aggregate accuracy at every tested budget.
  Five examples and one seed provide no evidence of a phase-awareness advantage.
- At 512 tokens, the methods do not fail on the same example: recent fails on
  sample 4, AgentKV on sample 3, and full cache on sample 1 (zero-based IDs).
- The full-cache failure is capped mid-reasoning while referring to the correct
  code, before producing a final answer. It remains incorrect under the declared
  evaluator; do not retroactively extract a code from reasoning and count it.
- At budget 256, incorrect recalled codes and missing final-answer formatting
  occur even without hitting the cap. A longer output limit alone is therefore
  not established as a solution to aggressive-compression failures.
- Final KV storage is lower under eviction. AgentKV at 512 uses at most 72.25 MB
  of final KV in these episodes versus 159.42 MB for full cache (about 55% less),
  plus 3.67 MB of query buffers and position metadata. These are final-state
  measurements, not equivalent-workload peak-memory savings.
- Peak allocated GPU memory is essentially unchanged. In this reference path,
  prompt processing happens before turn-boundary compaction, so reduced final
  KV storage does not establish reduced prefill peak memory.
- Both compressed methods have lower decode throughput than full cache here.
  Query capture, buffer maintenance, scoring and gathers add work; further
  profiling would be required to attribute the slowdown precisely. Median
  measured eviction time is roughly 44–50 ms per episode, only one component.
- End-of-generation cache lengths can exceed the configured budget because
  decode compression is periodic. The budget holds immediately after compaction,
  and this export respects the budget-plus-interval growth bound.

## Next experiment

Keep this pilot unchanged. First increase the generation cap in a new pilot
configuration (for example, 512 instead of 192), retaining the same seed, inputs,
policies and budgets to diagnose truncation. Use a new output directory; do not
resume into or combine with this experiment. If capped reasoning persists, inspect
the response and task design before scaling up.

After settings are finalized, use a separate seed and larger fixed sample count
for evaluation. Report accuracy, truncation, actual token counts and resources
together, including negative results. A stronger claim about agentic workloads
will require a more varied task set or a public benchmark subset.

## Files and reproduction

- [summary.json](summary.json): original exported configuration-level metrics.
- [samples.jsonl](samples.jsonl): original per-episode outputs and diagnostics.
- [metadata.json](metadata.json): cloud provenance without the lengthy pip listing.
- [audit.json](audit.json): validation outcome and checksum of the result ZIP.

The untouched results ZIP under `artifacts/` retains the original full manifest,
including `pip freeze`. The original source ZIP is also preserved there.

```bash
python -m Performance_Metrics.audit_agentkv \
  --results artifacts/agentkv-results.zip \
  --source artifacts/agentkv-source.zip \
  --out Performance_Metrics/results/agentkv_pilot
```

Only the Python standard library is required for this audit. The raw files
contain generated synthetic records and model responses, not external user data.
