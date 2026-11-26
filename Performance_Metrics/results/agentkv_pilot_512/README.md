# AgentKV pilot rerun: 512-token output cap

The rerun completed all 35 measured episodes without recorded runtime failures.
No generated turn reached the output cap (0/70 turns). The export audit passed:
source hash matches the original uploaded bundle, sample coverage is complete,
reported correctness and key summary metrics recompute, and cache bounds hold.

## Controlled comparison

Compared with the [first pilot](../agentkv_pilot/README.md), the only recorded
experiment-config change is `max_new_tokens: 192 -> 512`. Model/tokenizer revision,
source hash, seed, five examples, budgets, precision and policy settings match.
Both manifests identify a Tesla T4, Python 3.13.16, torch 2.14.1+cu130,
transformers 4.57.1 and CUDA 13.0. The complete pip listings also match.
This does not establish identical physical GPU clocks/load between sessions.

| Method | Cache budget | Correct, cap 192 | Correct, cap 512 | Decode tokens/s, cap 512 | Median episode seconds | Max final KV MB | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full cache | unlimited | 4/5 | 4/5 | 25.46 | 9.43 | 159.88 | 1374.86 |
| Recent | 256 | 0/5 | 0/5 | 20.60 | 12.73 | 43.58 | 1373.71 |
| AgentKV | 256 | 0/5 | 0/5 | 20.53 | 13.11 | 42.55 | 1373.71 |
| Recent | 512 | 4/5 | 4/5 | 20.64 | 12.76 | 73.17 | 1373.71 |
| AgentKV | 512 | 4/5 | 4/5 | 20.52 | 12.20 | 72.25 | 1373.71 |
| Recent | 1024 | 4/5 | 5/5 | 20.63 | 12.25 | 130.97 | 1373.71 |
| AgentKV | 1024 | 4/5 | 4/5 | 20.68 | 12.13 | 130.97 | 1373.71 |

MB is decimal. Episode latency covers both turns. Final KV storage excludes
query buffers (3.67 MB per compressed policy) and position metadata; it is not
peak GPU memory. Actual output lengths differ, so timing is not a fixed-work
comparison. These small fixed-order runs cannot establish statistical superiority.

## What the longer generation reveals

- **Full cache, sample 1:** the answer mentions the expected `197251` but ends
  without `</think>` or a separate final answer. The first run cut off this same
  trace. Removing the cap therefore resolves truncation but not format compliance.
- **AgentKV, budget 1024, sample 1:** similarly mentions `197251` and ends without
  the final-answer delimiter. It remains incorrect under the unchanged evaluator.
- **Recent, budget 1024:** now completes all five final answers correctly.
- **Recent, budget 512, sample 4:** emits final code `898753` instead of `898975`.
- **AgentKV, budget 512, sample 3:** now completes its reasoning but emits final
  code `719824` instead of `712982`. The earlier truncation concealed an incorrect
  completed answer; it was not merely a harmless token-limit failure.
- **Budget 256:** both policies still fail every example without any truncation.

The declared metric requires a completed reasoning delimiter followed by exactly
the expected digits (apart from whitespace/end tokens). We have not changed that
rule or counted an answer found inside reasoning as correct. Missing-delimiter
cases should be distinguished from incorrect final-code cases in later reports.

## Conclusions and next workflow

The increased cap removes the observed truncation confound. It does **not** show
an AgentKV accuracy or speed advantage. At 1024, recent-query eviction has one
more strictly correct answer than AgentKV in this five-example pilot; this is
not enough evidence to generalize. Both compressed methods remain slower than
full cache, and peak allocated memory stays essentially unchanged.

Keep both pilot exports unchanged. Before a larger run, finalize the protocol:
retain 512 as the candidate output cap, keep the strict primary metric, and add
separate diagnostic counts for truncation, absent final delimiter and incorrect
final answers. Diagnostic counts must not replace the primary score. A fresh,
fixed evaluation seed and more varied instances are needed; simply repeatedly
rerunning the same five examples will not demonstrate a general benefit.

The baseline's format failures should be reported explicitly. If the prompting
or answer protocol is changed, validate it as a new pilot and label the changed
protocol rather than merging its results into these runs. A public benchmark
subset remains a possible extension, not a completed component.

## Reproduction and provenance

- Config: `Attention/configs/agentkv/pilot_512.json`.
- [summary.json](summary.json): original exported metrics.
- [samples.jsonl](samples.jsonl): original per-episode outputs.
- [metadata.json](metadata.json): recorded environment/config/source identity.
- [audit.json](audit.json): integrity checks and ZIP checksum.
- Original source SHA-256: `f73578c1db67e9e4acfbd853bd77c572aa02e9c03d5ec7ca7f16f1027c2317c3`.
- The recorded Git commit is the base commit, not a committed implementation.

```bash
python -m Attention.experiments.agentkv.run \
  --config Attention/configs/agentkv/pilot_512.json \
  --out artifacts/agentkv-pilot-512

python -m Performance_Metrics.audit_agentkv \
  --results artifacts/agentkv-results-512.zip \
  --source artifacts/agentkv-source.zip \
  --out Performance_Metrics/results/agentkv_pilot_512
```

The original embedded-source notebook and both result ZIPs remain in the ignored
`artifacts/` directory. No model inference was rerun locally during this audit.
