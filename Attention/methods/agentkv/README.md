# AgentKV reference implementation

Contribution for [issue #27](https://github.com/krsnoki/DA4022/issues/27).
Implements phase-aware representative-query eviction, a matched recent-query
baseline, and full-cache generation on Qwen3. CPU correctness tests and a
download-free random-model smoke experiment are available. A [five-example
Tesla T4 pilot](../../../Performance_Metrics/results/agentkv_pilot/README.md)
completed all 35 measured episodes. Both compressed policies tie in aggregate
accuracy; neither improves throughput over full cache in this small run.
The [512-token cap rerun](../../../Performance_Metrics/results/agentkv_pilot_512/README.md)
eliminates truncation but still shows no AgentKV advantage. It separates missing
final-answer delimiters from incorrect recalled codes without changing scoring.

## Reviewed implementation and next run

The [pinned upstream comparison](FIDELITY_REVIEW.md) identified score-reduction
and candidate-normalization differences in the original reference implementation.
`scoring_profile=reference_v1` preserves that behavior for old configs.
`scoring_profile=upstream_mean` matches the released preset's mean-GQA score over
unprotected candidates. `reviewed_pilot.json` also uses its sink16/recent8 settings.
This is a new experiment; do not merge its results with the existing pilots.
Parser and serving differences remain documented; it is not a full upstream port.

New diagnostics explain strict-score failures and record whether the first live
response repeats the expected code. All policies retain the same primary metric.

```bash
# Download-free CPU check of the integration gate:
python -m Attention.experiments.agentkv.verify_model --smoke

# Cloud: fail on numerical integration errors before running the new pilot:
python -m Attention.experiments.agentkv.verify_model --out artifacts/reviewed-verification.json
python -m Attention.experiments.agentkv.run --config Attention/configs/agentkv/reviewed_pilot.json --out artifacts/agentkv-reviewed-pilot
```

The self-contained next notebook is built from a **new** source archive:

```bash
python -m Attention.experiments.agentkv.package_source --out artifacts/agentkv-reviewed-source.zip
python -m Attention.experiments.agentkv.build_review_notebook
```

Open `artifacts/agentkv_reviewed_pilot.ipynb` in Colab, select T4 GPU and Run all.
It embeds source and downloads `agentkv-reviewed-results.zip`. It runs tests,
the pretrained gate, and only the revised five-example pilot. Source packaging
refuses to overwrite an existing archive; use versioned names for later revisions.

After reviewing that pilot, the frozen 30-example length-stratified evaluation
can be run using `run_suite --verification artifacts/reviewed-verification.json`.
It uses seeds 101/202/303 with 24/48/72 distractors, ten examples per stratum,
seven configurations each (210 measured episodes). It remains one synthetic
retrieval task family; it is not an official agent benchmark. Keep the three
strata separate and do not interpret data-seed variation as repeated timing trials.

## What runs where

| Work | Machine | Downloads |
|---|---|---|
| Edit, Git, read results | Local laptop | None |
| Unit tests and random-model smoke | Local CPU or cloud CPU | Python packages only |
| Qwen3-0.6B pilot | Cloud CUDA GPU | Pinned model/tokenizer |

No training is required. The normal benchmark refuses to download weights when
CUDA is unavailable. Target a 16 GB cloud GPU initially; memory fit must be
verified in the pilot. Free cloud availability is not guaranteed.

## CPU setup and verification

Run commands from the repository root, not its parent folder. Tested locally
with Python 3.13.1, torch 2.8.0+cpu, transformers 4.57.1 and float32 on Windows.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r Attention/requirements-agentkv.txt
.venv/Scripts/python.exe -m pytest Attention/tests -q
.venv/Scripts/python.exe -m Attention.experiments.agentkv.run --smoke --out artifacts/agentkv-smoke
```

The smoke command uses a tiny random Qwen3 and a byte tokenizer, executes all
three policies across two turns, and writes `smoke_only` results. Its generated
text, accuracy, and timings are not evidence about a pretrained model. CPU peak
memory and energy are left null. Packages, including CPU PyTorch, can require
hundreds of MB; local installation is optional if using cloud CPU tests instead.

## Cloud pilot

Use [the notebook](../../experiments/agentkv/cloud_pilot.ipynb), or execute:

To prepare the upload ZIP locally, run:

```powershell
.venv/Scripts/python.exe -m Attention.experiments.agentkv.package_source
```

Upload `artifacts/agentkv-source.zip` to the cloud runtime's file browser. It
contains source, tests, docs and config, plus a source snapshot identifier;
it excludes weights, environments, artifacts and Git metadata. Open the notebook
in Colab, select a GPU runtime, and run its cells. This does not require pushing
the working tree to GitHub first. Both the cloud installation and pretrained
weights use cloud disk/network, not local model execution.

Alternatively, from an existing cloud checkout:

```bash
# In a cloud CUDA runtime, from a checkout of this contribution:
python -m pip install -r Attention/requirements-agentkv.txt
python -m pytest Attention/tests -q
python -m Attention.experiments.agentkv.run \
  --config Attention/configs/agentkv/pilot.json \
  --out artifacts/agentkv-pilot
```

Keep the cloud runtime's working CUDA-enabled PyTorch installation; do not use
the CPU wheel command there. torch 2.8.0 is the locally tested reference version;
the first cloud pilot ran with torch 2.14.1+cu130 on a Tesla T4. The runner records
actual versions and `pip freeze` instead of assuming the runtime matches local.

The config fixes Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, seed 42, five synthetic examples,
three cache budgets (256/512/1024), batch one, and greedy generation capped at
192 new tokens per turn. It runs one full-cache baseline, then recent-query and
AgentKV at each budget. One separate seeded warm-up episode runs per policy
configuration. Tokenization/model loading are excluded from timed intervals.

Before interpreting a larger run, inspect the pilot: does full cache solve the
task, do reasoning spans finish, and do contexts actually exceed budgets?
If baseline accuracy is poor or generations are capped, revise the task/model
or output cap and record that as a separate experiment. Do not tune on a final
evaluation set. Use a new seed and larger fixed example count for evaluation
after selecting settings on pilot data. This is not a BFCL/tau2-bench reproduction.

## Output and restart

- `manifest.json`: config, source hash, commit when available, software and hardware.
- `samples.jsonl`: answers, correctness, token counts, timing, cache diagnostics,
  and failures; flushed after every example.
- `summary.json`: one row per configuration using the shared metrics fields.

```bash
python -m Attention.experiments.agentkv.run \
  --out artifacts/agentkv-pilot --resume
```

Resume requires matching config, source files, major recorded software versions,
and device name. It skips existing rows, including failures; a fresh directory
is required to retry failures. A changed cloud GPU requires a new run. Hardware
name matching cannot guarantee identical clock/load conditions across sessions.
Keep raw rows and distinguish resumed measurements when reporting results.

Latency includes two input-prefill phases and two live-generation phases;
prefill includes explicit turn-boundary compaction. Decode includes periodic
eviction and caching the last generated token, needed for persistent turns.
Tokens/s uses generated tokens divided by decode seconds. GPU timing is
synchronized. Eviction seconds are a subset of these times, not an additional
time to add. Query capture overhead is included in method latency.

GPU peak allocated memory includes model weights, query buffers, cache, and
temporary tensors, reset per episode. `kv_bytes` describes the final retained
cache, not process peak memory. Per-turn snapshots expose actual representative
counts; early in a trajectory some phase buffers can be underfilled. p95 is
null for fewer than 20 successful examples. Energy is unmeasured.

## Algorithm and supported contract

- Qwen3, batch one, unpadded inputs, default RoPE, full attention layers, eval mode.
- `append(ids, fragments)` takes **new** input tokens, never the full old transcript.
- One decoded fragment per input token; a phase change applies when a marker
  completes. Prefill chunks and generated tokens follow the same parser.
- Per-layer queries have shape `[query_heads, representatives, head_dim]` and
  keys `[kv_heads, cached_tokens, head_dim]`. Buffers store detached post-RoPE queries.
- Float32 softmax over candidate keys, then average across representatives.
  In reference_v1, GQA query-head scores reduce by max within a KV group;
  upstream_mean averages query heads and excludes protected entries first. Optional odd-kernel max
  pooling defaults to 1 (disabled). Both compressed policies use identical rules.
- Protected original sink positions and recent retained entries count toward
  the budget. Remaining entries are selected per KV head; ties prefer earlier
  retained positions. Keep sets are sorted chronologically before gathering.
- No historical causal mask is used for representative scoring: this estimates
  future attention over the current candidate pool. Actual model decoding is causal.
- Physical cache length determines causal masking; logical positions determine
  RoPE. Different heads can retain different positions while sharing cache length.
- Decode compression runs every configured number of fed generated tokens;
  the experiment additionally compresses after each supplied prompt turn.
  This is a documented reference adaptation of the paper's schedule.
- Cache can exceed the target budget between compression events and during
  prefill. The budget is a post-compression target, not a strict peak bound.

`Qwen3Session` is a context manager: hooks are removed on exit, including errors.
It is not compatible with concurrent requests, beam search, training, sliding
windows, dynamic RoPE, compilation or CUDA graphs. The adapter requires
transformers 4.57.1 because it accesses that version's dynamic cache layers.

## Limits and attribution

This is an independent implementation of the paper's reference algorithm, with
documented GQA reduction, parser closing-tag behavior, and turn-boundary
compression choices. It does not import or vendor the authors' serving engine.
The upstream release's CUDA/Triton optimizations, paged allocator, batching and
concurrency are not reproduced. No throughput improvement is promised.

See [the paper card](PAPER_CARD.md), [the paper](https://arxiv.org/html/2609.14872v1),
and [upstream code](https://github.com/LiuTaowen-Tony/agentkv).
