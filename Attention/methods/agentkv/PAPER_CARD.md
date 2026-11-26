# Paper implementation card: AgentKV

- **Paper / OpenReview:** https://openreview.net/forum?id=s80e8otSxP
- **Implementation reference:** https://arxiv.org/html/2609.14872v1 (sections 3.3, 3.4 and appendices D/E).
- **Version caveat:** OpenReview served a verification challenge during inspection;
  equivalence of the workshop PDF and arXiv text has not been established.
- **Issue:** https://github.com/krsnoki/DA4022/issues/27
- **Primary module:** Attention
- **Supporting modules:** Backbones, Dataloaders, Performance_Metrics
- **Team:** Yash Pratap Singh, Khushal Nagwal, Pranjal Kirar, Babita Verma, Aditya Raj
- **Reviewer:** not assigned
- **Status:** partial-reproduction (CPU reference, pretrained numerical gate, and fixed T4 synthetic evaluation completed)
- **Upstream:** https://github.com/LiuTaowen-Tony/agentkv; Apache-2.0 for AgentKV;
  upstream bundled components have their own licenses. No upstream code is vendored.
- **Backbone:** Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca
- **Dataset:** synthetic.agent_memory.v1, seeded generated evaluation episodes
- **Calibration dataset:** null; no training or learned calibration
- **Hardware:** CPU float32 for correctness; CUDA cloud pilot planned in float16

## Research summary

Recent queries can underrepresent other recurring agent phases. AgentKV keeps
small query buffers for reasoning, actions, tool observations, and other context,
and scores candidate keys against their union. Our implementation retains the
highest-scoring keys within a configured budget after protecting sink/recent
entries. Query buffers are separate from the KV cache and have their own cost.

Expected behavior is a workload-dependent quality/memory trade-off. Smaller
caches alone do not establish a speedup: scoring and tensor gathers add overhead.

## Implementation

- **Input:** batch-one Qwen3 token IDs, aligned decoded fragments, per-layer
  queries/keys/values and persistent logical positions. No padding.
- **Output:** next-token logits, compacted cache, original retained positions,
  query-buffer state, cache/memory diagnostics.
- **Config:** `Attention/configs/agentkv/pilot.json`.
- **Tests:** `python -m pytest Attention/tests -q`.
- **Offline smoke:** `python -m Attention.experiments.agentkv.run --smoke --out artifacts/agentkv-smoke`.
- **GPU pilot:** `python -m Attention.experiments.agentkv.run --out artifacts/agentkv-pilot`.
- **Baselines:** unmodified full cache; matched recent-query eviction using the
  same representative capacity, score, protections and compression schedule.

## Results

| Source | Quality | Latency | Throughput | Peak GPU memory | Energy | Notes |
|---|---|---|---|---|---|---|
| Local CPU verification | not evaluated | not a pretrained benchmark | not evaluated | not measured | not measured | 25 tests passed; random-model end-to-end smoke passed |
| Pretrained full-cache baseline | 4/5 exact match | 10.03 s median episode | 24.50 decode tokens/s | 1373.79 MB allocated | not measured | Five synthetic episodes; one final turn truncated |
| Pretrained recent, B=512 | 4/5 exact match | 12.55 s median episode | 20.66 decode tokens/s | 1373.71 MB allocated | not measured | Five synthetic episodes |
| Pretrained AgentKV, B=512 | 4/5 exact match | 11.95 s median episode | 20.73 decode tokens/s | 1373.71 MB allocated | not measured | Five synthetic episodes; one final turn truncated |
| Reviewed evaluation, full cache | 20/30 exact match | 8.92 s median episode | 26.35 decode tokens/s | 1421.75 MB allocated | not measured | Three fixed synthetic length strata; no truncation |
| Reviewed evaluation, recent B=512 | 17/30 exact match | 11.34 s median episode | 21.26 decode tokens/s | 1422.50 MB allocated | not measured | Same task/strata; no truncation |
| Reviewed evaluation, AgentKV B=512 | 15/30 exact match | 11.37 s median episode | 21.28 decode tokens/s | 1422.50 MB allocated | not measured | Same task/strata; no truncation |

No paper performance number is claimed as reproduced. The random model's
outputs are not task-quality measurements. See the [complete pilot report and
raw results](../../../Performance_Metrics/results/agentkv_pilot/README.md) for all
budgets, provenance, truncation diagnostics and limitations. This pilot shows
no aggregate accuracy advantage over recent-query eviction and no throughput
improvement over full cache.

A [second pilot with a 512-token output cap](../../../Performance_Metrics/results/agentkv_pilot_512/README.md)
completed all 35 episodes with no truncated turns. Full cache remains 4/5 under
strict scoring because one response lacks a final-answer delimiter. At B=1024,
recent-query eviction reaches 5/5 and AgentKV remains 4/5, also with a delimiter
failure. At B=512 both remain 4/5 with incorrect final-code failures. The two
runs remain separate; these five examples do not establish general superiority.

## Differences and limitations

The [subsequent fidelity review](FIDELITY_REVIEW.md) pins the upstream engine
commit and distinguishes original `reference_v1` results from the new
`upstream_mean` scoring option. The [reviewed pilot](../../../Performance_Metrics/results/agentkv_reviewed_pilot/README.md)
passed the pretrained numerical gate and completed 35 episodes without truncation.
AgentKV scores 1/5, 5/5, 5/5 at budgets 256/512/1024, versus recent's 3/5, 4/5, 4/5.
It changes GQA reduction, candidate normalization and protection settings.
Do not attribute differences between profiles solely to phase awareness.

The [fixed reviewed evaluation](../../../Performance_Metrics/results/agentkv_evaluation/README.md)
contains 210 measured episodes across three context-length strata. It finds
recent-query eviction above AgentKV at all three budgets (9/30 vs. 2/30 at 256,
17/30 vs. 15/30 at 512, and 22/30 vs. 20/30 at 1024); full cache is faster. This
is still a narrow synthetic retrieval task and does not establish broad method
rankings.

- Smaller model and synthetic scripted tool observations instead of full benchmarks.
- Live answers, but tool calls/observations are supplied fixtures, not an autonomous agent.
- Plain tensor compaction, not the authors' paged allocator or optimized serving engine.
- Original pilots used GQA max reduction and no score pooling. The new reviewed
  option uses the released preset's mean reduction over unprotected candidates;
  parser and serving paths still differ from upstream.
- Closing tags reset phase to others; split markers apply on completion.
- Additional turn-boundary compression; periodic decode interval remains configurable.
- Greedy generation and one seed in the pilot; no confidence or superiority claims.
- Cache/query/protection budgets are explicit; peak memory can exceed retained-cache budget.
- A no-eviction adapter equivalence check and multi-turn positional tests pass on
  tiny random Qwen3. Pretrained CUDA execution completed in the initial pilot;
  the reviewed pilot also passed the tested pretrained no-eviction numerical
  comparisons (maximum tested logit difference 0.0). This is not universal equivalence.

## Checklist

- [x] Issue opened before coding
- [x] Reference paper version and upstream attribution recorded
- [x] Model/tokenizer revision pinned
- [x] Deterministic synthetic loader and seed recorded
- [x] CPU tests and offline smoke added
- [x] Pretrained baseline runs on cloud hardware (initial five-example pilot)
- [x] Resource and task-quality results recorded (pilot only)
- [x] Pretrained numerical integration gate passed on cloud hardware
- [x] Fixed 30-example, three-stratum cloud evaluation completed and audited
