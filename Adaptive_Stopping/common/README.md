# `Adaptive_Stopping/common`: step-level stopping framework

Shared, method-agnostic utilities for policies that decide **after each completed reasoning step** whether a reasoning model should stop. They contain nothing specific to one method. The Step-Tagging Early-Stopping implementation (`methods/step_tagging/`) and the baselines in `baselines/` are built on them.

The framework does not import any backbone, dataset loader or metric. It works on token IDs, text and any Hugging Face causal LM passed in by the caller.

## Contract

| Module | What it provides |
|---|---|
| `policy.py` | `StepState` (step index, step text, generated-token count, `in_thinking`), the `StoppingPolicy` interface (`reset`, `should_stop(state, metrics, budget)`, `report`), and `budget_exceeded` for shared `max_steps` / `max_tokens` caps |
| `steps.py` | `"\n\n"` step segmentation, offline (`segment_token_ids`, `segment_text_pieces`) and online (`StepSegmenter`, `IncrementalDetokenizer`). For any chunking of the text it gives exactly `str.split("\n\n")`. A step is *closed* by the token that completes its delimiter; steps after `</think>` are marked as outside the thinking phase. |
| `replay.py` | `replay_policy`: runs a policy over the steps of a cached trace and returns where it stops. Only closed steps inside `<think>` can trigger a stop. Fast paths for threshold sweeps (`st_es_cut_points`, `step_count_cut_points`) are checked against `replay_policy` in the tests. |
| `cached_generation.py` | `generate_full_traces`: batched sampling of complete traces for offline replay. It uses per-row prefill into a static KV cache, continuation rounds of a fixed number of tokens, shortest-context-first batching, and a KV-token budget that fits small GPUs. |
| `live_generation.py` | `generate_with_policy`: online generation at batch size 1 with a policy as a Hugging Face stopping criterion. On a stop, `continue_with_exit_prompt` appends the exit prompt and generates the answer while reusing the KV cache. Latency, time to first token and policy time are measured. |
| `forced_answer.py` | Exit prompts (`EXIT_PROMPTS`), the 100-token answer budget, and stop-when-the-box-closes (`BoxCloseCriteria`). `force_answers` works on a batch of truncated contexts; `force_answers_shared_prefix` does every cut point of one trace with shared-prefix KV reuse and chunked prefill. |

The baselines in this layer (`baselines/no_stop.py`, `baselines/step_count.py`) implement the contract: `NoStopping` never stops (full generation), and `StepCountStopping(k)` stops after k closed reasoning steps (the paper's count baseline).

## Use

```python
from Adaptive_Stopping.baselines.step_count import StepCountStopping
from Adaptive_Stopping.common.replay import replay_policy
from Adaptive_Stopping.common.steps import segment_token_ids

steps = segment_token_ids(gen_ids, tokenizer)  # cached generated token IDs of one trace
outcome = replay_policy(StepCountStopping(10), steps, len(gen_ids))
outcome.stopped, outcome.cut_tokens  # whether and where (in generated tokens) the policy stops
```

A new step-level method only implements `StoppingPolicy.should_stop`. Replay, live generation and answer forcing work unchanged.

## Configuration and dependencies

- **Python packages:** `Adaptive_Stopping/requirements-st_es.txt`, after installing PyTorch (tested: torch 2.11.0+cu128, transformers 5.24.1, Python 3.12). The `transformers` 5.x API is required (`generate(cache=...)`, `StaticCache`, `DynamicCache.layers`).
- **Generated data:** caches and run outputs are written outside the repository (`.gitignore` covers them).
- **Tuning knobs:** the GPU memory knobs (KV-token budget, chunk size, prefill chunk) are function arguments. The defaults fit an 8 GB GPU.

## Tests

```bash
python -m pytest Adaptive_Stopping/tests/test_steps.py Adaptive_Stopping/tests/test_forced_answer.py -q
```

- **This layer's tests** (CPU only) check:
  - online segmentation against `str.split` under random chunking;
  - thinking flags and step end tokens around `</think>`;
  - delimiters split across tokens, and empty generations;
  - multi-byte characters in incremental detokenisation;
  - brace counting, the per-row box-close stop and batch budgets for answer forcing.
- **Tests that also use the ST-ES policy** live with the method: replay equivalence, policy edge cases, and the tiny-random-model check that cached generation, shared-prefix forced answers and KV-reuse live answers equal their reference implementations.
- **Tokenizer test.** One test uses the real backbone tokenizer and is skipped unless it is in the local Hugging Face cache.

## Limitations

- **Batch size.** Live generation runs at batch size 1; batched live stopping is not implemented.
- **Batching and randomness.** Cached traces are sampled in batched continuation rounds. The sampling distribution is that of standard generation, but the exact token stream depends on the batch schedule.
- **Windows.** GPUs exceeding dedicated memory spill into shared memory instead of raising an out-of-memory error. The callers cap the allocator and use static KV caches.
