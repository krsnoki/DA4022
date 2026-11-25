# Synthetic agent memory v1

- **Dataset ID:** synthetic.agent_memory.v1
- **Source/version:** `Dataloaders/synthetic/agent_memory.py`, v1; implementation commit/source hash recorded per run
- **Role/split:** synthetic evaluation, not a public benchmark split
- **Task:** retain a six-digit collection code across scripted tool interactions
- **Samples:** generated on demand; pilot has five, default generator twenty
- **Seed:** configurable; fixed sample IDs include generator version, seed and index
- **Fields:** sample_id, first_prompt, continuation, expected
- **Preprocessing:** explicit Qwen-style chat/phase markers, tokenizer from pinned model revision
- **Batching:** one unpadded episode; prompt chunks configured by experiment
- **Download/cache:** none; no third-party records, tools or APIs
- **License:** generated fixtures contain no copied external dataset; repository has no declared top-level license

First, a scripted order lookup returns a code and the model is asked to remember
it. The same model/cache continues through a second scripted tool observation
with distractor records, then answers the code question. Assistant answers are
generated live; tool calls and tool results are fixtures. This is a controlled
memory diagnostic, not a test of tool selection or an official agent benchmark.

The expected code is not inserted into the continuation prompt. Warm-up uses a
different seed. Pilot data must not become a held-out test set after tuning;
choose and record a separate evaluation seed/config once the pilot is complete.
Code collisions among randomly generated distractors are possible and should be
considered when expanding the generator; v1 does not model realistic records.

Exact match accepts only the six digits after a completed `</think>` marker,
ignoring end-of-message tokens and whitespace. A code occurring only in reasoning
does not count. Capped/unfinished reasoning is incorrect, and truncation is logged.

Smoke test: `python -m pytest Attention/tests/test_agentkv_experiment.py -q`.
