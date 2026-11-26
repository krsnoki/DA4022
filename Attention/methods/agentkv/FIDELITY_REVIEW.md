# Implementation fidelity review

Reviewed upstream engine commit:
`254c57bc84e4a7895159ba1062bd46c83626f511` (the engine submodule referenced by
the authors' AgentKV repository at inspection). This review distinguishes the
released preset from the arXiv v1 description; neither is silently substituted
for the other. No upstream source was copied into this repository.

## Sources inspected

- [Preset factory](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/eviction/__init__.py)
- [Stage-Q buffers/selection](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/eviction/stageq.py)
- [SnapKV scoring and protected candidates](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/eviction/snapkv.py)
- [Query ring](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/eviction/query_ring.py)
- [Prefill phase labeling](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/tokenizer/tokenize.py)
- [Semantic phase mappings](https://github.com/LiuTaowen-Tony/kv-management-minisgl/blob/254c57bc84e4a7895159ba1062bd46c83626f511/python/minisgl/phase.py)
- [Paper reference](https://arxiv.org/html/2609.14872v1), section 3.3, appendices D/E.

## Findings and decisions

| Component | Existing pilots (`reference_v1`) | Reviewed option (`upstream_mean`) |
|---|---|---|
| Query representatives | Four FIFO buffers, total 32 | Unchanged; agrees with released preset capacity |
| Grouped-query reduction | Max of mean attention probabilities across grouped query heads | Mean over both query heads and representatives, matching released AgentKV preset |
| Scoring candidate pool | All cached keys, including protected entries | Protected entries excluded before softmax, matching released SnapKV path |
| Sink/recent protection | 32/32 in pilot configs | 16/8 in new configs, matching released preset factory and inherited recent-window default |
| Token score pooling | Disabled, kernel 1 | Unchanged; matches released AgentKV preset |
| Ties | Stable preference for earlier positions | Same deterministic tie handling; upstream topk need not choose identical tied indices |
| Score precision | Float32 score/softmax | Same; upstream uses its input dtype for logits, so low-precision numerical parity is not claimed |

The old GQA maximum was a reference choice, not the released preset's mean.
It can change rankings. Excluding protected entries also changes normalization
and can change rankings after averaging queries. Both differences are now
explicit, selectable and tested. Existing pilot configs default to
`reference_v1`; raw results and original source bundles remain unchanged.

Do not compare the revised profile to the earlier pilots as though only phase
awareness changed: score reduction, candidate normalization and protection
settings also differ. Within each new experiment, recent and AgentKV use the
same reviewed score/settings and differ only in representative selection.

## Deliberately unreproduced behavior

- **Parser:** the released tokenizer uses character spans and offset-majority
  labels for prefill. Its legacy marker table maps assistant openings to stage 1
  and `</think>` to stage 2. Our streaming completion-based parser retains the
  explicitly documented four semantic phases and returns closing tags to others.
  Therefore `upstream_mean` means scoring compatibility, not full parser or
  end-to-end equivalence to the upstream release. Decode parser parity has not
  been established.
- **Compression timing:** our explicit turn-boundary compression plus periodic
  decode schedule remains an adaptation, not a verified serving-scheduler port.
- **Serving lifecycle:** we physically gather tensors, but do not implement the
  paged allocator, fused kernels, CUDA graphs, multi-request concurrency or
  chunked-prefill optimizations of the upstream engine.
- **Model/task:** Qwen3-0.6B and scripted retrieval episodes remain a reduced
  experiment. More examples or context lengths do not create a general agent benchmark.

## Correctness evidence

CPU tests compare the mean-GQA/candidate-only score against a separately written
loop reference and verify protected key contents cannot affect candidate scoring.
A regression test preserves `reference_v1` selection behavior. Existing tests
cover phase parsing, FIFO state, compaction and positional handling.

`verify_model.py` adds a cloud gate comparing per-step no-eviction logits and final
K/V tensors against unmodified Hugging Face Qwen3 for the same forced inputs.
It checks next-token argmax, uses declared tolerances, and then checks finite
continuation after forced compression plus phase-buffer activity. Its random-model
CPU mode and reviewed pretrained Tesla T4 execution passed; the tested
no-eviction maximum logit difference was 0.0.

## Evaluation protocol

The reviewed notebook ran the pretrained gate and then the reviewed profile's
five-example pilot at the 512-token output cap. This separated the scoring-profile
change from the fixed larger evaluation.

Prepared evaluation: 10 examples each at seeds 101/202/303 with 24/48/72
distractors, respectively; seven method/budget configurations, 210 episodes.
The completed export passed audit and is reported by stratum in
`Performance_Metrics/results/agentkv_evaluation/`. The seeds generate different
data, not multiple stochastic decoding runs. This varies context length within
one synthetic task family. Do not describe it as a diverse agent benchmark or
isolate seed effects from length effects. No settings were selected using those
evaluation results.

Primary exact match is unchanged. Extra diagnostics count truncation, missing
final delimiters and incorrect final answers, and record whether the first live
response repeated the expected code. Repetition is a possible alternative source
of retained information, not evidence that the original tool KV survived.
