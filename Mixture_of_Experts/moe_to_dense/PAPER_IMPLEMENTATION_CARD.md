# Paper implementation card: Pruning and Distilling Mixture-of-Experts into Dense Language Models

- **Paper / OpenReview:** https://openreview.net/forum?id=ZqRh99OQEK
- **Primary module:** `Mixture_of_Experts/moe_to_dense`
- **Secondary modules:** `Performance_Metrics`
- **Owner / reviewer:** pending
- **Status:** partial-reproduction
- **Tracking issue:** [DA4022 #31](https://github.com/krsnoki/DA4022/issues/31)
- **Upstream code / licence:** https://github.com/krafton-ai/moe-to-dense, Apache-2.0, commit `02cedd4c5e3c4d79dfb78c295a42e2277ec0cda4`
- **Backbone IDs:** local `TinyMoELM`; pretrained adapters pending
- **Dataset IDs and split:** local deterministic synthetic train/calibration/evaluation splits
- **Hardware assumptions:** CPU baseline; real DeepSeek validation requires substantially larger CUDA hardware

## Research summary

### Problem

MoE models have sparse per-token compute but all expert weights still consume storage and memory. The method converts routed expert blocks into conventional dense FFNs.

### Method

Collect routing and expert-output statistics, select important/diverse experts, optionally merge selected experts into groups, concatenate group matrices into a dense SwiGLU block with static down-projection scaling, and recover behavior using forward-KL distillation from the frozen MoE teacher.

### Expected behaviour

The paper reports strongest performance from diversity-aware pure pruning followed by distillation. Locally, the implementation is expected only to reduce parameters, remove routing, and reduce teacher/student KL on a controlled synthetic task.

## Implementation

- **Input contract:** router probabilities `[T,E]`, expert outputs `[T,E,H]`, selected indices `[T,k]`; expert matrices in canonical SwiGLU orientation
- **Output contract:** dense `DenseSwiGLU`, selected IDs/groups/scores, checkpoints and JSON metrics
- **Configuration:** `Mixture_of_Experts/moe_to_dense/config.yaml`
- **Reproduction command:** `python -m Mixture_of_Experts.moe_to_dense.scripts.reproduce --config Mixture_of_Experts/moe_to_dense/config.yaml`
- **Tests:** `python -m pytest Mixture_of_Experts/moe_to_dense/tests -q`

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | not copied | not copied | not copied | not copied | not measured | See paper for Qwen3 results |
| Local baseline | PPL 6.167 | 1.146 ms | 167,466 tok/s | 187.77 MiB | not measured | Tiny top-k routed teacher, CPU; unoptimized dispatch |
| Local method | PPL 7.378 | 0.161 ms | 1,191,437 tok/s | 186.97 MiB | not measured | DO-ACP dense student after KD; synthetic |

## Differences and limitations

- Synthetic model/data; no language benchmark.
- One MoE layer rather than a multi-billion-parameter transformer.
- No architecture-specific pretrained adapter yet.
- Peak CPU process working set includes startup and model loading; energy was not measured.
- One seed and a tiny synthetic expert set cannot establish method rankings.

## Reproduction checklist

- [x] Exact paper link and upstream commit recorded
- [x] Local model/data configuration pinned
- [x] Baseline runs on the same setup
- [x] Tests and seed handling added
- [x] Resource and quality metrics reported where measured
- [x] Licence/data terms checked
