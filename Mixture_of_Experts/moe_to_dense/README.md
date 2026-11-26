# MoE-to-Dense conversion and distillation

This module implements a small, dependency-light version of the pipeline from **Pruning and Distilling Mixture-of-Experts into Dense Language Models**. It supports streaming calibration, SF/CP/ACP/DO-CP/DO-ACP expert selection, pure pruning, round-robin grouping, score-weighted merging, SwiGLU block concatenation, uniform/proportional down-projection scaling, and forward-KL distillation.

The local deliverable is a **methodological reproduction on a tiny synthetic language model**. It is not a reproduction of the paper's natural-language benchmarks. The local machine has a 4 GB RTX 3050 Laptop GPU, but the installed PyTorch build is CPU-only. Even the smallest investigated pretrained candidate has 3.8B total parameters; simultaneous teacher/student training is not feasible here.

## What is implemented

| Component | Status | Verification |
|---|---|---|
| Streaming calibration | Implemented | Accumulates O(E²), not per-token activations |
| SF, CP, ACP | Implemented | Controlled-score tests |
| DO-CP, DO-ACP | Implemented | Greedy log-det and diversity test |
| Pure pruning (`K = k`) | Implemented | Singleton groups preserve weights exactly |
| `K > k` merging | Implemented | Round-robin + weighted-average tests |
| Uniform/proportional scaling | Implemented | Numerical output-equivalence tests |
| Dense conversion | Implemented | No routed layer remains in tiny student |
| Forward-KL distillation | Implemented | Masking, freeze/update, checkpoint/resume tests |
| Pretrained backbone adapter | Interface only | Requires a larger CUDA environment and model-specific implementation |
| WikiText/FineWeb/downstream evaluation | Not run | Infeasible without a pretrained teacher and larger hardware |

## Design

The core algorithms do not know about a particular Hugging Face model. `model_adapter.py` defines the extraction/replacement boundary. Model-specific code maps an MoE layer to three canonical expert matrices:

- `gate`: `[expert_intermediate, hidden]`
- `up`: `[expert_intermediate, hidden]`
- `down`: `[hidden, expert_intermediate]`

`calibration.py` emits sufficient statistics. `scoring.py` selects experts. `merging.py` creates `k` group representatives. `concatenation.py` builds a standard dense SwiGLU FFN. `distillation.py` trains only the student.

The tiny adapter and model in `synthetic.py` exercise the same boundaries end to end and deliberately contain a real top-k router in the teacher. The converted student has no router.

## Install and test

Install PyTorch for the intended CPU/CUDA environment first, then:

```bash
python -m pip install pytest psutil
python -m pytest Mixture_of_Experts/moe_to_dense/tests -q
```

Expected local result: `31 passed`.

Peak memory is measured in a fresh CPU process using the OS peak working set (Windows) or peak RSS (Linux/macOS). It includes runtime startup, model loading, warmup and forward passes; it is not an isolated tensor-allocation measurement.

## Complete reproduction

`config.yaml` is JSON-compatible YAML and contains every experiment setting.

```bash
python -m Mixture_of_Experts.moe_to_dense.scripts.reproduce \
  --config Mixture_of_Experts/moe_to_dense/config.yaml
```

Outputs are written to `results/synthetic/`: calibration statistics, teacher/student state dictionaries, resumable checkpoints, the exact used config, `results.json`, and automatically generated `metrics.csv`.

The teacher dispatches only top-k expert-token assignments during inference; calibration intentionally evaluates all experts for the Gram matrix. Checkpoints are saved only after optimizer updates. Resume uses the same accumulation interval, and extending a run that ended with a partial accumulation window is rejected explicitly.

Generated `.pt` files are excluded from the submission and ignored by Git. Running the reproduction command regenerates them locally.

## Separate stages

The same pipeline can be executed stage-by-stage. Use one output directory for all four commands:

```bash
python -m Mixture_of_Experts.moe_to_dense.scripts.stage calibration \
  --config Mixture_of_Experts/moe_to_dense/config.yaml \
  --output-dir Mixture_of_Experts/moe_to_dense/results/staged

python -m Mixture_of_Experts.moe_to_dense.scripts.stage conversion \
  --config Mixture_of_Experts/moe_to_dense/config.yaml \
  --output-dir Mixture_of_Experts/moe_to_dense/results/staged --method do_acp

python -m Mixture_of_Experts.moe_to_dense.scripts.stage distillation \
  --config Mixture_of_Experts/moe_to_dense/config.yaml \
  --output-dir Mixture_of_Experts/moe_to_dense/results/staged --method do_acp

python -m Mixture_of_Experts.moe_to_dense.scripts.stage evaluation \
  --config Mixture_of_Experts/moe_to_dense/config.yaml \
  --output-dir Mixture_of_Experts/moe_to_dense/results/staged --method do_acp
```

These exact commands were smoke-tested locally.

## Adding a real backbone

Implement `MoEModelAdapter.iter_moe_layers` and `replace_with_dense` for the chosen architecture. Before claiming support, test all of the following:

1. router probability normalization and top-k conventions;
2. shared experts and dense-only layers;
3. projection orientation, biases, and gated activation;
4. exact dense config/checkpoint serialization, not only in-memory replacement;
5. tokenizer/vocabulary identity between teacher and student;
6. removal of every routed layer after conversion.

DeepSeek-V2-Lite needs special treatment: two shared experts are always active, top-6 routed weights are not renormalized, and its first FFN is dense. A generic pure-SwiGLU adapter is insufficient.

## Attribution

- Paper: Kim et al., *Pruning and Distilling Mixture-of-Experts into Dense Language Models*, arXiv:2605.28207 (accessed 2026-11-17).
- Reference implementation: `krafton-ai/moe-to-dense`, inspected at commit `02cedd4c5e3c4d79dfb78c295a42e2277ec0cda4`, Apache-2.0.
- DA4022 base: inspected at commit `368db370fa85b4997b44c063b9b697ba34fb38f5`.

This module was independently written for the DA4022 interfaces. No upstream source file was copied.
