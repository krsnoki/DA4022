# Hardware assessment and model selection

Assessment date: 2026-11-17 (Asia/Calcutta).

## Available environment

| Item | Observed value |
|---|---|
| OS | Windows 11, build 10.0.26300 |
| CPU | 16 logical processors; exact model unavailable to the process |
| RAM | 15.3 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4,096 MiB |
| Driver / CUDA driver capability | NVIDIA 610.91 / CUDA 13.3 |
| PyTorch | 2.8.0+cpu |
| CUDA available to PyTorch | No |
| Free disk before experiments | 94.1 GiB |

The discrete GPU exists but cannot be used by the installed CPU-only PyTorch build. Installing a CUDA build would not solve the capacity limit: 4 GB VRAM is far below the weight memory of every credible pretrained candidate.

## Candidate comparison

| Candidate | Total / active | Experts / top-k | Approx. BF16 weights | Architecture notes | Local decision |
|---|---:|---:|---:|---|---|
| Qwen1.5-MoE-A2.7B | 14.3B / 2.7B | 60 / 4 + shared expert | 28.6 GB | SwiGLU, Transformers support; shared expert handling needed | Infeasible |
| DeepSeek-V2-Lite | 15.7B / 2.4B | 64 routed + 2 shared / 6 | 31.4 GB | Paper/upstream-supported; special non-renormalized routing and dense first layer | Infeasible; best fidelity on larger hardware |
| Phi-tiny-MoE-instruct | 3.8B / 1.1B | 16 / 2 | 7.6 GB | SwiGLU-like custom PhiMoE code, MIT model license | Smallest candidate, still infeasible for simultaneous KD |
| Tiny synthetic MoE | 16,768 parameters locally | 8 / 2 | 65.5 KiB | Controlled routed SwiGLU model | Selected for executable validation |

BF16 estimates include weights only (`2 bytes × parameters`). They exclude activations, the dense student, gradients, Adam states, temporary logits, framework overhead, and checkpoints. Phi-tiny therefore cannot fit a full teacher/student training experiment into 4 GB even though quantized inference might be possible.

## Decision

The selected local backbone is the synthetic `TinyMoELM`. It validates algorithms, tensor layout, router removal, serialization, forward KL, and efficiency measurement without pretending to establish language-model quality.

For a future real experiment, use **DeepSeek-V2-Lite only on a substantially larger multi-GPU or ≥40 GB inference environment**, because it is supported by the paper and upstream code. If the goal is the smallest engineering trial rather than closest paper fidelity, evaluate Phi-tiny-MoE first, but write and validate a PhiMoE-specific adapter before downloading weights.

## Memory gate for a real run

Before any download, budget separately for:

1. complete teacher weights, not active parameters;
2. calibration activations and the E×E Gram accumulator;
3. dense student weights and conversion workspace;
4. student gradients plus optimizer states;
5. simultaneous teacher/student logits during KL training;
6. checkpoints and evaluation caches.

Quantization or CPU offload may enable a smoke test but changes latency, numerical behavior, and experimental comparability. Any such run must be labeled separately.
