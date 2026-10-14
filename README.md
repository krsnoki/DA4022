
# AdaptFM-DA4022: Resource-Adaptive Deep Learning

This repository is a **research and engineering library** for implementing and comparing methods that make deep-learning inference adapt to hardware, latency, memory, energy, or quality constraints. AdaptFM workshop papers are used as research starting points; the repository is responsible for a common implementation interface, reproducibility, and comparable benchmarks.

> **Design rule:** backbones, data pipelines, adaptation methods, and metrics are separate concepts. Do not put all of them into a paper-specific folder.

## Repository map

| Path | Responsibility | Put here |
|---|---|---|
| `Backbones/` | Reusable baseline models | Supervised, self-supervised, generative-pretrained, and task-finetuned model wrappers/metadata |
| `Dataloaders/` | Dataset and batch construction | Train/eval/calibration splits, preprocessing, collators, deterministic sampling |
| `Quantization_and_Pruning/` | Model compression | PTQ/QAT, low-bit formats, weight/layer/token pruning, structured sparsity |
| `Attention/` | Attention and context efficiency | Efficient/sparse/linear attention, KV-related attention methods, token/context compression |
| `Mixture_of_Experts/` | Sparse and elastic experts | Expert routing, expert pruning, load balancing, MoE quantization and serving |
| `Adaptive_Stopping/` | Variable-depth/variable-compute inference | Early exit, learned stopping, adaptive reasoning budgets, confidence-based stopping |
| `LoRA/` | Parameter-efficient adaptation baseline | LoRA adapters, rank/target-module configs, merge/unmerge tests |
| `Performance_Metrics/` | Shared experiment protocol | Latency, throughput, memory, quality, energy, profiling and report schemas |
| `PAPER_INDEX.md` | Cross-repository paper map | Paper-to-module assignments and direct paper links |
| `templates/` | Contribution templates | Paper implementation card, model card, dataset card |

The module README files explain each module's scope and link relevant papers. Papers may appear in more than one module when methods genuinely cross boundaries; choose one **primary owner** for implementation and cross-link the other modules.

## Important terminology

- **Backbone**: the base network/architecture and its pretrained weights, such as ResNet, ViT, BERT, a decoder-only language model, or a vision-language model.
- **Pretraining regime**: how the base weights were learned. Use `supervised`, `self_supervised`, `unsupervised_legacy`, or `unknown`; record the actual objective rather than relying only on labels.
- **Task-finetuned model**: a backbone whose weights have also been tuned on a labelled or instruction-following task. Keep its parent backbone and fine-tuning dataset in metadata.
- **Adaptation method**: a method applied to or used with a backbone, e.g. quantization, pruning, LoRA, attention optimization, MoE routing, or early stopping.
- **Dataloader**: the code that turns a dataset/split into batches. Dataloaders do not belong inside a backbone or paper implementation.

In modern ML, many methods casually called “unsupervised” are more precisely **self-supervised** (e.g. masked-token or next-token prediction). Preserve that distinction in metadata.

## Quick start

The repository currently documents research modules; install requirements only for the module you are working on. Avoid one giant environment containing every optional CUDA/kernel package.

```bash
git clone <your-repository-url>
cd AdaptFM-DA4022
python -m venv .venv
source .venv/bin/activate             # Linux/macOS
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your OS, Python version, and CPU/CUDA setup. Then install module-specific dependencies as documented by that module. Do not assume a CUDA implementation will run on CPU or on a different GPU architecture.

Check local hardware before a benchmark:

```bash
python -c "import platform; print(platform.platform())"
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count())"
```

## Minimum standard for every implementation

1. **Name the paper and method version.** Link its paper and record any upstream code, license, or unavailable assets.
2. **Keep a baseline.** Run the same backbone, dataset split, input length/resolution, batch size, and evaluation settings without the method.
3. **Record configuration.** Save model ID/revision, dataset version/split, seeds, precision, batch size, input shapes, software versions, hardware, and all relevant method hyperparameters.
4. **Measure more than parameter count.** Report task quality plus runtime latency/throughput and peak memory. Report energy only when a real energy measurement and the measurement procedure are available.
5. **Test correctness.** Include a small CPU smoke test where feasible, shape/dtype tests, and deterministic/config validation.
6. **Be transparent.** Label results as `reproduced`, `partial`, `approximate`, or `not reproduced`. Do not present estimated latency or unmeasured energy as measured results.
7. **Respect licenses and data access rules.** Do not commit model weights, private data, dataset archives, API keys, or large generated artifacts. Supply download instructions and checksums where permitted.

## Paper implementation workflow

1. Select a paper from [`PAPER_INDEX.md`](PAPER_INDEX.md) and open an issue before coding.
2. Write down the expected input/output contract, required backbone(s), data needs, hardware assumptions, and baseline.
3. Add a paper card using [`templates/PAPER_IMPLEMENTATION_TEMPLATE.md`](templates/PAPER_IMPLEMENTATION_TEMPLATE.md).
4. Implement the method in the owning module; keep experiment-specific scripts/configs outside shared core utilities.
5. Add a minimal test and a reproduction command.
6. Add one row to the result schema in `Performance_Metrics/README_metrics.md` and include an unmodified baseline result.
7. Submit a PR that includes docs, config, tests, and limitations.

Suggested status labels: `proposed`, `reading`, `baseline-ready`, `implementing`, `tested`, `reproduced`, `partial-reproduction`, `blocked`.

## Proposed growth structure

```text
AdaptFM-DA4022/
├── README.md
├── PAPER_INDEX.md
├── Adaptive_Stopping/
├── Attention/
├── Backbones/
│   ├── Supervised/
│   ├── Self_Supervised/
│   ├── Generative_Pretrained/
│   └── Task_Finetuned/
├── Dataloaders/
│   ├── supervised/
│   ├── self_supervised/
│   ├── calibration/
│   ├── evaluation/
│   └── synthetic/
├── LoRA/
├── Mixture_of_Experts/
├── Performance_Metrics/
├── Quantization_and_Pruning/
└── templates/
```

Only create code directories when they have an owner and a concrete implementation. Empty scaffolding is not a contribution by itself.

## Source of paper list

The paper map is based on the [official AdaptFM accepted-paper list](https://adaptfm.gitlab.io/accepted-papers) and the workshop site ([AdaptFM: Resource-Adaptive Foundation Model Inference](https://adaptfm.gitlab.io/)). The categories in this repository are **our engineering classification**, not official workshop tracks. Paper relevance and reproducibility should be reviewed when a contributor starts work; acceptance does not imply that public code or checkpoints exist.
