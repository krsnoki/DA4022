# Jacobian-noise experiments: how to run

Method code: [`quantization/qat/jacobian_noise/`](../../quantization/qat/jacobian_noise/README.md) (paper card, design choices, limitations).

## 0. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or the CUDA build for your GPU
pip install -r Quantization_and_Pruning/requirements-jacobian-noise.txt
pip install lm-eval            # optional: zero-shot tasks for the LLM evaluation
```

## 1. CPU: tests and smoke runs (no GPU, no downloads)

```bash
python -m pytest Quantization_and_Pruning/tests -q                     # ~15 s
MODEL=smoke_lm     bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh
MODEL=smoke_siglip bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh
```

The smoke configs use tiny randomly initialised models and synthetic data. Their numbers mean nothing; they only check the plumbing.

## 2. The experiment matrix

| Run | Fine-tune | Noise (`noise.mode`) | Purpose |
|---|---|---|---|
| R0 | no | – | original checkpoint (`EVAL_PRETRAINED=1`) |
| R1 | yes | `none` | **baseline B** |
| R2 | yes | `jacobian` | **method B+N** |
| R3 | yes | `uniform_matched` | **ablation**: same noise energy, no Jacobian shaping |
| R4 (optional) | yes | `jacobian` with `noise.alpha` ∈ {0.1, 0.25, 1.0, 2.0} | α sensitivity |
| R5 (optional) | yes | `jacobian_layer` | row-wise vs layer-wise σ |

Every checkpoint is evaluated at `FP16, W8A8, W6A6, (W5A5), W4A8, W4A4, W4A4QK4`, with RTN weights and optionally GPTQ weights (`WEIGHT_METHODS="rtn gptq"`). Activations are always per-token RTN. Use at least 3 seeds for the headline comparison.

```bash
# everything for one model (train + eval + summary):
MODEL=siglip224 SEEDS="0 1 2" bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh
MODEL=qwen05 WEIGHT_METHODS="rtn gptq" bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh

# single steps:
python Quantization_and_Pruning/experiments/jacobian_noise/train_siglip.py \
    --config Quantization_and_Pruning/configs/jacobian_noise/siglip_base_224.yaml --set noise.mode=jacobian experiment.seed=0
python Quantization_and_Pruning/experiments/jacobian_noise/eval_quant.py \
    --config Quantization_and_Pruning/configs/jacobian_noise/eval_siglip_base_224.yaml \
    --set model.path=outputs/jacobian_noise/siglip224__jacobian_a0.5__s0/checkpoint
python Quantization_and_Pruning/experiments/jacobian_noise/summarize.py \
    --results outputs/jacobian_noise/eval/results.csv --out outputs/jacobian_noise/summary --plot
```

Any config value can be overridden with `--set a.b=value`. For `run_matrix.sh`, use `EXTRA_TRAIN="train.max_steps=200"` / `EXTRA_EVAL="eval.siglip.num_eval=2000"`.

### Outputs
- `outputs/jacobian_noise/<run>/`: `train_log.jsonl` (loss, σ, val metrics), `run.json` (resolved config, environment, σ history), `noise_state.pt`, and `checkpoint/`. `run_matrix.sh` deletes the checkpoint after evaluation unless `KEEP_CKPT=1`.
- `outputs/jacobian_noise/eval/results.csv`: one row per (checkpoint, quantization, metric) in the schema of [`Performance_Metrics/README_metrics.md`](../../../Performance_Metrics/README_metrics.md).
- `eval/results_detail.jsonl`: everything, including the quantization config.
- `eval/jacobian_diagnostics.jsonl`: mean ‖J‖² per layer of each FP checkpoint. Does noise training make attention sharper?
- `summary/summary.md`: tables of mean ± std over seeds, with Δ vs the no-noise baseline (copy into the report / paper card).

`outputs/` is git-ignored. Commit only the final `results.csv` / `summary.md` you report.

## 3. Running on Kaggle (free GPU)

Kaggle gives ~30 GPU-hours per week per account (T4 or P100, 16 GB), with up to 12 h per session. You need a phone-verified account (for internet access in kernels) and an API token in `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create New Token).

```bash
pip install kaggle
git push origin jacobian-noise                       # the job clones your fork at this branch

python Quantization_and_Pruning/experiments/jacobian_noise/kaggle/make_job.py \
    --user <kaggle-username> --name jn-siglip224-s0 \
    --repo https://github.com/<github-user>/DA4022.git --branch jacobian-noise \
    --command 'MODEL=siglip224 SEEDS="0" bash Quantization_and_Pruning/experiments/jacobian_noise/run_matrix.sh'
kaggle kernels push   -p kaggle_jobs/jn-siglip224-s0
kaggle kernels status <kaggle-username>/jn-siglip224-s0
kaggle kernels output <kaggle-username>/jn-siglip224-s0 -p results/jn-siglip224-s0
```

The job:
1. prints `nvidia-smi`, clones the branch and installs the requirements;
2. **runs the CPU test-suite first** (it fails fast if the environment is broken);
3. runs the command, writing results to `/kaggle/working/outputs`.

Caches and checkpoints go to `/kaggle/tmp`. For a private fork, add a Kaggle secret `GITHUB_TOKEN`. For lm-eval tasks, add `--extra-pip "lm-eval"`.

Notes:
- Keep each job under ~10 h. Split by seed, or by mode with `MODES=jacobian`.
- Merge results from several jobs by concatenating their `results.csv` files (keep one header), then run `summarize.py`.
- The Kaggle GPU type (T4 vs P100) is chosen in the kernel's settings page. Record it; it appears in every results row anyway.
- Never run the noise and baseline arms of a comparison with different configs. Only `noise.*` and `experiment.seed` should differ.

### Rough budget (estimates, not measured yet)
| Job | Estimate on a T4 |
|---|---|
| SigLIP-224 fine-tune, 1500 steps × 32 images (streamed) | ~0.5–1 h |
| SigLIP-224 eval, 6 bit settings × 10k val images | ~0.5 h |
| Qwen2.5-0.5B fine-tune, 1000 steps × 8 sequences | ~0.5–1 h |
| Qwen2.5-0.5B eval, 6 settings, WikiText-2 only | ~0.3 h |
| + lm-eval 7 tasks × 500 examples per setting | +1–2 h |

Replace these with measured times once the first jobs finish.
