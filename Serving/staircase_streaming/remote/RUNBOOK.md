# Runbook: full GPU run of the staircase-streaming benchmark

Everything except this run has already been built and tested on CPU (the mock backend and a fake HTTP server). On the GPU machine, one script does the rest.

## 1. Get a GPU machine

| Option | Use config | Notes |
|---|---|---|
| **1× 24 GB GPU** (RTX 4090 / 3090, L4, A10G) on RunPod, Vast.ai, Lambda, … | `configs/full.yaml` (reported results) or `configs/default.yaml` (smaller, quicker) | Recommended. Easiest and most reliable |
| 1× 16 GB GPU (RTX 4060 Ti 16 GB, A4000, …) | `configs/full.yaml` with `gpu_memory_utilization` 0.36 / 0.36 / 0.18 | Tighter |
| **Kaggle notebook, 2× T4** (free) | `configs/kaggle_2xT4.yaml` | Free. T4 is an older GPU (fp16 only); see troubleshooting |

Requirements:
- Linux (Ubuntu 22.04/24.04 images are fine) with an NVIDIA driver that supports CUDA 12.
- Python 3.10–3.12.
- **≥ 50 GB disk** (vLLM + PyTorch ≈ 10 GB, models ≈ 7.5 GB, plus cache).
- Internet access. **No Hugging Face token is needed** (all models are ungated).

On RunPod, a "PyTorch" template works. Attaching a volume keeps the downloaded models between sessions; in that case set `HF_HOME=/workspace/hf_cache`.

## 2. Put the code on the machine

Option A: push the branch to the fork from a local machine, then clone it on the GPU machine:

```bash
# local machine
git push -u origin staircase-streaming
# GPU machine
git clone -b staircase-streaming https://github.com/nit-singh/DA4022.git
cd DA4022
```

Option B: no push. Create a zip on the local machine and upload it (scp, or the provider's file browser):

```bash
git archive --format=zip -o staircase.zip staircase-streaming
```

## 3. Run it

We use `tmux` (or `nohup`) so an SSH disconnect doesn't kill the run.

**First, a sanity check.** This installs everything, downloads the models, starts the servers, runs the tests, and prints 3 baseline and staircase answers:

```bash
tmux new -s stair
SANITY_ONLY=1 bash Serving/staircase_streaming/remote/run_all.sh Serving/staircase_streaming/configs/full.yaml
```

Look at the printed answers. They should be real sentences, and the staircase TTFT should be clearly lower than the baseline's. If that looks right, **run the full benchmark** (it reuses the installed environment):

```bash
SKIP_INSTALL=1 bash Serving/staircase_streaming/remote/run_all.sh Serving/staircase_streaming/configs/full.yaml
```

Measured runtime with `configs/full.yaml` (2026-11-24, 1× RTX 4090): 1 792 latency requests + 2 638 GSM8K requests took about 5 h 40 min. `configs/default.yaml` is a smaller, quicker workload. For a quick end-to-end check first, add `LIMIT=2`. The run is **resumable**: if it stops, run the same command again and it continues where it left off.

### Kaggle (2× T4)

1. Notebook settings: Accelerator **GPU T4 x2**, Internet **on** (needs a phone-verified account).
2. Upload `staircase.zip` as a Kaggle Dataset named `da4022-staircase`, or `git clone` the pushed branch.
3. Run these cells:

```python
!cp -r /kaggle/input/da4022-staircase/DA4022 /kaggle/working/ 2>/dev/null || true   # or: !git clone -b staircase-streaming https://github.com/nit-singh/DA4022.git /kaggle/working/DA4022
%cd /kaggle/working/DA4022
!SANITY_ONLY=1 bash Serving/staircase_streaming/remote/run_all.sh Serving/staircase_streaming/configs/kaggle_2xT4.yaml
!SKIP_INSTALL=1 bash Serving/staircase_streaming/remote/run_all.sh Serving/staircase_streaming/configs/kaggle_2xT4.yaml
```

Kaggle sessions time out after several hours. Results so far are kept in `results/` inside `/kaggle/working`, so re-running continues the run. Download the output before the session ends.

## 4. Bring the results back

At the end, the script prints the path of a tarball such as `results/staircase_streaming/full_20261124_1527.tar.gz`. It contains:
- `metrics.csv`: the repository schema.
- `summary_detailed.csv` and `summary.json`.
- Raw per-request JSONL.
- The plots.
- Logs: the vLLM server logs, `sanity.log` and `benchmark.log`.
- `pip_freeze.txt` and `env_report.json`.

Copy it to the local machine (with `scp`, the provider's file browser, or Kaggle's "Output" download) and extract it into `DA4022/results/staircase_streaming/` (this folder is gitignored). The reported numbers in `PAPER_CARD.md` and `Performance_Metrics/` are taken from `metrics.csv`, `summary_detailed.csv` and `raw_requests.jsonl` in this archive.

**Then stop or delete the GPU instance.** Paid providers keep billing until the instance is deleted.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Server exits with CUDA out-of-memory / "free memory ... less than desired" | Lower that server's `gpu_memory_utilization` or `max_model_len` (e.g. 4096) in the config. Make sure no other process is using the GPU (check `nvidia-smi`). |
| Out of memory while capturing CUDA graphs | Add `"--enforce-eager"` to `vllm.extra_args` in the config. |
| T4 / Kaggle: errors about bf16 or the attention backend | `dtype: half` is already set. Also add `"--enforce-eager"`. If vLLM still refuses the GPU, note the error and use a 24 GB Ampere/Ada GPU instead. |
| Server exits with `Ninja build failed` / `nvcc fatal : Unknown option '--compress-mode=size'` | The image's system CUDA toolkit is older than FlashInfer's JIT expects (seen with `pytorch/pytorch:2.5.1-cuda12.4` + vLLM 0.34.0 / FlashInfer 0.7.3). Run with `export VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM then uses its PyTorch sampler and needs no JIT build. The 2026-11-24 RTX 4090 run used this. |
| HTTP 400 mentioning `continue_final_message` | The installed vLLM is too old for the chat "continue" extension. Upgrade: `pip install -U vllm`. |
| A port is already in use | Change the `port` values under `servers:`. |
| Run interrupted | Re-run the same command. Finished requests are skipped (`raw_requests.jsonl`). |
| Need to see what a server is doing | `tail -f results/staircase_streaming/<run>/vllm_logs/<model>.log` |

Manual control, if we need it:

```bash
source .venv-gpu/bin/activate
python -m Serving.staircase_streaming.launch_vllm --config <cfg> start|status|stop
python -m Serving.staircase_streaming.run_benchmark --config <cfg> [--sanity] [--limit N] [--skip-quality]
```
