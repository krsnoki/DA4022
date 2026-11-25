# TERMINATOR experiments (Qwen3-1.7B, MATH-500)

Scripts for the partial reproduction of TERMINATOR (Nagle et al., AdaptFM
2026, arXiv 2603.12529). The method code is in
`Adaptive_Stopping/methods/learned_exit/terminator/`; the paper card is
`PAPER_CARD.md` in that folder.

## Pipeline

| Step | Script | Where it ran | Output |
|---|---|---|---|
| A. traces | `generate_traces.py` | Kaggle 2x T4, vLLM | one thinking trace per problem (token ids, answer, correctness) |
| B. labels | `build_labels.py` | CPU (inside the train kernel) | first answer position per trace, train/val split |
| C. probe | `train_probe.py` | Kaggle T4 | `probe.pt`, per-token probabilities for val and test traces |
| D. offline eval | `evaluate.py` | Kaggle T4, vLLM | `per_problem.jsonl`, `summary.csv`, threshold selection |
| E. latency | `latency.py` | Kaggle 2x T4, HF loop, batch 1 | per-problem latency, tokens/s, probe overhead, peak memory |

Step D uses the fact that the probe is causal: its prediction at thinking
token t only depends on tokens up to t. The exit point that TERMINATOR would
pick during generation can therefore be read from the probe probabilities of
the full no-stop trace, and only the answer has to be generated again after
`thinking[:exit] + exit message + </think>`. The test
`test_cached_probe_matches_full_pass` checks that the online probe with its
own KV cache gives the same probabilities as one pass over the whole trace.

## Reproduction

All model work runs on Kaggle. Each Kaggle kernel is one Python script built
from the repo files by `kaggle/build_kernel.py`, which embeds the files and
appends a stage driver. The Kaggle user name is read from the local Kaggle
API config.

```bash
# from the repo root
B=Adaptive_Stopping/experiments/terminator/kaggle/build_kernel.py

# A: traces (500 MATH-500 + 2000 MATH train, 2x T4). This hit Kaggle's 12 h
# limit after 1,280 train traces; the finished shards were merged and
# uploaded as a private dataset, terminator-traces-data.
python $B --stage stage_traces.py --slug terminator-traces --out-dir kernels/traces
kaggle kernels push -p kernels/traces --accelerator NvidiaTeslaT4

# B + C: labels and probe training (about 1 h on one T4)
python $B --stage stage_train_eval.py --slug terminator-train --out-dir kernels/train \
    --env STAGES=train PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --datasets terminator-traces-data
kaggle kernels push -p kernels/train --accelerator NvidiaTeslaT4

# D: offline evaluation, one vLLM shard per T4
python $B --stage stage_train_eval.py --slug terminator-eval --out-dir kernels/eval \
    --env STAGES=eval EVAL_DEADLINE_S=14400 --datasets terminator-traces-data \
    --kernel-sources terminator-train
kaggle kernels push -p kernels/eval --accelerator NvidiaTeslaT4

# E: online latency, 30 problems, no stopping (GPU 1) and TERMINATOR (GPU 0)
# at the same time
python $B --stage stage_train_eval.py --slug terminator-latency --out-dir kernels/latency \
    --env STAGES=latency N_LATENCY=30 LAT_DEADLINE_S=7200 --kernel-sources terminator-train
kaggle kernels push -p kernels/latency --accelerator NvidiaTeslaT4

# fetch outputs
kaggle kernels output <user>/terminator-eval -p out/eval
```

The Kaggle CLI log of these kernels came back empty, so every stage writes
its own log files (`log_train.txt`, `log_eval_shard*.txt`,
`log_latency_*.txt`) into the kernel output.

`kaggle/stage_debug_nan.py` is a short check (about 2 minutes on a T4) that
prints, per decoder layer and per probe sub-module, the largest value and
the positions that are not finite, for fp16 autocast and for float32. It was
used to find why the first training run had a NaN loss (see below).

The scripts can also be run directly on any CUDA machine, for example:

```bash
python Adaptive_Stopping/experiments/terminator/generate_traces.py --split math500 --out traces_math500.jsonl
python Adaptive_Stopping/experiments/terminator/generate_traces.py --split train --n 2000 --out traces_train.jsonl
python Adaptive_Stopping/experiments/terminator/build_labels.py --traces traces_train.jsonl traces_math500.jsonl \
    --out labels.jsonl --summary labels_summary.json
python Adaptive_Stopping/experiments/terminator/train_probe.py --traces traces_train.jsonl traces_math500.jsonl \
    --labels labels.jsonl --out-dir probe/
python Adaptive_Stopping/experiments/terminator/evaluate.py --traces traces_math500.jsonl traces_train.jsonl \
    --labels labels.jsonl --probs probe/probs_test.jsonl probe/probs_val.jsonl --out-dir eval/
python Adaptive_Stopping/experiments/terminator/latency.py --method no_stop --n 30 --out latency_no_stop.jsonl
python Adaptive_Stopping/experiments/terminator/latency.py --method terminator --probe probe/probe.pt --n 30 \
    --out latency_terminator.jsonl
```

The settings are recorded in `Adaptive_Stopping/configs/terminator/qwen3_1_7b_math500.yaml`
and are the script defaults.

CPU tests (tiny random Qwen3, no downloads):

```bash
python -m pytest Adaptive_Stopping/tests -q
```
