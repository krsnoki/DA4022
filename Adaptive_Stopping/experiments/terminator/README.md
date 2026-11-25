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

## Results

Qwen3-1.7B, fp16, Kaggle Tesla T4, one sampled trace per problem (seed 0).
All files are in `results/`. Code commit for every row: `1c9acce`. They were
made with

```bash
python Adaptive_Stopping/experiments/terminator/summarize_results.py --train out/train \
    --eval out/eval/eval --latency out/latency --out-dir Adaptive_Stopping/experiments/terminator/results

# plots and metric rows: once on the offline rows, once on the latency rows of
# the 25 problems both latency runs finished (--metrics-csv appends, so remove
# the old files first)
R=Adaptive_Stopping/experiments/terminator/results
P=Adaptive_Stopping/experiments/terminator/plot_results.py
rm -f $R/plots_offline/metrics_rows.csv $R/plots_latency/metrics_rows.csv
gunzip -c $R/per_problem.jsonl.gz > per_problem.jsonl
python $P --inputs per_problem.jsonl --out-dir $R/plots_offline --commit 1c9acce \
    --device "Tesla T4" --metrics-csv $R/plots_offline/metrics_rows.csv
python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; c=set.intersection(*[{x['uid'] for x in r if x['method']==m} for m in {x['method'] for x in r}]); [print(json.dumps(x)) for x in r if x['uid'] in c]" \
    $R/latency_rows.jsonl > latency_common.jsonl
python $P --inputs latency_common.jsonl --out-dir $R/plots_latency --commit 1c9acce \
    --device "Tesla T4" --metrics-csv $R/plots_latency/metrics_rows.csv
```

### Probe training (`train_summary.json`)

| | value |
|---|---|
| labelled traces (train / val / test) | 969 / 110 / 432 (of 1,152 / 128 / 500) |
| probe parameters | 50.3 M (one Qwen3 decoder layer + norm + linear head) |
| optimizer steps (2 epochs, 16 traces per step) | 122, none skipped, no non-finite loss or hidden state |
| training time on one T4 | 3,560 s |
| val token macro F1 at evals 1-6 | 0.819, 0.891, **0.892**, 0.890, 0.878, 0.879 (best checkpoint kept) |
| val BCE at evals 1-6 | 0.332, 0.218, 0.219, 0.226, 0.271, 0.270 |
| test token macro F1 / accuracy (432 MATH-500 traces, prob > 0.5) | 0.848 / 0.863 (F1 of label 1: 0.896, label 0: 0.800) |

### Offline accuracy and compression, MATH-500 test, n = 500 (`summary.csv`)

Window 10, stop when more than half of the window is above the threshold.
Compression is mean thinking tokens divided by the no-stop mean; per-sample is
the mean of per-problem ratios (the paper's definition).

| method | threshold | accuracy | mean thinking tokens | compression | per-sample | stop rate |
|---|---|---|---|---|---|---|
| no stop | - | 0.870 | 3,474 | 1.000 | 1.000 | 0.000 |
| TERMINATOR | 0.5 | 0.806 | 1,200 | 0.345 | 0.333 | 0.986 |
| TERMINATOR | 0.6 | 0.836 | 1,327 | 0.382 | 0.359 | 0.980 |
| TERMINATOR | 0.7 (paper) | 0.838 | 1,472 | 0.424 | 0.390 | 0.960 |
| TERMINATOR | 0.8 | 0.850 | 1,634 | 0.470 | 0.428 | 0.918 |
| TERMINATOR | 0.9 | 0.850 | 1,848 | 0.532 | 0.487 | 0.870 |
| fixed budget (1,472 tokens) | - | 0.804 | 1,427 | 0.411 | 0.555 | 0.844 |
| oracle (cut at the labelled answer position) | - | 0.852 | 1,724 | 0.496 | 0.442 | 0.858 |

- The validation rule (the threshold with the fewest thinking tokens whose
  val accuracy is at most 1 point below no stop, 0.922 on 128 val traces)
  selected no threshold: the best
  val accuracy was 0.906 at 0.9 (`summary_val.csv`,
  `threshold_selection.json`). The table therefore lists every threshold.
- With a fixed budget equal to TERMINATOR's mean thinking length at 0.7
  (1,472 tokens), accuracy is 3.4 points lower than TERMINATOR at 0.7
  (0.804 vs 0.838).
- TERMINATOR at 0.8-0.9 is 2 points below no stop while using 47-53% of the
  thinking tokens. The oracle cut loses 1.8 points as well, so part of the
  drop comes from re-sampling the answer after a forced exit, not from the
  probe.
- Exit position minus labelled answer position, over the 432 labelled test
  traces (`plots_offline/terminator_exit_offset_hist.png` pools all five
  thresholds): median +7, +8, +11, +17.5, +52 tokens for thresholds 0.5-0.9;
  exits before the labelled position: 31.5%, 25.0%, 22.0%, 15.5%, 9.7%.

### Online latency, batch 1, HF token loop (`latency_summary.csv`)

30 MATH-500 problems, 2 h limit; no stop finished 25, so the table uses the
25 problems both runs finished. Both runs used one T4 each on the same
machine at the same time; threshold 0.7.

| method | accuracy | mean thinking tokens | median latency (s) | p95 latency (s) | tokens/s | probe ms/token | peak memory (MB) |
|---|---|---|---|---|---|---|---|
| no stop | 0.92 | 4,227 | 304.5 | 527.8 | 16.7 | - | 4,469 |
| TERMINATOR | 0.84 | 1,605 | 107.6 | 380.4 | 18.0 | 2.55 | 4,711 |

p95 is the nearest-rank value; `plots_latency/summary.csv` uses linear
interpolation (512.9 s and 339.8 s). Tokens/s counts thinking and answer
tokens; it is higher for TERMINATOR because its sequences are shorter. The
probe adds 192 MB of weights and 241 MB of peak memory. Energy was not
measured.
