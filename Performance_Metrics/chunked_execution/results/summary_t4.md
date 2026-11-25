# Kaggle Tesla T4 — chunked matmul → softmax → matmul

Setup: fp16, A=(8192×1024), B=(1024×32768), 10 warmup / 100 timed iterations, mean of 3 runs.
Status: `partial`. Energy = NVML power-sampling estimate (idle not subtracted).

| chunks | median latency (ms) | speedup | throughput (TFLOP/s) | peak mem (MB) | energy (J/iter) |
|---:|---:|---:|---:|---:|---:|
| 1 (baseline) | 48.76 | 1.00x | 22.5 | 1176 | 3.30 |
| 2 | 47.13 | 1.03x | 23.2 | 1048 | 3.28 |
| 4 | 47.35 | 1.03x | 23.2 | 792 | 3.22 |
| 8 | 48.20 | 1.01x | 22.8 | 600 | 3.30 |
| 16 | 51.24 | 0.95x | 21.4 | 504 | 3.48 |
| 32 | 59.81 | 0.82x | 18.4 | 456 | 4.04 |


Baseline latency varied 48.2–49.6 ms across runs (~3%), about the size of the latency gain.