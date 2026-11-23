# BASTION reproduction summary

Runs: runs/sweep_qwen3-8b.jsonl
External AR runs: runs/e1_qwen3-8b.jsonl
External AR records loaded: 494

| Model | Method | GSM8K Speedup | τ | MATH500 Speedup | τ | AIME25 Speedup | τ | HumanEval Speedup | τ | MBPP Speedup | τ | LCB Speedup | τ | MT-Bench Speedup | τ | Alpaca Speedup | τ | Average Speedup | τ | Pooled Speedup | τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | BASTION (adaptive) | 4.90× | 7.30 | 5.95× | 9.50 | 4.86× | 8.22 | 5.17× | 8.14 | 4.48× | 6.85 | 4.69× | 7.38 | 2.99× | 4.96 | 3.12× | 4.81 | 4.52× | 7.15 | 4.10× | 7.07 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | 4.88× | 7.44 | 5.84× | 9.17 | 4.84× | 8.11 | 5.17× | 8.32 | 4.47× | 7.12 | 4.82× | 9.31 | 2.93× | 4.78 | 3.16× | 4.93 | 4.51× | 7.40 | 4.08× | 7.19 |
| Qwen3-8B | BASTION-Fixed-32 | 4.91× | 7.22 | 5.85× | 9.31 | 4.95× | 8.18 | 5.12× | 7.99 | 4.50× | 6.85 | 4.67× | 7.14 | 2.96× | 4.78 | 3.11× | 4.79 | 4.51× | 7.03 | 4.09× | 6.99 |
| Qwen3-8B | BASTION-Fixed-64 | 4.79× | 7.71 | 5.69× | 9.61 | 4.80× | 8.57 | 5.05× | 8.58 | 4.53× | 7.47 | 4.58× | 8.02 | 2.89× | 5.01 | 3.08× | 5.21 | 4.43× | 7.52 | 4.01× | 7.36 |
| Qwen3-8B | BASTION-Fixed-128 | 4.53× | 8.12 | 5.35× | 10.10 | 4.58× | 8.98 | 4.84× | 9.10 | 4.21× | 7.73 | 4.33× | 8.83 | 2.80× | 5.32 | 2.95× | 5.50 | 4.20× | 7.96 | 3.83× | 7.82 |
| Qwen3-8B | BASTION-Fixed-256 | 3.80× | 8.26 | 4.57× | 10.54 | 3.85× | 9.30 | 4.08× | 9.32 | 3.54× | 7.97 | 3.71× | 8.43 | 2.43× | 5.95 | 2.59× | 5.76 | 3.57× | 8.19 | 3.28× | 8.18 |
| Qwen3-8B | BASTION-Fixed-512 | 2.30× | 8.69 | 2.76× | 10.76 | 2.35× | 9.68 | 2.44× | 9.62 | 2.19× | 8.44 | 2.31× | 10.24 | 1.45× | 6.16 | 1.55× | 6.22 | 2.17× | 8.73 | 1.98× | 8.56 |
| Qwen3-8B | BASTION-Fixed-1024 | 1.29× | 8.96 | 1.56× | 11.21 | 1.35× | 10.23 | 1.36× | 9.81 | 1.24× | 8.81 | 1.32× | 10.25 | 0.82× | 6.36 | 0.87× | 6.27 | 1.23× | 8.99 | 1.12× | 8.89 |

Speedup = mean TPOT(AR) / mean TPOT(method) over records with a matching AR record (same model, dataset, problem (orig_idx), turn and prompt hash). τ = AAL pooled over steps. Average = unweighted mean over the datasets shown (paper convention); Pooled = all records pooled (upstream group-alias convention).

## Per-group details

| Model | Method | Dataset | n rec | AR-matched | no AR | hash≠ | mean TPOT ms | tok/s | speedup | AAL | mean tree | mean out tok | hit max_new |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | BASTION (adaptive) | gsm8k | 16 | 16 | 0 | 0 | 4.52 | 221.3 | 4.90× | 7.30 | 34.2 | 265 | 0% |
| Qwen3-8B | BASTION (adaptive) | math500 | 16 | 16 | 0 | 0 | 3.76 | 266.0 | 5.95× | 9.50 | 33.3 | 926 | 6% |
| Qwen3-8B | BASTION (adaptive) | aime25 | 16 | 16 | 0 | 0 | 4.64 | 215.3 | 4.86× | 8.22 | 33.0 | 1674 | 50% |
| Qwen3-8B | BASTION (adaptive) | humaneval | 16 | 16 | 0 | 0 | 4.30 | 232.3 | 5.17× | 8.14 | 33.8 | 445 | 0% |
| Qwen3-8B | BASTION (adaptive) | mbpp | 16 | 16 | 0 | 0 | 4.94 | 202.6 | 4.48× | 6.85 | 33.9 | 288 | 0% |
| Qwen3-8B | BASTION (adaptive) | lcb | 16 | 16 | 0 | 0 | 4.75 | 210.4 | 4.69× | 7.38 | 33.3 | 300 | 0% |
| Qwen3-8B | BASTION (adaptive) | mt-bench | 32 | 32 | 0 | 0 | 7.42 | 134.7 | 2.99× | 4.96 | 32.8 | 473 | 3% |
| Qwen3-8B | BASTION (adaptive) | alpaca | 16 | 16 | 0 | 0 | 7.04 | 142.1 | 3.12× | 4.81 | 32.9 | 294 | 0% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | gsm8k | 16 | 16 | 0 | 0 | 4.53 | 220.6 | 4.88× | 7.44 | 44.4 | 270 | 0% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | math500 | 16 | 16 | 0 | 0 | 3.83 | 261.2 | 5.84× | 9.17 | 40.9 | 983 | 6% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | aime25 | 16 | 16 | 0 | 0 | 4.67 | 214.4 | 4.84× | 8.11 | 39.9 | 1747 | 44% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | humaneval | 16 | 16 | 0 | 0 | 4.31 | 232.0 | 5.17× | 8.32 | 43.1 | 454 | 0% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | mbpp | 16 | 16 | 0 | 0 | 4.95 | 201.9 | 4.47× | 7.12 | 42.7 | 316 | 0% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | lcb | 16 | 16 | 0 | 0 | 4.62 | 216.6 | 4.82× | 9.31 | 37.2 | 463 | 12% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | mt-bench | 32 | 32 | 0 | 0 | 7.57 | 132.1 | 2.93× | 4.78 | 38.4 | 440 | 0% |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | alpaca | 16 | 16 | 0 | 0 | 6.95 | 144.0 | 3.16× | 4.93 | 38.9 | 314 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | gsm8k | 16 | 16 | 0 | 0 | 4.51 | 221.6 | 4.91× | 7.22 | 32.0 | 270 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | math500 | 16 | 16 | 0 | 0 | 3.83 | 261.3 | 5.85× | 9.31 | 32.0 | 903 | 6% |
| Qwen3-8B | BASTION-Fixed-32 | aime25 | 16 | 16 | 0 | 0 | 4.56 | 219.2 | 4.95× | 8.18 | 32.0 | 1776 | 50% |
| Qwen3-8B | BASTION-Fixed-32 | humaneval | 16 | 16 | 0 | 0 | 4.35 | 230.1 | 5.12× | 7.99 | 32.0 | 451 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | mbpp | 16 | 16 | 0 | 0 | 4.92 | 203.3 | 4.50× | 6.85 | 32.0 | 293 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | lcb | 16 | 16 | 0 | 0 | 4.77 | 209.5 | 4.67× | 7.14 | 32.0 | 322 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | mt-bench | 32 | 32 | 0 | 0 | 7.49 | 133.5 | 2.96× | 4.78 | 32.0 | 446 | 0% |
| Qwen3-8B | BASTION-Fixed-32 | alpaca | 16 | 16 | 0 | 0 | 7.04 | 142.1 | 3.11× | 4.79 | 32.0 | 293 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | gsm8k | 16 | 16 | 0 | 0 | 4.62 | 216.3 | 4.79× | 7.71 | 64.0 | 271 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | math500 | 16 | 16 | 0 | 0 | 3.93 | 254.5 | 5.69× | 9.61 | 64.0 | 875 | 6% |
| Qwen3-8B | BASTION-Fixed-64 | aime25 | 16 | 16 | 0 | 0 | 4.71 | 212.5 | 4.80× | 8.57 | 64.0 | 1778 | 56% |
| Qwen3-8B | BASTION-Fixed-64 | humaneval | 16 | 16 | 0 | 0 | 4.41 | 226.8 | 5.05× | 8.58 | 64.0 | 451 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | mbpp | 16 | 16 | 0 | 0 | 4.88 | 204.9 | 4.53× | 7.47 | 64.0 | 333 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | lcb | 16 | 16 | 0 | 0 | 4.87 | 205.5 | 4.58× | 8.02 | 64.0 | 250 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | mt-bench | 32 | 32 | 0 | 0 | 7.66 | 130.5 | 2.89× | 5.01 | 64.0 | 466 | 0% |
| Qwen3-8B | BASTION-Fixed-64 | alpaca | 16 | 16 | 0 | 0 | 7.11 | 140.7 | 3.08× | 5.21 | 64.0 | 297 | 0% |
| Qwen3-8B | BASTION-Fixed-128 | gsm8k | 16 | 16 | 0 | 0 | 4.89 | 204.4 | 4.53× | 8.12 | 128.0 | 262 | 0% |
| Qwen3-8B | BASTION-Fixed-128 | math500 | 16 | 16 | 0 | 0 | 4.18 | 239.0 | 5.35× | 10.10 | 128.0 | 906 | 12% |
| Qwen3-8B | BASTION-Fixed-128 | aime25 | 16 | 16 | 0 | 0 | 4.93 | 202.7 | 4.58× | 8.98 | 128.0 | 1733 | 31% |
| Qwen3-8B | BASTION-Fixed-128 | humaneval | 16 | 16 | 0 | 0 | 4.60 | 217.2 | 4.84× | 9.10 | 128.0 | 443 | 0% |
| Qwen3-8B | BASTION-Fixed-128 | mbpp | 16 | 16 | 0 | 0 | 5.25 | 190.4 | 4.21× | 7.73 | 128.0 | 307 | 0% |
| Qwen3-8B | BASTION-Fixed-128 | lcb | 16 | 16 | 0 | 0 | 5.15 | 194.4 | 4.33× | 8.83 | 128.0 | 372 | 6% |
| Qwen3-8B | BASTION-Fixed-128 | mt-bench | 32 | 32 | 0 | 0 | 7.91 | 126.5 | 2.80× | 5.32 | 128.0 | 442 | 0% |
| Qwen3-8B | BASTION-Fixed-128 | alpaca | 16 | 16 | 0 | 0 | 7.43 | 134.6 | 2.95× | 5.50 | 128.0 | 295 | 0% |
| Qwen3-8B | BASTION-Fixed-256 | gsm8k | 16 | 16 | 0 | 0 | 5.83 | 171.5 | 3.80× | 8.26 | 256.0 | 260 | 0% |
| Qwen3-8B | BASTION-Fixed-256 | math500 | 16 | 16 | 0 | 0 | 4.89 | 204.3 | 4.57× | 10.54 | 256.0 | 930 | 6% |
| Qwen3-8B | BASTION-Fixed-256 | aime25 | 16 | 16 | 0 | 0 | 5.86 | 170.6 | 3.85× | 9.30 | 256.0 | 1732 | 44% |
| Qwen3-8B | BASTION-Fixed-256 | humaneval | 16 | 16 | 0 | 0 | 5.46 | 183.2 | 4.08× | 9.32 | 256.0 | 445 | 0% |
| Qwen3-8B | BASTION-Fixed-256 | mbpp | 16 | 16 | 0 | 0 | 6.26 | 159.8 | 3.54× | 7.97 | 256.0 | 312 | 0% |
| Qwen3-8B | BASTION-Fixed-256 | lcb | 16 | 16 | 0 | 0 | 6.01 | 166.5 | 3.71× | 8.43 | 256.0 | 275 | 0% |
| Qwen3-8B | BASTION-Fixed-256 | mt-bench | 32 | 32 | 0 | 0 | 9.11 | 109.7 | 2.43× | 5.95 | 256.0 | 463 | 3% |
| Qwen3-8B | BASTION-Fixed-256 | alpaca | 16 | 16 | 0 | 0 | 8.46 | 118.2 | 2.59× | 5.76 | 256.0 | 302 | 0% |
| Qwen3-8B | BASTION-Fixed-512 | gsm8k | 16 | 16 | 0 | 0 | 9.62 | 104.0 | 2.30× | 8.69 | 512.0 | 267 | 0% |
| Qwen3-8B | BASTION-Fixed-512 | math500 | 16 | 16 | 0 | 0 | 8.12 | 123.1 | 2.76× | 10.76 | 512.0 | 861 | 6% |
| Qwen3-8B | BASTION-Fixed-512 | aime25 | 16 | 16 | 0 | 0 | 9.62 | 103.9 | 2.35× | 9.68 | 512.0 | 1801 | 56% |
| Qwen3-8B | BASTION-Fixed-512 | humaneval | 16 | 16 | 0 | 0 | 9.13 | 109.5 | 2.44× | 9.62 | 512.0 | 458 | 0% |
| Qwen3-8B | BASTION-Fixed-512 | mbpp | 16 | 16 | 0 | 0 | 10.12 | 98.9 | 2.19× | 8.44 | 512.0 | 318 | 0% |
| Qwen3-8B | BASTION-Fixed-512 | lcb | 16 | 16 | 0 | 0 | 9.63 | 103.9 | 2.31× | 10.24 | 512.0 | 347 | 6% |
| Qwen3-8B | BASTION-Fixed-512 | mt-bench | 32 | 32 | 0 | 0 | 15.31 | 65.3 | 1.45× | 6.16 | 512.0 | 478 | 3% |
| Qwen3-8B | BASTION-Fixed-512 | alpaca | 16 | 16 | 0 | 0 | 14.12 | 70.8 | 1.55× | 6.22 | 512.0 | 311 | 0% |
| Qwen3-8B | BASTION-Fixed-1024 | gsm8k | 16 | 16 | 0 | 0 | 17.17 | 58.2 | 1.29× | 8.96 | 1024.0 | 272 | 0% |
| Qwen3-8B | BASTION-Fixed-1024 | math500 | 16 | 16 | 0 | 0 | 14.32 | 69.8 | 1.56× | 11.21 | 1024.0 | 969 | 12% |
| Qwen3-8B | BASTION-Fixed-1024 | aime25 | 16 | 16 | 0 | 0 | 16.78 | 59.6 | 1.35× | 10.23 | 1024.0 | 1648 | 50% |
| Qwen3-8B | BASTION-Fixed-1024 | humaneval | 16 | 16 | 0 | 0 | 16.33 | 61.2 | 1.36× | 9.81 | 1024.0 | 460 | 0% |
| Qwen3-8B | BASTION-Fixed-1024 | mbpp | 16 | 16 | 0 | 0 | 17.81 | 56.2 | 1.24× | 8.81 | 1024.0 | 323 | 0% |
| Qwen3-8B | BASTION-Fixed-1024 | lcb | 16 | 16 | 0 | 0 | 16.86 | 59.3 | 1.32× | 10.25 | 1024.0 | 366 | 6% |
| Qwen3-8B | BASTION-Fixed-1024 | mt-bench | 32 | 32 | 0 | 0 | 27.17 | 36.8 | 0.82× | 6.36 | 1024.0 | 472 | 3% |
| Qwen3-8B | BASTION-Fixed-1024 | alpaca | 16 | 16 | 0 | 0 | 25.28 | 39.6 | 0.87× | 6.27 | 1024.0 | 314 | 0% |
