# BASTION reproduction summary

Runs: runs/e1_qwen3-4b.jsonl, runs/e1_qwen3-8b.jsonl
External AR runs: none
External AR records loaded: 0

| Model | Method | GSM8K Speedup | τ | MATH500 Speedup | τ | AIME25 Speedup | τ | HumanEval Speedup | τ | MBPP Speedup | τ | LCB Speedup | τ | MT-Bench Speedup | τ | Alpaca Speedup | τ | Average Speedup | τ | Pooled Speedup | τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | DFlash | 4.62× | 6.23 | 5.57× | 8.19 | 4.49× | 7.02 | 4.52× | 6.52 | 4.14× | 5.50 | 3.97× | 5.89 | 2.26× | 3.25 | 2.07× | 2.96 | 3.95× | 5.69 | 3.43× | 5.30 |
| Qwen3-4B | BASTION (adaptive) | 5.26× | 8.26 | 5.88× | 10.01 | 4.88× | 8.96 | 5.15× | 8.44 | 4.93× | 7.67 | 4.53× | 8.12 | 2.79× | 4.79 | 2.60× | 4.36 | 4.50× | 7.58 | 4.05× | 7.22 |
| Qwen3-8B | DFlash | 4.83× | 6.28 | 5.71× | 8.04 | 4.97× | 7.15 | 4.76× | 6.42 | 4.27× | 5.55 | 4.39× | 6.62 | 2.28× | 3.14 | 2.14× | 2.90 | 4.17× | 5.76 | 3.57× | 5.25 |
| Qwen3-8B | BASTION (adaptive) | 5.17× | 7.85 | 5.78× | 9.31 | 4.98× | 8.34 | 5.17× | 8.06 | 4.65× | 7.01 | 4.67× | 7.33 | 2.71× | 4.33 | 2.60× | 4.02 | 4.47× | 7.03 | 4.00× | 6.62 |

Speedup = mean TPOT(AR) / mean TPOT(method) over records with a matching AR record (same model, dataset, problem (orig_idx), turn and prompt hash). τ = AAL pooled over steps. Average = unweighted mean over the datasets shown (paper convention); Pooled = all records pooled (upstream group-alias convention).

## Per-group details

| Model | Method | Dataset | n rec | AR-matched | no AR | hash≠ | mean TPOT ms | tok/s | speedup | AAL | mean tree | mean out tok | hit max_new |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | AR | gsm8k | 64 | 64 | 0 | 0 | 13.75 | 72.7 | 1.00× | – | – | 293 | 0% |
| Qwen3-4B | AR | math500 | 64 | 64 | 0 | 0 | 13.95 | 71.7 | 1.00× | – | – | 890 | 9% |
| Qwen3-4B | AR | aime25 | 30 | 30 | 0 | 0 | 14.20 | 70.4 | 1.00× | – | – | 1732 | 50% |
| Qwen3-4B | AR | humaneval | 64 | 64 | 0 | 0 | 13.84 | 72.2 | 1.00× | – | – | 431 | 2% |
| Qwen3-4B | AR | mbpp | 64 | 64 | 0 | 0 | 13.75 | 72.7 | 1.00× | – | – | 284 | 0% |
| Qwen3-4B | AR | lcb | 64 | 64 | 0 | 0 | 13.88 | 72.1 | 1.00× | – | – | 258 | 3% |
| Qwen3-4B | AR | mt-bench | 80 | 80 | 0 | 0 | 13.87 | 72.1 | 1.00× | – | – | 487 | 0% |
| Qwen3-4B | AR | alpaca | 64 | 64 | 0 | 0 | 13.68 | 73.1 | 1.00× | – | – | 382 | 0% |
| Qwen3-4B | DFlash | gsm8k | 64 | 64 | 0 | 0 | 2.98 | 335.9 | 4.62× | 6.23 | – | 293 | 0% |
| Qwen3-4B | DFlash | math500 | 64 | 64 | 0 | 0 | 2.51 | 399.1 | 5.57× | 8.19 | – | 902 | 9% |
| Qwen3-4B | DFlash | aime25 | 30 | 30 | 0 | 0 | 3.16 | 316.1 | 4.49× | 7.02 | – | 1790 | 63% |
| Qwen3-4B | DFlash | humaneval | 64 | 64 | 0 | 0 | 3.06 | 326.4 | 4.52× | 6.52 | – | 441 | 2% |
| Qwen3-4B | DFlash | mbpp | 64 | 64 | 0 | 0 | 3.32 | 301.4 | 4.14× | 5.50 | – | 287 | 0% |
| Qwen3-4B | DFlash | lcb | 64 | 64 | 0 | 0 | 3.49 | 286.4 | 3.97× | 5.89 | – | 262 | 2% |
| Qwen3-4B | DFlash | mt-bench | 80 | 80 | 0 | 0 | 6.13 | 163.1 | 2.26× | 3.25 | – | 483 | 0% |
| Qwen3-4B | DFlash | alpaca | 64 | 64 | 0 | 0 | 6.62 | 151.0 | 2.07× | 2.96 | – | 388 | 0% |
| Qwen3-4B | BASTION (adaptive) | gsm8k | 64 | 64 | 0 | 0 | 2.62 | 382.4 | 5.26× | 8.26 | 59.5 | 302 | 0% |
| Qwen3-4B | BASTION (adaptive) | math500 | 64 | 64 | 0 | 0 | 2.37 | 421.3 | 5.88× | 10.01 | 56.2 | 911 | 12% |
| Qwen3-4B | BASTION (adaptive) | aime25 | 30 | 30 | 0 | 0 | 2.91 | 343.8 | 4.88× | 8.96 | 59.2 | 1836 | 60% |
| Qwen3-4B | BASTION (adaptive) | humaneval | 64 | 64 | 0 | 0 | 2.69 | 372.0 | 5.15× | 8.44 | 57.7 | 451 | 2% |
| Qwen3-4B | BASTION (adaptive) | mbpp | 64 | 64 | 0 | 0 | 2.79 | 358.7 | 4.93× | 7.67 | 61.3 | 277 | 0% |
| Qwen3-4B | BASTION (adaptive) | lcb | 64 | 64 | 0 | 0 | 3.06 | 326.5 | 4.53× | 8.12 | 57.1 | 258 | 3% |
| Qwen3-4B | BASTION (adaptive) | mt-bench | 80 | 80 | 0 | 0 | 4.97 | 201.4 | 2.79× | 4.79 | 62.9 | 480 | 1% |
| Qwen3-4B | BASTION (adaptive) | alpaca | 64 | 64 | 0 | 0 | 5.25 | 190.4 | 2.60× | 4.36 | 63.1 | 399 | 0% |
| Qwen3-8B | AR | gsm8k | 64 | 64 | 0 | 0 | 22.17 | 45.1 | 1.00× | – | – | 300 | 0% |
| Qwen3-8B | AR | math500 | 64 | 64 | 0 | 0 | 22.34 | 44.8 | 1.00× | – | – | 856 | 12% |
| Qwen3-8B | AR | aime25 | 30 | 30 | 0 | 0 | 22.59 | 44.3 | 1.00× | – | – | 1750 | 57% |
| Qwen3-8B | AR | humaneval | 64 | 64 | 0 | 0 | 22.26 | 44.9 | 1.00× | – | – | 438 | 0% |
| Qwen3-8B | AR | mbpp | 64 | 64 | 0 | 0 | 22.12 | 45.2 | 1.00× | – | – | 280 | 0% |
| Qwen3-8B | AR | lcb | 64 | 64 | 0 | 0 | 22.20 | 45.0 | 1.00× | – | – | 213 | 2% |
| Qwen3-8B | AR | mt-bench | 80 | 80 | 0 | 0 | 22.21 | 45.0 | 1.00× | – | – | 460 | 0% |
| Qwen3-8B | AR | alpaca | 64 | 64 | 0 | 0 | 21.99 | 45.5 | 1.00× | – | – | 384 | 0% |
| Qwen3-8B | DFlash | gsm8k | 64 | 64 | 0 | 0 | 4.59 | 217.8 | 4.83× | 6.28 | – | 302 | 0% |
| Qwen3-8B | DFlash | math500 | 64 | 64 | 0 | 0 | 3.91 | 255.5 | 5.71× | 8.04 | – | 837 | 9% |
| Qwen3-8B | DFlash | aime25 | 30 | 30 | 0 | 0 | 4.54 | 220.0 | 4.97× | 7.15 | – | 1691 | 57% |
| Qwen3-8B | DFlash | humaneval | 64 | 64 | 0 | 0 | 4.68 | 213.7 | 4.76× | 6.42 | – | 437 | 0% |
| Qwen3-8B | DFlash | mbpp | 64 | 64 | 0 | 0 | 5.18 | 193.1 | 4.27× | 5.55 | – | 283 | 0% |
| Qwen3-8B | DFlash | lcb | 64 | 64 | 0 | 0 | 5.06 | 197.6 | 4.39× | 6.62 | – | 237 | 3% |
| Qwen3-8B | DFlash | mt-bench | 80 | 80 | 0 | 0 | 9.72 | 102.9 | 2.28× | 3.14 | – | 458 | 0% |
| Qwen3-8B | DFlash | alpaca | 64 | 64 | 0 | 0 | 10.25 | 97.5 | 2.14× | 2.90 | – | 377 | 0% |
| Qwen3-8B | BASTION (adaptive) | gsm8k | 64 | 64 | 0 | 0 | 4.28 | 233.5 | 5.17× | 7.85 | 34.1 | 299 | 0% |
| Qwen3-8B | BASTION (adaptive) | math500 | 64 | 64 | 0 | 0 | 3.86 | 258.8 | 5.78× | 9.31 | 33.5 | 867 | 11% |
| Qwen3-8B | BASTION (adaptive) | aime25 | 30 | 30 | 0 | 0 | 4.54 | 220.3 | 4.98× | 8.34 | 33.1 | 1677 | 57% |
| Qwen3-8B | BASTION (adaptive) | humaneval | 64 | 64 | 0 | 0 | 4.31 | 232.0 | 5.17× | 8.06 | 33.9 | 439 | 0% |
| Qwen3-8B | BASTION (adaptive) | mbpp | 64 | 64 | 0 | 0 | 4.76 | 210.2 | 4.65× | 7.01 | 34.1 | 278 | 0% |
| Qwen3-8B | BASTION (adaptive) | lcb | 64 | 64 | 0 | 0 | 4.76 | 210.2 | 4.67× | 7.33 | 33.6 | 194 | 0% |
| Qwen3-8B | BASTION (adaptive) | mt-bench | 80 | 80 | 0 | 0 | 8.20 | 121.9 | 2.71× | 4.33 | 32.8 | 480 | 1% |
| Qwen3-8B | BASTION (adaptive) | alpaca | 64 | 64 | 0 | 0 | 8.46 | 118.2 | 2.60× | 4.02 | 32.7 | 390 | 0% |
