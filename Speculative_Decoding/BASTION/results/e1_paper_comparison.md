# E1: local results vs paper Table 1 (A100, T=0)

## Qwen3-4B

| Benchmark | DFlash local | DFlash paper | BASTION local | BASTION paper | BASTION/DFlash local | BASTION/DFlash paper | τ DFlash local / paper | τ BASTION local / paper | mean tree (local) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 4.62× | 5.23× | 5.26× | 7.44× | 1.139 | 1.423 | 6.23 / 6.50 | 8.26 / 9.07 | 59.5 |
| math500 | 5.57× | 6.22× | 5.88× | 8.57× | 1.056 | 1.378 | 8.19 / 8.12 | 10.01 / 10.60 | 56.2 |
| aime25 | 4.49× | 6.16× | 4.88× | 8.23× | 1.088 | 1.336 | 7.02 / 7.48 | 8.96 / 9.70 | 59.2 |
| humaneval | 4.52× | 5.34× | 5.15× | 7.91× | 1.140 | 1.481 | 6.52 / 6.66 | 8.44 / 9.38 | 57.7 |
| mbpp | 4.14× | 5.09× | 4.93× | 7.41× | 1.190 | 1.456 | 5.50 / 6.31 | 7.67 / 8.99 | 61.3 |
| lcb | 3.97× | 5.71× | 4.53× | 7.81× | 1.140 | 1.368 | 5.89 / 7.23 | 8.12 / 9.48 | 57.1 |
| mt-bench | 2.26× | 2.74× | 2.79× | 4.53× | 1.235 | 1.653 | 3.25 / 4.63 | 4.79 / 6.58 | 62.9 |
| alpaca | 2.07× | 2.12× | 2.60× | 3.53× | 1.261 | 1.665 | 2.96 / 2.90 | 4.36 / 4.64 | 63.1 |
| Average | 3.95× | 4.83× | 4.50× | 6.93× | 1.139 | 1.435 | 5.69 / 6.23 | 7.58 / 8.56 | 59.6 |

Mean tree in the Average row = unweighted mean of per-dataset means; pooled over all 36048 steps: 60.1 (median 64, max 192). Distribution of realized sizes: 32: 5840 (16.2%), 64: 28814 (79.9%), 96: 1389 (3.9%), 128: 4 (0.0%), 192: 1 (0.0%).

Implied cycle cost τ/speedup in AR-step units, mean over datasets (approximate): DFlash local 1.44, paper 1.32; BASTION local 1.69, paper 1.25.

95% bootstrap intervals (prompts resampled within dataset, 2000 resamples): DFlash average [3.88, 4.04], BASTION average [4.43, 4.59], ratio of averages [1.126, 1.151].

| Benchmark | BASTION/DFlash 95% interval | prompts where BASTION TPOT < DFlash TPOT |
|---|---|---:|
| gsm8k | [1.117, 1.161] | 60/64 |
| math500 | [1.032, 1.080] | 45/64 |
| aime25 | [1.039, 1.147] | 24/30 |
| humaneval | [1.112, 1.166] | 58/64 |
| mbpp | [1.161, 1.220] | 61/64 |
| lcb | [1.099, 1.184] | 54/64 |
| mt-bench | [1.207, 1.261] | 72/80 |
| alpaca | [1.234, 1.286] | 63/64 |

Resumed run: mean TPOT (ms) per session for datasets that span sessions:

- humaneval ar: session 0: 13.89 ms (n=23), session 1: 13.81 ms (n=41)
- humaneval bastion: session 0: 2.68 ms (n=22), session 1: 2.69 ms (n=42)
- humaneval dflash: session 0: 3.05 ms (n=22), session 1: 3.07 ms (n=42)

## Qwen3-8B

| Benchmark | DFlash local | DFlash paper | BASTION local | BASTION paper | BASTION/DFlash local | BASTION/DFlash paper | τ DFlash local / paper | τ BASTION local / paper | mean tree (local) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 4.83× | 4.96× | 5.17× | 6.92× | 1.072 | 1.395 | 6.28 / 6.51 | 7.85 / 8.85 | 34.1 |
| math500 | 5.71× | 6.06× | 5.78× | 8.02× | 1.013 | 1.323 | 8.04 / 8.25 | 9.31 / 10.53 | 33.5 |
| aime25 | 4.97× | 5.91× | 4.98× | 7.69× | 1.001 | 1.301 | 7.15 / 7.20 | 8.34 / 9.46 | 33.1 |
| humaneval | 4.76× | 5.25× | 5.17× | 7.64× | 1.086 | 1.455 | 6.42 / 6.64 | 8.06 / 9.44 | 33.9 |
| mbpp | 4.27× | 4.91× | 4.65× | 7.10× | 1.088 | 1.446 | 5.55 / 6.04 | 7.01 / 8.65 | 34.1 |
| lcb | 4.39× | 5.90× | 4.67× | 7.64× | 1.064 | 1.295 | 6.62 / 7.64 | 7.33 / 9.91 | 33.6 |
| mt-bench | 2.28× | 2.81× | 2.71× | 4.31× | 1.185 | 1.534 | 3.14 / 4.46 | 4.33 / 6.28 | 32.8 |
| alpaca | 2.14× | 2.28× | 2.60× | 3.54× | 1.211 | 1.553 | 2.90 / 3.15 | 4.02 / 4.68 | 32.7 |
| Average | 4.17× | 4.76× | 4.47× | 6.61× | 1.071 | 1.389 | 5.76 / 6.24 | 7.03 / 8.48 | 33.5 |

Mean tree in the Average row = unweighted mean of per-dataset means; pooled over all 37311 steps: 33.2 (median 32, max 256). Distribution of realized sizes: 32: 35943 (96.3%), 64: 1302 (3.5%), 96: 60 (0.2%), 128: 1 (0.0%), 160: 2 (0.0%), 192: 1 (0.0%), 256: 2 (0.0%).

Implied cycle cost τ/speedup in AR-step units, mean over datasets (approximate): DFlash local 1.38, paper 1.33; BASTION local 1.57, paper 1.29.

95% bootstrap intervals (prompts resampled within dataset, 2000 resamples): DFlash average [4.08, 4.27], BASTION average [4.39, 4.55], ratio of averages [1.058, 1.083].

| Benchmark | BASTION/DFlash 95% interval | prompts where BASTION TPOT < DFlash TPOT |
|---|---|---:|
| gsm8k | [1.050, 1.094] | 50/64 |
| math500 | [0.992, 1.034] | 31/64 |
| aime25 | [0.948, 1.058] | 13/30 |
| humaneval | [1.058, 1.113] | 54/64 |
| mbpp | [1.058, 1.116] | 52/64 |
| lcb | [1.017, 1.107] | 42/64 |
| mt-bench | [1.160, 1.210] | 70/80 |
| alpaca | [1.187, 1.238] | 62/64 |
