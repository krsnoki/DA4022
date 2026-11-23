# E4 budget sweep diagnostics: Qwen3-8B

144 prompt/turn keys present in every configuration (gsm8k 16, math500 16, aime25 16, humaneval 16, mbpp 16, lcb 16, mt-bench 32, alpaca 16); 0 sweep keys skipped. AR, DFlash and 'Adaptive warm (E1 repeat)' are the E1 records of the same prompts. Speedup = mean TPOT(AR) / mean TPOT(config) per dataset; Average = unweighted mean over datasets.

| Dataset | DFlash (E1) | Adaptive warm (E1 repeat) | Fixed-32 | Fixed-64 | Fixed-128 | Fixed-256 | Fixed-512 | Fixed-1024 | Adaptive warm | Adaptive upstream-cold | best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gsm8k | 4.45 | 4.90 | 4.91 | 4.79 | 4.53 | 3.80 | 2.30 | 1.29 | 4.90 | 4.88 | Fixed-32 |
| math500 | 5.87 | 5.93 | 5.85 | 5.69 | 5.35 | 4.57 | 2.76 | 1.56 | 5.95 | 5.84 | Fixed-32 |
| aime25 | 5.01 | 4.86 | 4.95 | 4.80 | 4.58 | 3.85 | 2.35 | 1.35 | 4.86 | 4.84 | Fixed-32 |
| humaneval | 4.58 | 5.17 | 5.12 | 5.05 | 4.84 | 4.08 | 2.44 | 1.36 | 5.17 | 5.17 | Fixed-32 |
| mbpp | 4.13 | 4.48 | 4.50 | 4.53 | 4.21 | 3.54 | 2.19 | 1.24 | 4.48 | 4.47 | Fixed-64 |
| lcb | 4.35 | 4.68 | 4.67 | 4.58 | 4.33 | 3.71 | 2.31 | 1.32 | 4.69 | 4.82 | Fixed-32 |
| mt-bench | 2.55 | 2.98 | 2.96 | 2.89 | 2.80 | 2.43 | 1.45 | 0.82 | 2.99 | 2.93 | Fixed-32 |
| alpaca | 2.63 | 3.11 | 3.11 | 3.08 | 2.95 | 2.59 | 1.55 | 0.87 | 3.12 | 3.16 | Fixed-32 |
| **Average** | **4.197** | **4.517** | **4.509** | **4.428** | **4.198** | **3.571** | **2.168** | **1.226** | **4.521** | **4.514** | Oracle 4.513 |

Ratios of averages with 95% bootstrap intervals (2000 resamples of prompts within dataset; the oracle's per-dataset N is fixed from the full data):

| Ratio | point | 95% interval |
|---|---:|---|
| Adaptive warm / Oracle (best fixed N per dataset) | 1.002 | [0.993, 1.011] |
| Adaptive warm / Fixed-32 | 1.003 | [0.995, 1.011] |
| Adaptive warm / Fixed-64 | 1.021 | [1.011, 1.032] |
| Adaptive warm / Fixed-128 | 1.077 | [1.062, 1.092] |
| Adaptive warm / Fixed-256 | 1.266 | [1.251, 1.282] |
| Adaptive warm / Fixed-512 | 2.085 | [2.053, 2.118] |
| Adaptive warm / Fixed-1024 | 3.687 | [3.636, 3.742] |
| Adaptive warm / Adaptive upstream-cold | 1.001 | [0.986, 1.016] |
| Adaptive warm / DFlash (E1) | 1.077 | [1.055, 1.096] |
| Adaptive warm / Adaptive warm (E1 repeat) | 1.001 | [1.001, 1.001] |
| Adaptive upstream-cold / DFlash (E1) | 1.076 | [1.055, 1.095] |

Realized adaptive tree sizes (pooled over steps):

- Adaptive warm (E1 repeat): mean 33.2 over 11642 steps; 32: 96.5%, 64: 3.4%, 96: 0.1%
- Adaptive warm: mean 33.2 over 11642 steps; 32: 96.5%, 64: 3.4%, 96: 0.1%
- Adaptive upstream-cold: mean 40.0 over 12100 steps; 32: 76.9%, 64: 21.4%, 96: 1.6%, 128: 0.1%, 160: 0.0%
