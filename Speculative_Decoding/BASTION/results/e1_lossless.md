# Lossless check (T=0): speculative outputs vs AR

| Model | Method | Dataset | compared | exact | exact rate | first-div. pos (min / median / max) | prefix agreement (all / diverged) | no AR | prompt≠ |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| Qwen3-4B | BASTION (adaptive) | gsm8k | 64 | 31 | 48.4% | 5 / 90 / 339 | 0.673 / 0.366 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | math500 | 64 | 11 | 17.2% | 3 / 176 / 1639 | 0.414 / 0.292 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | aime25 | 30 | 0 | 0.0% | 3 / 230 / 1409 | 0.207 / 0.207 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | humaneval | 64 | 20 | 31.2% | 2 / 33 / 464 | 0.467 / 0.224 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | mbpp | 64 | 25 | 39.1% | 5 / 91 / 723 | 0.635 / 0.402 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | lcb | 64 | 36 | 56.2% | 1 / 52 / 650 | 0.703 / 0.322 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | mt-bench | 80 | 20 | 25.0% | 3 / 80 / 694 | 0.466 / 0.288 | 0 | 0 |
| Qwen3-4B | BASTION (adaptive) | alpaca | 64 | 22 | 34.4% | 5 / 53 / 399 | 0.503 / 0.243 | 0 | 0 |
| Qwen3-4B | DFlash | gsm8k | 64 | 28 | 43.8% | 3 / 62 / 281 | 0.609 / 0.305 | 0 | 0 |
| Qwen3-4B | DFlash | math500 | 64 | 12 | 18.8% | 4 / 178 / 1652 | 0.413 / 0.278 | 0 | 0 |
| Qwen3-4B | DFlash | aime25 | 30 | 1 | 3.3% | 3 / 216 / 1373 | 0.217 / 0.190 | 0 | 0 |
| Qwen3-4B | DFlash | humaneval | 64 | 22 | 34.4% | 3 / 61 / 492 | 0.531 / 0.286 | 0 | 0 |
| Qwen3-4B | DFlash | mbpp | 64 | 30 | 46.9% | 8 / 70 / 723 | 0.673 / 0.384 | 0 | 0 |
| Qwen3-4B | DFlash | lcb | 64 | 36 | 56.2% | 5 / 82 / 641 | 0.727 / 0.375 | 0 | 0 |
| Qwen3-4B | DFlash | mt-bench | 80 | 22 | 27.5% | 3 / 78 / 287 | 0.466 / 0.263 | 0 | 0 |
| Qwen3-4B | DFlash | alpaca | 64 | 21 | 32.8% | 5 / 63 / 311 | 0.510 / 0.271 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | gsm8k | 64 | 21 | 32.8% | 2 / 61 / 507 | 0.592 / 0.393 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | math500 | 64 | 13 | 20.3% | 3 / 123 / 779 | 0.377 / 0.218 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | aime25 | 30 | 1 | 3.3% | 3 / 141 / 948 | 0.158 / 0.129 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | humaneval | 64 | 11 | 17.2% | 2 / 61 / 449 | 0.392 / 0.266 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | mbpp | 64 | 25 | 39.1% | 11 / 121 / 515 | 0.691 / 0.493 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | lcb | 64 | 42 | 65.6% | 4 / 92 / 267 | 0.793 / 0.398 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | mt-bench | 80 | 21 | 26.2% | 1 / 56 / 570 | 0.457 / 0.264 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | alpaca | 64 | 19 | 29.7% | 2 / 89 / 380 | 0.509 / 0.302 | 0 | 0 |
| Qwen3-8B | DFlash | gsm8k | 64 | 23 | 35.9% | 3 / 99 / 497 | 0.604 / 0.382 | 0 | 0 |
| Qwen3-8B | DFlash | math500 | 64 | 14 | 21.9% | 3 / 107 / 819 | 0.354 / 0.173 | 0 | 0 |
| Qwen3-8B | DFlash | aime25 | 30 | 0 | 0.0% | 3 / 176 / 948 | 0.158 / 0.158 | 0 | 0 |
| Qwen3-8B | DFlash | humaneval | 64 | 7 | 10.9% | 3 / 59 / 584 | 0.355 / 0.276 | 0 | 0 |
| Qwen3-8B | DFlash | mbpp | 64 | 19 | 29.7% | 5 / 98 / 343 | 0.583 / 0.407 | 0 | 0 |
| Qwen3-8B | DFlash | lcb | 64 | 42 | 65.6% | 6 / 80 / 195 | 0.784 / 0.373 | 0 | 0 |
| Qwen3-8B | DFlash | mt-bench | 80 | 22 | 27.5% | 1 / 79 / 721 | 0.486 / 0.291 | 0 | 0 |
| Qwen3-8B | DFlash | alpaca | 64 | 22 | 34.4% | 3 / 90 / 498 | 0.547 / 0.310 | 0 | 0 |
| Qwen3-4B | DFlash | ALL | 494 | 172 | 34.8% | 3 / 96 / 1652 | 0.537 / 0.290 | – | – |
| Qwen3-4B | BASTION (adaptive) | ALL | 494 | 165 | 33.4% | 1 / 93 / 1639 | 0.528 / 0.291 | – | – |
| Qwen3-8B | DFlash | ALL | 494 | 149 | 30.2% | 1 / 90 / 948 | 0.506 / 0.293 | – | – |
| Qwen3-8B | BASTION (adaptive) | ALL | 494 | 153 | 31.0% | 1 / 90 / 948 | 0.518 / 0.302 | – | – |

Exact = generated token ids identical to the AR output for the same prompt (T=0). First-divergence position is a 0-based index into the generated tokens.
