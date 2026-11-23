# Lossless check (T=0): speculative outputs vs AR

| Model | Method | Dataset | compared | exact | exact rate | first-div. pos (min / median / max) | prefix agreement (all / diverged) | no AR | prompt≠ |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| Qwen3-8B | BASTION (adaptive) | gsm8k | 16 | 5 | 31.2% | 9 / 72 / 314 | 0.598 / 0.416 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | math500 | 16 | 1 | 6.2% | 24 / 143 / 463 | 0.268 / 0.219 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | aime25 | 16 | 0 | 0.0% | 3 / 120 / 948 | 0.116 / 0.116 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | humaneval | 16 | 5 | 31.2% | 9 / 73 / 303 | 0.495 / 0.265 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | mbpp | 16 | 6 | 37.5% | 34 / 126 / 370 | 0.667 / 0.467 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | lcb | 16 | 11 | 68.8% | 32 / 131 / 267 | 0.792 / 0.334 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | mt-bench | 32 | 13 | 40.6% | 1 / 102 / 570 | 0.581 / 0.294 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive) | alpaca | 16 | 7 | 43.8% | 18 / 90 / 300 | 0.637 / 0.355 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | gsm8k | 16 | 4 | 25.0% | 9 / 60 / 280 | 0.524 / 0.365 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | math500 | 16 | 4 | 25.0% | 24 / 153 / 671 | 0.412 / 0.216 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | aime25 | 16 | 0 | 0.0% | 3 / 228 / 657 | 0.150 / 0.150 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | humaneval | 16 | 1 | 6.2% | 3 / 115 / 586 | 0.390 / 0.349 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | mbpp | 16 | 6 | 37.5% | 22 / 136 / 832 | 0.729 / 0.567 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | lcb | 16 | 10 | 62.5% | 61 / 94 / 297 | 0.727 / 0.272 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | mt-bench | 32 | 15 | 46.9% | 15 / 85 / 388 | 0.616 / 0.278 | 0 | 0 |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | alpaca | 16 | 8 | 50.0% | 47 / 108 / 466 | 0.699 / 0.399 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | gsm8k | 16 | 7 | 43.8% | 3 / 72 / 178 | 0.610 / 0.306 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | math500 | 16 | 2 | 12.5% | 24 / 140 / 542 | 0.301 / 0.201 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | aime25 | 16 | 0 | 0.0% | 3 / 230 / 452 | 0.139 / 0.139 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | humaneval | 16 | 2 | 12.5% | 9 / 70 / 584 | 0.368 / 0.278 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | mbpp | 16 | 3 | 18.8% | 41 / 149 / 370 | 0.617 / 0.529 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | lcb | 16 | 10 | 62.5% | 32 / 106 / 483 | 0.767 / 0.379 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | mt-bench | 32 | 14 | 43.8% | 1 / 85 / 479 | 0.593 / 0.277 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | alpaca | 16 | 8 | 50.0% | 18 / 101 / 212 | 0.645 / 0.291 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | gsm8k | 16 | 6 | 37.5% | 9 / 53 / 279 | 0.549 / 0.279 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | math500 | 16 | 2 | 12.5% | 18 / 131 / 542 | 0.292 / 0.191 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | aime25 | 16 | 0 | 0.0% | 3 / 166 / 627 | 0.140 / 0.140 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | humaneval | 16 | 3 | 18.8% | 3 / 73 / 584 | 0.445 / 0.317 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | mbpp | 16 | 3 | 18.8% | 2 / 98 / 370 | 0.595 / 0.502 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | lcb | 16 | 9 | 56.2% | 32 / 83 / 267 | 0.696 / 0.305 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | mt-bench | 32 | 11 | 34.4% | 12 / 62 / 568 | 0.588 / 0.372 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-64 | alpaca | 16 | 9 | 56.2% | 7 / 47 / 122 | 0.650 / 0.201 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | gsm8k | 16 | 4 | 25.0% | 23 / 92 / 261 | 0.588 / 0.451 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | math500 | 16 | 1 | 6.2% | 3 / 128 / 542 | 0.266 / 0.217 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | aime25 | 16 | 0 | 0.0% | 3 / 230 / 453 | 0.146 / 0.146 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | humaneval | 16 | 1 | 6.2% | 3 / 76 / 584 | 0.342 / 0.299 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | mbpp | 16 | 4 | 25.0% | 2 / 112 / 247 | 0.589 / 0.451 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | lcb | 16 | 8 | 50.0% | 22 / 109 / 331 | 0.652 / 0.303 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | mt-bench | 32 | 12 | 37.5% | 1 / 74 / 481 | 0.578 / 0.325 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-128 | alpaca | 16 | 8 | 50.0% | 7 / 68 / 212 | 0.624 / 0.248 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | gsm8k | 16 | 4 | 25.0% | 4 / 80 / 314 | 0.578 / 0.438 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | math500 | 16 | 2 | 12.5% | 3 / 140 / 463 | 0.298 / 0.198 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | aime25 | 16 | 0 | 0.0% | 3 / 280 / 948 | 0.192 / 0.192 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | humaneval | 16 | 2 | 12.5% | 9 / 60 / 584 | 0.413 / 0.330 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | mbpp | 16 | 4 | 25.0% | 2 / 97 / 370 | 0.592 / 0.456 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | lcb | 16 | 9 | 56.2% | 3 / 80 / 267 | 0.684 / 0.277 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | mt-bench | 32 | 14 | 43.8% | 12 / 91 / 506 | 0.621 / 0.327 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-256 | alpaca | 16 | 8 | 50.0% | 18 / 108 / 331 | 0.676 / 0.353 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | gsm8k | 16 | 4 | 25.0% | 23 / 60 / 316 | 0.537 / 0.383 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | math500 | 16 | 2 | 12.5% | 3 / 84 / 542 | 0.231 / 0.121 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | aime25 | 16 | 0 | 0.0% | 3 / 267 / 1048 | 0.184 / 0.184 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | humaneval | 16 | 1 | 6.2% | 3 / 62 / 584 | 0.348 / 0.304 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | mbpp | 16 | 1 | 6.2% | 2 / 64 / 370 | 0.379 / 0.337 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | lcb | 16 | 11 | 68.8% | 58 / 83 / 267 | 0.765 / 0.248 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | mt-bench | 32 | 14 | 43.8% | 9 / 117 / 481 | 0.644 / 0.367 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-512 | alpaca | 16 | 8 | 50.0% | 7 / 120 / 384 | 0.710 / 0.420 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | gsm8k | 16 | 5 | 31.2% | 9 / 106 / 322 | 0.620 / 0.447 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | math500 | 16 | 1 | 6.2% | 3 / 153 / 927 | 0.301 / 0.255 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | aime25 | 16 | 0 | 0.0% | 3 / 160 / 948 | 0.132 / 0.132 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | humaneval | 16 | 4 | 25.0% | 6 / 117 / 337 | 0.517 / 0.356 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | mbpp | 16 | 2 | 12.5% | 2 / 154 / 284 | 0.561 / 0.498 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | lcb | 16 | 11 | 68.8% | 32 / 98 / 297 | 0.782 / 0.302 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | mt-bench | 32 | 13 | 40.6% | 1 / 62 / 388 | 0.545 / 0.233 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-1024 | alpaca | 16 | 9 | 56.2% | 7 / 100 / 275 | 0.670 / 0.246 | 0 | 0 |
| Qwen3-8B | BASTION-Fixed-32 | ALL | 144 | 46 | 31.9% | 1 / 104 / 584 | 0.515 / 0.287 | – | – |
| Qwen3-8B | BASTION-Fixed-64 | ALL | 144 | 43 | 29.9% | 2 / 85 / 627 | 0.505 / 0.294 | – | – |
| Qwen3-8B | BASTION-Fixed-128 | ALL | 144 | 38 | 26.4% | 1 / 116 / 584 | 0.485 / 0.300 | – | – |
| Qwen3-8B | BASTION-Fixed-256 | ALL | 144 | 43 | 29.9% | 2 / 111 / 948 | 0.520 / 0.315 | – | – |
| Qwen3-8B | BASTION-Fixed-512 | ALL | 144 | 41 | 28.5% | 2 / 90 / 1048 | 0.494 / 0.292 | – | – |
| Qwen3-8B | BASTION-Fixed-1024 | ALL | 144 | 45 | 31.2% | 1 / 105 / 948 | 0.519 / 0.301 | – | – |
| Qwen3-8B | BASTION (adaptive) | ALL | 144 | 48 | 33.3% | 1 / 100 / 948 | 0.526 / 0.289 | – | – |
| Qwen3-8B | BASTION (adaptive, upstream cold profile) | ALL | 144 | 48 | 33.3% | 3 / 106 / 832 | 0.540 / 0.311 | – | – |

Exact = generated token ids identical to the AR output for the same prompt (T=0). First-divergence position is a 0-based index into the generated tokens.
