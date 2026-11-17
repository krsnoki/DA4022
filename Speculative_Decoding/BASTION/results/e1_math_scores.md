# Boxed-answer accuracy (quality sanity check)

| Model | Dataset | Method | n | correct | accuracy | no box | hit max_new_tokens |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | gsm8k | AR | 64 | 56 | 87.5% | 0 | 0 |
| Qwen3-4B | gsm8k | BASTION (adaptive) | 64 | 57 | 89.1% | 0 | 0 |
| Qwen3-4B | gsm8k | DFlash | 64 | 56 | 87.5% | 0 | 0 |
| Qwen3-4B | math500 | AR | 64 | 48 | 75.0% | 6 | 6 |
| Qwen3-4B | math500 | BASTION (adaptive) | 64 | 46 | 71.9% | 8 | 8 |
| Qwen3-4B | math500 | DFlash | 64 | 47 | 73.4% | 6 | 6 |
| Qwen3-4B | aime25 | AR | 30 | 5 | 16.7% | 15 | 15 |
| Qwen3-4B | aime25 | BASTION (adaptive) | 30 | 4 | 13.3% | 17 | 18 |
| Qwen3-4B | aime25 | DFlash | 30 | 4 | 13.3% | 19 | 19 |
| Qwen3-8B | gsm8k | AR | 64 | 59 | 92.2% | 0 | 0 |
| Qwen3-8B | gsm8k | BASTION (adaptive) | 64 | 58 | 90.6% | 0 | 0 |
| Qwen3-8B | gsm8k | DFlash | 64 | 59 | 92.2% | 0 | 0 |
| Qwen3-8B | math500 | AR | 64 | 47 | 73.4% | 8 | 8 |
| Qwen3-8B | math500 | BASTION (adaptive) | 64 | 48 | 75.0% | 7 | 7 |
| Qwen3-8B | math500 | DFlash | 64 | 48 | 75.0% | 6 | 6 |
| Qwen3-8B | aime25 | AR | 30 | 5 | 16.7% | 16 | 17 |
| Qwen3-8B | aime25 | BASTION (adaptive) | 30 | 5 | 16.7% | 16 | 17 |
| Qwen3-8B | aime25 | DFlash | 30 | 5 | 16.7% | 17 | 17 |
