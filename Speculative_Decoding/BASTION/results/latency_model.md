# Verification-latency model accuracy (E5)

| GPU | Model | points | RMSE uncal (ms) | RMSE cal (ms) | reduction | RMSE uncal c∈{64,256,1024} | RMSE cal c∈{64,256,1024} | LOCO cal RMSE (mean over held-out c) | in-situ RMSE cal (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b4500 | qwen3-4b | 192 | 20.46 | 3.82 | 81.35% | 19.28 | 3.28 | 4.13 | 1.68 |
| b4500 | qwen3-8b | 192 | 29.92 | 5.08 | 83.02% | 28.74 | 4.34 | 5.31 | 1.91 |

Paper (A100, App. H.4): 29.2→2.3 ms (Qwen3-4B), 30.6→3.1 ms (Qwen3-8B), 26.4→3.5 ms (Llama-3.1-8B). Calibrated RMSE here is in-sample unless stated (LOCO column).

Note: GPU 'b4500' is not in upstream's GPU table; its spec {'display_name': 'NVIDIA RTX PRO 4500 Blackwell', 'peak_throughput_bf16': 202000000000000.0, 'memory_bandwidth': 896000000000.0} was registered at runtime (tools/bastion_local.py). The uncalibrated roofline RMSE depends directly on these assumed peak-throughput/bandwidth values; the calibrated fit absorbs a constant scale error.
