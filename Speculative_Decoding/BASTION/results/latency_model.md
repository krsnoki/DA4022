# Verification-latency model accuracy (E5)

| GPU | Model | points | RMSE uncal (ms) | RMSE cal (ms) | reduction | RMSE uncal c∈{64,256,1024} | RMSE cal c∈{64,256,1024} | LOCO cal RMSE (mean over held-out c) | in-situ RMSE cal (ms), all runs pooled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b4500 | qwen3-4b | 192 | 20.46 | 3.82 | 81.35% | 19.28 | 3.28 | 4.13 | 1.68 |
| b4500 | qwen3-8b | 192 | 29.92 | 5.08 | 83.02% | 28.74 | 4.34 | 5.31 | 6.25 |

Paper (A100, App. H.4): 29.2→2.3 ms (Qwen3-4B), 30.6→3.1 ms (Qwen3-8B), 26.4→3.5 ms (Llama-3.1-8B). Calibrated RMSE here is in-sample unless stated (LOCO column).

In-situ check per run: calibrated prediction vs verification latency measured at every BASTION step (tree mask, real contexts). Mean measured / predicted latency (ms) per realized tree-size bin (bin b holds sizes b to 2b-1).

| Model | Run | steps | RMSE uncal (ms) | RMSE cal (ms) | mean bias cal (ms) | measured / predicted by tree-size bin |
|---|---|---:|---:|---:|---:|---|
| qwen3-4b | e1_qwen3-4b | 36048 | 8.68 | 1.68 | -0.56 | 32: 18.9 / 18.2 (n=5840), 64: 19.0 / 18.5 (n=30203), 128: 22.8 / 23.0 (n=5) |
| qwen3-8b | e1_qwen3-8b | 37311 | 10.40 | 1.91 | -0.94 | 32: 28.9 / 28.0 (n=35943), 64: 31.6 / 31.4 (n=1362), 128: 39.3 / 41.4 (n=4), 256: 48.7 / 52.4 (n=2) |
| qwen3-8b | sweep_qwen3-8b | 87312 | 30.52 | 7.36 | -1.84 | 32: 29.3 / 28.0 (n=32452), 64: 32.2 / 31.4 (n=14504), 128: 36.2 / 38.1 (n=10687), 256: 44.4 / 51.6 (n=10158), 512: 81.7 / 79.2 (n=9962), 1024: 153.8 / 135.7 (n=9549) |

Note: GPU 'b4500' is not in upstream's GPU table; its spec {'display_name': 'NVIDIA RTX PRO 4500 Blackwell', 'peak_throughput_bf16': 202000000000000.0, 'memory_bandwidth': 896000000000.0} was registered at runtime (tools/bastion_local.py). The uncalibrated roofline RMSE depends directly on these assumed peak-throughput/bandwidth values; the calibrated fit absorbs a constant scale error.
