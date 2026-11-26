# Local experiment report

Status: **partial, small-scale methodological reproduction**. These are measured local results, not estimates and not paper-reported numbers.

## Protocol

- Device: CPU, one PyTorch thread; RTX 3050 not used because PyTorch is CPU-only.
- Seed: 4022.
- Teacher: one-layer tiny routed LM, 8 SwiGLU experts, top-2 routing.
- Data: deterministic synthetic modular sequences; 96 train, 48 calibration, 48 evaluation samples; length 24.
- Teacher fitting: 80 AdamW steps.
- Conversion: `K = k = 2`, pure pruning, uniform scaling.
- Distillation: 50 steps, forward KL, gradient accumulation 2.
- Timing: batch 8 × length 24, 3 warm-ups, median of 30 repetitions; model forward only.

## Results

| Model / scoring | Selected experts | Parameters | Relative size | Pre-KD PPL | Post-KD PPL | Pre-KD KL | Post-KD KL | Median latency | Tokens/s | Peak CPU memory (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Routed teacher | dynamic top-2 | 16,768 | 100% | — | 6.167 | — | — | 1.146 ms | 167,466 | 187.77 |
| Dense SF | 5, 0 | 7,296 | 43.51% | 14.713 | 7.730 | 0.611 | 0.122 | 0.179 ms | 1,074,126 | 186.80 |
| Dense ACP | 6, 3 | 7,296 | 43.51% | 13.823 | 7.379 | 0.565 | 0.095 | 0.167 ms | 1,152,115 | 186.60 |
| Dense DO-ACP | 6, 7 | 7,296 | 43.51% | 13.716 | 7.378 | 0.583 | 0.094 | 0.161 ms | 1,191,437 | 186.97 |

Teacher training loss fell from 4.357 to 1.544. Distillation loss fell from 0.649 to 0.129 for SF, 0.574 to 0.099 for ACP, and 0.578 to 0.093 for DO-ACP. Students use 56.49% fewer parameters than the teacher. DO-ACP selected a different second expert than ACP and finished with the lowest measured KL and perplexity, although the differences are too small and the experiment too limited for a general claim.

The fastest local dense forward was roughly 7.1× faster than the routed teacher for this tiny CPU workload. The teacher evaluates only the selected expert-token assignments, but its unoptimized Python dispatch and gather/scatter overhead dominate this tiny benchmark. Calibration evaluates all experts separately and is not included in inference timing. The identical student architectures' latency differences are measurement noise. Do not extrapolate these ratios to optimized GPU LLM inference.

## Verification

- `31 passed` in 5.88 s, including dtype/config validation, sparse dispatch equivalence, checkpoint resume equivalence and CSV export checks.
- The all-in-one reproduction completed successfully.
- The four standalone stages completed successfully before the final equation-alignment audit; the final all-in-one measurements above are authoritative.
- A fairness defect found by the staged run—later students inheriting frozen teacher flags—was corrected and covered by a regression test before final measurements.

## Limitations

- Synthetic tokens do not measure natural-language modeling or reproduce WikiText-2.
- No pretrained checkpoint, FineWeb-Edu distillation, or downstream benchmark was run.
- Only round-robin grouping is implemented; the paper's clustering alternatives are outside the feasible core scope.
- Peak CPU process working set includes Python/PyTorch startup, model loading, warmup and inference. Differences near a few MiB should not be interpreted as precise tensor-allocation savings.
- Latency lacks p95 and multi-seed confidence intervals; the experiment is a deterministic smoke benchmark.
- The generic adapter boundary is present, but Qwen/DeepSeek/Phi checkpoint conversion remains unimplemented and unverified.
- Only one seed and a very small expert set were evaluated.

Machine-readable measurements and checkpoints are under `results/synthetic/`.
