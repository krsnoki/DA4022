# Jacobian-guided noise injection: local results

Accuracy-type metrics are shown in %, perplexity raw. Mean ± std over seeds.
Δ = method − no-noise fine-tuning baseline (positive Δ is better for accuracy, negative for perplexity).
All quantization is simulated (fake-quant).

## multimodal.siglip_base_p16_224 — imagenet_zeroshot_top1

| train method | FP16 | rtn-W6A6 | rtn-W5A5 | rtn-W4A4 |
|---|---:|---:|---:|---:|
| pretrained(no-finetune) | 73.40 (Δ +1.12) | 72.34 (Δ +1.08) | 67.78 (Δ +2.38) | 5.98 (Δ -0.92) |
| finetune(no-noise) | 72.28 ± 0.00 | 71.26 ± 0.00 | 65.40 ± 0.00 | 6.90 ± 0.00 |
| finetune+uniform_matched_noise | 72.21 ± 0.01 (Δ -0.07) | 71.38 ± 0.06 (Δ +0.12) | 65.95 ± 0.04 (Δ +0.55) | 7.06 ± 0.28 (Δ +0.16) |
| finetune+jacobian_noise(mode=jacobian,alpha=0.5) | 72.19 ± 0.07 (Δ -0.09) | 71.37 ± 0.35 (Δ +0.11) | 66.24 ± 0.25 (Δ +0.84) | 6.73 ± 0.04 (Δ -0.17) |
| finetune+jacobian_noise(mode=jacobian,alpha=4) | 72.52 ± 0.03 (Δ +0.24) | 71.18 ± 0.14 (Δ -0.08) | 66.50 ± 0.59 (Δ +1.10) | 7.33 ± 0.61 (Δ +0.43) |
