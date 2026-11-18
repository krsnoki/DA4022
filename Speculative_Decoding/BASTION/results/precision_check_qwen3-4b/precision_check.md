# Precision check: Qwen/Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16

8 gsm8k prompts (orig_idx [1101, 120, 474, 991, 192, 1020, 350, 480]), max_new_tokens=256, T=0, attention sdpa, device cuda:0.
In `X vs Y`, Y is the reference. Margin = top-1 minus top-2 logit of Y's target model, teacher-forced on prompt + common prefix, at the first divergence (same dtype as Y). TF argmax = ref: the teacher-forced argmax equals Y's token there.

| comparison | exact match | first divergences | median margin | max margin | TF argmax = ref |
|---|---|---|---|---|---|
| dflash-bf16 vs ar-bf16 | 4/8 (50%) | [33, 150, 21, 23] | 0.125 | 0.25 | 2/4 |
| bastion-bf16 vs ar-bf16 | 4/8 (50%) | [144, 64, 199, 23] | 0.188 | 0.25 | 3/4 |
| dflash-fp32 vs ar-fp32 | 8/8 (100%) | - | - | - | 0/0 |
| bastion-fp32 vs ar-fp32 | 8/8 (100%) | - | - | - | 0/0 |
| ar-bf16 vs ar-fp32 | 2/8 (25%) | [21, 33, 150, 0, 101, 44] | 0.083 | 0.175 | 6/6 |

Per-divergence margins (logits):

- dflash-bf16 vs ar-bf16 #120: pos 33, ref tok 525 vs 1184, top1-top2 0, ref-other 0
- dflash-bf16 vs ar-bf16 #474: pos 150, ref tok 220 vs 3070, top1-top2 0.25, ref-other -0.25
- dflash-bf16 vs ar-bf16 #1020: pos 21, ref tok 2303 vs 198, top1-top2 0.25, ref-other 0.25
- dflash-bf16 vs ar-bf16 #480: pos 23, ref tok 624 vs 198, top1-top2 0, ref-other 0
- bastion-bf16 vs ar-bf16 #120: pos 144, ref tok 1447 vs 320, top1-top2 0.25, ref-other 0.25
- bastion-bf16 vs ar-bf16 #474: pos 64, ref tok 1654 vs 12, top1-top2 0.25, ref-other 0.25
- bastion-bf16 vs ar-bf16 #991: pos 199, ref tok 8596 vs 3070, top1-top2 0.125, ref-other -0.125
- bastion-bf16 vs ar-bf16 #480: pos 23, ref tok 624 vs 198, top1-top2 0, ref-other 0
- ar-bf16 vs ar-fp32 #1101: pos 21, ref tok 95518 vs 66963, top1-top2 0.1285, ref-other 0.1285
- ar-bf16 vs ar-fp32 #120: pos 33, ref tok 1184 vs 525, top1-top2 0.01181, ref-other 0.01181
- ar-bf16 vs ar-fp32 #474: pos 150, ref tok 3070 vs 220, top1-top2 0.1754, ref-other 0.1754
- ar-bf16 vs ar-fp32 #192: pos 0, ref tok 1654 vs 10061, top1-top2 0.03755, ref-other 0.03755
- ar-bf16 vs ar-fp32 #1020: pos 101, ref tok 3757 vs 566, top1-top2 0.1351, ref-other 0.1351
- ar-bf16 vs ar-fp32 #480: pos 44, ref tok 25839 vs 3070, top1-top2 0.03513, ref-other 0.03513
