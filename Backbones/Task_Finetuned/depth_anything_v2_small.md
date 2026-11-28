# Backbone model card: Depth Anything V2 Small

- **Registry ID:** `vision.depth-anything-v2-small`
- **Modality:** vision
- **Architecture:** DINOv2 ViT-S/14 encoder (12 blocks, hidden 384, 6 heads, MLP 1536) + DPT neck and head
- **Pretraining regime:** mixed (self-supervised DINOv2 parent, then depth training; see the model card)
- **Training objective/data:** relative monocular depth. The Hugging Face model card reports 595K synthetic labeled images and 62M+ real unlabeled images. The [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414) abstract says student models learn through pseudo-labeled real images. Exact training split, full recipe for this checkpoint and overlap with NYUv2: unknown (not verified).
- **Weights URI and exact revision:** https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf, revision `5426e4f0f36572d16453bbda7a8389317b1bef99`
- **License/usage terms:** apache-2.0 (model card at this revision)
- **Tokenizer/processor revision:** no tokenizer. Image processor from the same repository and revision (`preprocessor_config.json`, `DPTImageProcessor`).
- **Parameter count / weight size:** 24785089, F32. `model.safetensors` is 99173660 bytes (Hugging Face file listing at this revision).
- **Parent backbone (if fine-tuned):** `vision.dinov2-vits14`. The exact parent checkpoint revision used in training is unknown. A parent-only baseline was not run, because the DINOv2 backbone has no depth head.
- **Input/output contract:** `pixel_values` (1, 3, H, W), H and W multiples of 14 -> `predicted_depth` (1, H, W), relative inverse depth (larger = closer, unknown scale and shift)

## Loading

```python
from Backbones.Task_Finetuned import depth_anything_v2 as dav2
model, processor, revision = dav2.load("cuda")   # or "cpu"; the pinned revision is the default
```

## Preprocessing and outputs

The Hugging Face image processor of the same revision (target size 518, keep aspect, multiple of 14, bicubic resample, rescale by 1/255, ImageNet normalisation, no padding). For NYUv2 480x640 frames the output is resized back to 480x640 with bilinear interpolation before scoring.

## Smoke test

`python -m pytest -q Quantization_and_Pruning/tests` (no download). It uses a tiny random Depth Anything model; for a (1, 3, 84, 112) input, `predicted_depth` is (1, 84, 112).

Real checkpoint, checked on CPU by `experiments/ora_q/measure_size.py`: a 480x640 NYUv2 frame becomes (1, 3, 518, 686) and `predicted_depth` is (1, 518, 686).

## Evaluation

The unmodified checkpoint is the baseline: run `experiments/ora_q/run_ora_q.py --methods fp32` (command in the paper card).

## Caveats

Requires transformers 5.x (layer names `q_proj`, `k_proj`, `v_proj`, `o_proj`). Weights are not committed.
