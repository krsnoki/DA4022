# Backbone model card: Depth Anything V2 Small

- **Registry ID:** `vision.depth-anything-v2-small`
- **Modality:** vision
- **Architecture:** DINOv2 ViT-S/14 encoder (12 blocks, hidden 384, 6 heads, MLP 1536) + DPT neck and head
- **Pretraining regime:** mixed (self-supervised DINOv2 parent, then depth training; see the model card)
- **Weights URI and exact revision:** https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf, revision `5426e4f0f36572d16453bbda7a8389317b1bef99`
- **License/usage terms:** apache-2.0 (model card at this revision)
- **Parameter count:** 24785089, F32 (Hugging Face safetensors metadata at this revision). Weight file size: not recorded.
- **Parent backbone:** DINOv2 ViT-S/14
- **Input/output contract:** `pixel_values` (1, 3, H, W), H and W multiples of 14 -> `predicted_depth` (1, H, W), relative inverse depth (larger = closer, unknown scale and shift)

## Loading

```python
from Backbones.Task_Finetuned import depth_anything_v2 as dav2
model, processor, revision = dav2.load("cuda")
```

## Preprocessing and outputs

The Hugging Face image processor of the same revision (resize to 518 on the short side, keep aspect, multiple of 14, ImageNet normalisation). For NYUv2 480x640 frames the output is resized back to 480x640 with bilinear interpolation before scoring.

## Smoke test

`pytest -q Quantization_and_Pruning/tests/test_ora_q.py` builds a tiny random model with the same architecture (no download).

## Caveats

Requires transformers 5.x (layer names `q_proj`, `k_proj`, `v_proj`, `o_proj`). Weights are not committed.
