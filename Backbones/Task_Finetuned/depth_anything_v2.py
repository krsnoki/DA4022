"""Depth Anything V2 (Small) from Hugging Face transformers.

Registry id: vision.depth-anything-v2-small. The weights are never modified;
quantized variants are deep copies (see Quantization_and_Pruning/.../build.py).
Weights stay in the Hugging Face cache and are never committed.
"""

import platform

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def resolve_revision(model_id=MODEL_ID, revision=None):
    """Exact commit hash of the checkpoint, so results can be pinned."""
    if revision:
        return revision
    try:
        from huggingface_hub import model_info
        return model_info(model_id).sha
    except Exception:
        return "unknown"


def load(device="cuda", model_id=MODEL_ID, revision=None):
    revision = resolve_revision(model_id, revision)
    rev = None if revision == "unknown" else revision
    processor = AutoImageProcessor.from_pretrained(model_id, revision=rev)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, revision=rev)
    model.to(device).eval()
    return model, processor, revision


def preprocess(processor, rgb, device):
    """uint8 HxWx3 numpy image -> pixel_values on device (processor defaults: 518, keep aspect, x14)."""
    return processor(images=rgb, return_tensors="pt")["pixel_values"].to(device)


@torch.no_grad()
def predict_disparity(model, pixel_values, out_hw):
    """Relative inverse depth resized to out_hw = (H, W), as a numpy array."""
    pred = model(pixel_values=pixel_values).predicted_depth          # (1, h, w)
    pred = F.interpolate(pred.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False)
    return pred.squeeze().float().cpu().numpy()


def get_model_metadata(model, revision):
    return {
        "id": "vision.depth-anything-v2-small",
        "hf_path": MODEL_ID,
        "revision": revision,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
