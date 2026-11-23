"""Local launcher for upstream ``bastion.benchmark`` on GPUs missing from upstream's table.

Upstream ``bastion/cost_model.py`` (pinned commit 1989b80) only knows A5000, A6000,
A100, H100, H200 and RTX PRO 6000 Blackwell. ``resolve_gpu_spec`` raises
``ValueError: Unsupported gpu_type: 'NVIDIA RTX PRO 4500 Blackwell'`` on this machine.

This wrapper registers the RTX PRO 4500 Blackwell at runtime (no upstream file is
modified) and then runs the unmodified upstream CLI. Usage, from $ROOT/upstream:

    ../.venv/bin/python ../tools/bastion_local.py --model ... --draft-model ... [same args]

Spec values: 896 GB/s GDDR7 (NVIDIA product page); 1617 FP4 AI TOPS (sparse, PNY
spec table; NVIDIA's page lists no TOPS figure) / 8 -> ~202 TFLOPS dense BF16, the
same /8 ratio that maps the RTX PRO 6000's 4000 FP4 TOPS to upstream's 504 TFLOPS. The roofline fit
(alpha_m*t_mem + beta_m, alpha_c*t_comp + beta_c) absorbs any constant scale
error in these numbers, so they act only as analytic priors.
"""
from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from bastion import cost_model  # noqa: E402

LOCAL_GPU_KEY = "b4500"
cost_model.GPU_SPECS.setdefault(
    LOCAL_GPU_KEY,
    {
        "display_name": "NVIDIA RTX PRO 4500 Blackwell",
        "peak_throughput_bf16": 202 * 1e12,
        "memory_bandwidth": 896 * 1e9,
    },
)
for alias in (
    "b4500",
    "rtxpro4500",
    "nvidiartxpro4500",
    "nvidiartxpro4500blackwell",
):
    cost_model.GPU_ALIASES.setdefault(alias, LOCAL_GPU_KEY)

# Memory fix for 32 GB GPUs (does not change any timed measurement).
# Upstream collect_calibration_data decorates its timed helpers (_warmup_forward,
# _measure_single_point) with @torch.inference_mode(), but the untimed context
# prefill in its main loop runs with autograd enabled, so every prefill keeps its
# activations alive (~9 GB extra for a 1024-token prefill on Qwen3-8B), which OOMs
# a 32 GB card. Running the whole routine under inference_mode only drops that
# autograd bookkeeping; the timed forwards were already in inference mode.
import torch  # noqa: E402

if not getattr(cost_model.collect_calibration_data, "_bastion_local_inference_mode", False):
    _orig_collect = cost_model.collect_calibration_data

    def _collect_calibration_data_no_grad(*args, **kwargs):
        with torch.inference_mode():
            return _orig_collect(*args, **kwargs)

    _collect_calibration_data_no_grad._bastion_local_inference_mode = True
    _collect_calibration_data_no_grad.__wrapped__ = _orig_collect
    cost_model.collect_calibration_data = _collect_calibration_data_no_grad

from bastion import benchmark  # noqa: E402

if __name__ == "__main__":
    benchmark.main()
