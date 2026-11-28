"""ORA-Q: operator-role-aware post-training quantization (fake quantization).

Paper: Structural Outlier-Aware Post-Training Quantization for Monocular Depth
Estimation, AdaptFM @ ICML 2026, https://openreview.net/forum?id=oqacjavnry
No upstream code was found; this is a reimplementation from the paper.
"""

from .build import build_quant_model, load_config, op_configs, validate_config
from .diagnose import clip_scores, diagnose, stability_scores
from .qmodules import (ActQuantizer, QuantConv2d, QuantLinear, calibrate, count_by_op,
                       quantize_model, quantizers, set_state)
from .quantizer import fake_quant, fake_quant_weight, from_groups, qparams_from_range, to_groups
