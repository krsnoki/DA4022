"""Jacobian-guided noise injection (Pandey et al., ICML 2026 AdaptFM workshop, arXiv:2608.20988).

Independent re-implementation; no official code was released.  See README.md.
"""

from .controller import JACOBIAN_MODES, VALID_MODES, JacobianNoiseConfig, JacobianNoiseController
from .hf_attention import (
    ATTN_IMPL_NAME,
    attach_jacobian_noise,
    detach_jacobian_noise,
    find_attention_modules,
    jacobian_noise_attention_forward,
    register_attention,
    use_jacobian_noise_attention,
)
from .jacobian import softmax_jacobian_explicit, softmax_jacobian_fro, softmax_jacobian_fro_sq

__all__ = [
    "ATTN_IMPL_NAME",
    "JACOBIAN_MODES",
    "VALID_MODES",
    "JacobianNoiseConfig",
    "JacobianNoiseController",
    "attach_jacobian_noise",
    "detach_jacobian_noise",
    "find_attention_modules",
    "jacobian_noise_attention_forward",
    "register_attention",
    "softmax_jacobian_explicit",
    "softmax_jacobian_fro",
    "softmax_jacobian_fro_sq",
    "use_jacobian_noise_attention",
]
