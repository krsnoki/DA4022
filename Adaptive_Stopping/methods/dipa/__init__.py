"""DIPA: Difficulty-Informed Probabilistic Allocation of Test-Time Compute."""

from Adaptive_Stopping.methods.dipa.allocator import (
    BudgetAllocator,
    DIPABudgetAllocator,
)
from Adaptive_Stopping.methods.dipa.proxies import (
    GEN_PROXIES,
    INPUT_PROXIES,
    compute_gen_difficulty,
    compute_input_difficulty,
    to_difficulty_score,
)
from Adaptive_Stopping.methods.dipa.types import (
    AllocationConfig,
    AllocationReport,
    ArmState,
    GenerationAttempt,
    ProblemInstance,
    StoppingPolicy,
)
from Adaptive_Stopping.methods.dipa.verifier import (
    extract_boxed_answer,
    normalize_math_answer,
    verify_math_answer,
)

__all__ = [
    "AllocationConfig",
    "AllocationReport",
    "ArmState",
    "BudgetAllocator",
    "DIPABudgetAllocator",
    "GEN_PROXIES",
    "GenerationAttempt",
    "INPUT_PROXIES",
    "ProblemInstance",
    "StoppingPolicy",
    "compute_gen_difficulty",
    "compute_input_difficulty",
    "extract_boxed_answer",
    "normalize_math_answer",
    "to_difficulty_score",
    "verify_math_answer",
]
