"""Shared test helpers (stand-ins for Member 2B's functions while they are TODO)."""

import torch


def keep_tables(normalized, importance, codebooks, selectors, selection_size, *, chunk_elements):
    """Update step that changes nothing; lets the fit pipeline run without lloyd.py."""
    return codebooks.clone()


def round_tables(codebooks, codebook_dtype=torch.bfloat16):
    """Minimal finalisation stand-in: plain dtype rounding."""
    return codebooks.to(codebook_dtype)


def brute_force_group_errors(normalized, importance, codebooks, selection_size):
    """Reference E_c[o, s] = sum_{k in s} I[k] * min_j (u - C[c, j])^2, float64, no chunking."""
    u = normalized.double()
    errs = []
    for c in range(2):
        d = (u.unsqueeze(-1) - codebooks[c].double()).square().min(dim=-1).values
        errs.append((d * importance.double()).reshape(u.shape[0], -1, selection_size).sum(-1))
    return torch.stack(errs)  # [2, out, in // S]
