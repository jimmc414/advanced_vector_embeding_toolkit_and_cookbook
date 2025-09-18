from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def split_embedding(vec: np.ndarray, dims: Sequence[int]) -> List[np.ndarray]:
    """Split ``vec`` into subspaces defined by ``dims`` lengths."""
    v = np.asarray(vec, dtype=np.float32)
    if sum(dims) != v.shape[0]:
        raise ValueError("dims must sum to vector length")
    parts: List[np.ndarray] = []
    start = 0
    for d in dims:
        end = start + int(d)
        parts.append(v[start:end].copy())
        start = end
    return parts


def merge_embedding(parts: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate subspaces back into a single embedding."""
    arrays = [np.asarray(p, dtype=np.float32) for p in parts]
    if not arrays:
        raise ValueError("parts must be non-empty")
    return np.concatenate(arrays).astype(np.float32)


def swap_subspace(
    base: np.ndarray,
    donor: np.ndarray,
    dims: Sequence[int],
    index: int,
) -> np.ndarray:
    """Return ``base`` with the subspace at ``index`` replaced by ``donor``'s subspace."""
    base_parts = split_embedding(base, dims)
    donor_parts = split_embedding(donor, dims)
    if index < 0 or index >= len(base_parts):
        raise IndexError("index out of range")
    base_parts[index] = donor_parts[index]
    return merge_embedding(base_parts)
