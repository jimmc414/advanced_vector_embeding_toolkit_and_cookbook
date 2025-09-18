from __future__ import annotations

import numpy as np

from ..utils import l2n

__all__ = ["remove_direction", "remove_directions", "nullspace_project"]


def _prepare_vectors(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("expected 1D or 2D array of vectors")
    return arr


def remove_direction(vec: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of ``vec`` that lies along ``direction``."""
    v = np.asarray(vec, dtype=np.float32)
    d = l2n(np.asarray(direction, dtype=np.float32), axis=None)
    proj = float(v @ d)
    return (v - proj * d).astype(np.float32)


def remove_directions(vecs: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Project ``vecs`` onto the nullspace orthogonal to ``directions``."""
    V = _prepare_vectors(vecs)
    if V.size == 0:
        return V.copy()
    D = _prepare_vectors(directions)
    if D.size == 0:
        return V.copy()
    B = l2n(D, axis=1)
    projection = V - (V @ B.T) @ B
    return projection.astype(np.float32)


def nullspace_project(vecs: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Alias for :func:`remove_directions` to highlight nullspace intent."""
    return remove_directions(vecs, directions)
