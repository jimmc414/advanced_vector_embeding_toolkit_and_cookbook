from __future__ import annotations

import numpy as np


def solve_linear_map(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """Solve ``W`` in ``X_src @ W ≈ X_tgt`` via least squares."""
    if X_src.shape != X_tgt.shape:
        raise ValueError("source and target matrices must have matching shape")
    if X_src.ndim != 2:
        raise ValueError("inputs must be 2-D")
    W, *_ = np.linalg.lstsq(X_src.astype(np.float64), X_tgt.astype(np.float64), rcond=None)
    return W.astype(np.float32)


def align_vectors(vecs: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply learned alignment matrix ``W`` to ``vecs``."""
    if vecs.shape[-1] != W.shape[0]:
        raise ValueError("vector dimension and transform shape mismatch")
    return (vecs.astype(np.float32) @ W.astype(np.float32)).astype(np.float32)


def alignment_error(X_src: np.ndarray, X_tgt: np.ndarray, W: np.ndarray) -> float:
    """Return relative Frobenius error of the alignment."""
    mapped = align_vectors(X_src, W)
    num = np.linalg.norm(mapped - X_tgt)
    den = np.linalg.norm(X_tgt) + 1e-12
    return float(num / den)
