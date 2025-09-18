from __future__ import annotations
import numpy as np
from ..utils import l2n

def apply_repellors(scores: np.ndarray, D: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Subtract λ * max_b cos(D, b) from scores."""
    if B.size == 0: return scores.astype(np.float32)
    Dn = l2n(D, axis=1)
    Bn = l2n(B, axis=1)
    pen = (Dn @ Bn.T).max(axis=1).astype(np.float32)
    return (scores.astype(np.float32) - float(lam) * pen).astype(np.float32)
