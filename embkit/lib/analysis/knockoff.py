from __future__ import annotations

from typing import Tuple

import numpy as np

from ..utils import l2n

__all__ = ["knockoff_scores", "knockoff_adjust"]


def knockoff_adjust(doc_vec: np.ndarray, attr_vec: np.ndarray, remove: bool = True, strength: float = 1.0) -> np.ndarray:
    """Return a document vector with the attribute direction removed or emphasized."""
    d = np.asarray(doc_vec, dtype=np.float32)
    attr = l2n(np.asarray(attr_vec, dtype=np.float32), axis=None)
    coeff = float(d @ attr)
    delta = strength * coeff * attr
    return (d - delta if remove else d + delta).astype(np.float32)


def knockoff_scores(query_vec: np.ndarray, doc_vecs: np.ndarray, attr_vec: np.ndarray, remove: bool = True, strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute original vs. knockoff-adjusted cosine scores for a batch of documents."""
    q = l2n(np.asarray(query_vec, dtype=np.float32), axis=None)
    D = np.asarray(doc_vecs, dtype=np.float32)
    if D.ndim == 1:
        D = D.reshape(1, -1)
    Dn = l2n(D, axis=1)
    base = (Dn @ q).astype(np.float32)
    attr = l2n(np.asarray(attr_vec, dtype=np.float32), axis=None)
    attr_proj = (Dn @ attr).astype(np.float32)
    adjusted = base - float(strength) * attr_proj if remove else base + float(strength) * attr_proj
    return base, adjusted
