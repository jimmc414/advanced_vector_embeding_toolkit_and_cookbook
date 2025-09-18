from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

__all__ = ["mine_hard_negatives", "triplet_margin"]


def mine_hard_negatives(ranked: Sequence[str], positives: Iterable[str], limit: int = 1) -> List[str]:
    """Return the top ``limit`` ranked items that are not in ``positives``."""
    positive_set = set(positives)
    hard: List[str] = []
    for doc_id in ranked:
        if doc_id in positive_set:
            continue
        hard.append(str(doc_id))
        if len(hard) >= limit:
            break
    return hard


def triplet_margin(query_vec: np.ndarray, pos_vec: np.ndarray, neg_vec: np.ndarray, margin: float = 0.2) -> float:
    """Standard triplet margin loss used for hard-negative training."""
    q = np.asarray(query_vec, dtype=np.float32)
    p = np.asarray(pos_vec, dtype=np.float32)
    n = np.asarray(neg_vec, dtype=np.float32)
    pos_sim = float(np.dot(q, p))
    neg_sim = float(np.dot(q, n))
    loss = margin + neg_sim - pos_sim
    return float(max(0.0, loss))
