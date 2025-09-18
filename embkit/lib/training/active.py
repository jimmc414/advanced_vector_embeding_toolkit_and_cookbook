from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = ["margin_uncertainty", "select_uncertain_queries"]


def margin_uncertainty(scores: Sequence[float]) -> float:
    """Return top-1 minus top-2 margin; large value => confident."""
    arr = np.sort(np.asarray(scores, dtype=np.float32))[::-1]
    if arr.size <= 1:
        return float("inf")
    return float(arr[0] - arr[1])


def select_uncertain_queries(score_map: Dict[str, Sequence[float]], threshold: float, limit: int | None = None) -> List[str]:
    """Return query IDs whose margin is below ``threshold``."""
    selected: List[str] = []
    for qid, scores in score_map.items():
        margin = margin_uncertainty(scores)
        if margin < threshold:
            selected.append(qid)
            if limit is not None and len(selected) >= limit:
                break
    return selected
