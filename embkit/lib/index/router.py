from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..utils import l2n

CentroidMap = Mapping[str, np.ndarray]
SearchFn = Callable[[np.ndarray, int], Tuple[Sequence[str], Sequence[float]]]


def nearest_centroids(query: np.ndarray, centroids: CentroidMap, top_n: int = 1) -> List[Tuple[str, float]]:
    """Return the top ``top_n`` centroid labels ranked by cosine similarity."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not centroids:
        return []
    qn = l2n(query.astype(np.float32), axis=None)
    sims: List[Tuple[str, float]] = []
    for name, vec in centroids.items():
        vn = l2n(vec.astype(np.float32), axis=None)
        sims.append((name, float(vn @ qn)))
    sims.sort(key=lambda x: -x[1])
    return sims[: min(top_n, len(sims))]


def route_and_search(
    query: np.ndarray,
    centroids: CentroidMap,
    searchers: Mapping[str, SearchFn],
    k: int = 10,
    fanout: int = 1,
) -> List[Tuple[str, str, float]]:
    """Route ``query`` to the best centroid(s) then search matching sub-indices."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = nearest_centroids(query, centroids, top_n=max(1, fanout))
    results: List[Tuple[str, str, float]] = []
    if not ranked:
        return results
    for name, _ in ranked:
        fn = searchers.get(name)
        if fn is None:
            continue
        ids, scores = fn(query, k)
        for doc_id, score in zip(ids, scores):
            results.append((name, str(doc_id), float(score)))
    results.sort(key=lambda x: -x[2])
    return results[: min(k, len(results))]


def merge_fanout(results: Iterable[List[Tuple[str, str, float]]], k: int) -> List[Tuple[str, str, float]]:
    """Combine routed result lists keeping the global top-``k`` by score."""
    if k <= 0:
        raise ValueError("k must be positive")
    merged: List[Tuple[str, str, float]] = []
    for batch in results:
        merged.extend(batch)
    merged.sort(key=lambda x: -x[2])
    return merged[: min(k, len(merged))]
