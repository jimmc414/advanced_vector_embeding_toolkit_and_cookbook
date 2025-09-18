from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..utils import l2n


@dataclass
class StreamRecord:
    vector: np.ndarray
    doc_id: str
    timestamp: float


class StreamingIndex:
    """Simple in-memory ANN index with TTL-style eviction."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._records: List[StreamRecord] = []

    def add(self, vectors: np.ndarray, ids: Sequence[str], timestamps: Sequence[float]) -> None:
        if vectors.shape[0] != len(ids) or len(ids) != len(timestamps):
            raise ValueError("vectors, ids, and timestamps must align")
        V = l2n(np.asarray(vectors, dtype=np.float32), axis=1)
        for vec, doc_id, ts in zip(V, ids, timestamps):
            self._records.append(StreamRecord(vec, str(doc_id), float(ts)))

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._records:
            return []
        q = l2n(np.asarray(query, dtype=np.float32), axis=None)
        sims = [float(rec.vector @ q) for rec in self._records]
        order = np.argsort(sims)[::-1][:k]
        return [(self._records[i].doc_id, sims[i]) for i in order]

    def prune_expired(self, current_ts: float, ttl_seconds: float) -> None:
        threshold = float(current_ts) - float(ttl_seconds)
        self._records = [rec for rec in self._records if rec.timestamp >= threshold]

    @property
    def size(self) -> int:
        return len(self._records)
