from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..utils import l2n, set_determinism


class DummyEncoder:
    """Deterministic hash 3-gram encoder for queries/documents."""

    def __init__(self, d: int = 32, seed: int = 42):
        set_determinism(seed)
        self.d = int(d)

    def _encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.d, dtype=np.float32)
        t = f"##{text.lower()}##"
        for i in range(len(t) - 2):
            tri = t[i : i + 3]
            h = hash(tri) % self.d
            v[h] += 1.0
        return l2n(v, axis=None)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string (legacy API)."""

        return self._encode(text)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of queries into L2-normalized vectors."""

        return self._encode_many(texts)

    def encode_document(self, text: str) -> np.ndarray:
        """Encode a single document string."""

        return self._encode(text)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of documents into L2-normalized vectors."""

        return self._encode_many(texts)

    def _encode_many(self, texts: Iterable[str]) -> np.ndarray:
        vecs = [self._encode(t) for t in texts]
        if not vecs:
            return np.zeros((0, self.d), dtype=np.float32)
        return np.stack(vecs, axis=0).astype(np.float32, copy=False)
