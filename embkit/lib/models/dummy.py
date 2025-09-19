from __future__ import annotations

from typing import Iterable, Sequence

import hashlib
import numpy as np

from ..utils import l2n, set_determinism


class DummyEncoder:
    """Deterministic hash 3-gram encoder for queries/documents.

    The encoder uses a seed-keyed BLAKE2b digest to map trigrams into
    buckets, ensuring reproducible bucket selection across processes without
    relying on ``PYTHONHASHSEED``.
    """

    def __init__(self, d: int = 32, seed: int = 42):
        set_determinism(seed)
        self.d = int(d)
        self.seed = int(seed)
        self._hash_key = str(self.seed).encode("utf-8")

    def _encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.d, dtype=np.float32)
        t = f"##{text.lower()}##"
        for i in range(len(t) - 2):
            tri = t[i : i + 3]
            digest = hashlib.blake2b(
                tri.encode("utf-8"), key=self._hash_key, digest_size=8
            ).digest()
            h = int.from_bytes(digest, "big") % self.d
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
