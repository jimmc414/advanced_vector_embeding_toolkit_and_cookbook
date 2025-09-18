from __future__ import annotations

from typing import Sequence

import numpy as np


def average_embedding(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    """Return L2-normalized mean embedding for ``vectors`` or ``None`` if empty."""
    if len(vectors) == 0:
        return None
    stacked = np.vstack([v.astype(np.float32) for v in vectors])
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return np.zeros_like(mean, dtype=np.float32)
    return (mean / norm).astype(np.float32)


def summarize_sentences(texts: Sequence[str], max_sentences: int = 2) -> str:
    """Extract the top-``max_sentences`` informative sentences across ``texts``."""
    sentences: list[str] = []
    for doc in texts:
        for part in doc.replace("\n", " ").split('.'):
            sent = part.strip()
            if len(sent.split()) >= 3:
                sentences.append(sent)
    sentences.sort(key=len, reverse=True)
    chosen = sentences[:max_sentences]
    return '. '.join(chosen).strip()
