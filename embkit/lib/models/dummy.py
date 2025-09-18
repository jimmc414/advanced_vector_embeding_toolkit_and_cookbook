from __future__ import annotations
import numpy as np
from ..utils import l2n, set_determinism

class DummyEncoder:
    """Deterministic hash 3-gram encoder for queries."""
    def __init__(self, d: int = 32, seed: int = 42):
        set_determinism(seed)
        self.d = int(d)
    def encode_query(self, text: str) -> np.ndarray:
        v = np.zeros(self.d, dtype=np.float32)
        t = f"##{text.lower()}##"
        for i in range(len(t)-2):
            tri = t[i:i+3]
            h = hash(tri) % self.d
            v[h] += 1.0
        return l2n(v, axis=None)
