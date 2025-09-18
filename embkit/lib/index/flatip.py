from __future__ import annotations
from typing import List, Tuple
import os, json
import numpy as np
import faiss  # type: ignore
from ..utils import l2n

class FlatIP:
    """Exact cosine via inner product on L2-normalized vectors."""
    def __init__(self, d: int):
        self.d = int(d)
        self.index = faiss.IndexFlatIP(self.d)
        self._ids: List[str] = []
        self._D: np.ndarray | None = None

    def add(self, D: np.ndarray, ids: List[str]) -> None:
        if D.dtype != np.float32: D = D.astype(np.float32)
        Dn = l2n(D, axis=1)
        if len(ids) != Dn.shape[0]:
            raise ValueError("ids length mismatch")
        self.index.add(Dn)
        self._ids = list(ids)
        self._D = Dn

    def search(self, q: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        qn = l2n(q.astype(np.float32), axis=None)[None, :]
        sims, I = self.index.search(qn, min(k, len(self._ids)))
        idx = I[0]
        idx = idx[idx >= 0]
        scores = sims[0][:len(idx)].astype(np.float32)
        return [self._ids[i] for i in idx], scores

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "flatip.faiss"))
        if self._D is not None:
            np.save(os.path.join(dirpath, "D_norm.npy"), self._D)
        with open(os.path.join(dirpath, "ids.txt"), "w", encoding="utf-8") as f:
            for s in self._ids: f.write(s+"\n")
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"d": self.d, "kind": "flatip"}, f)

    @classmethod
    def load(cls, dirpath: str) -> "FlatIP":
        with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["d"])
        obj.index = faiss.read_index(os.path.join(dirpath, "flatip.faiss"))
        Dp = os.path.join(dirpath, "D_norm.npy")
        if os.path.exists(Dp):
            obj._D = np.load(Dp).astype(np.float32)
        with open(os.path.join(dirpath, "ids.txt"), "r", encoding="utf-8") as f:
            obj._ids = [ln.strip() for ln in f if ln.strip()]
        return obj
