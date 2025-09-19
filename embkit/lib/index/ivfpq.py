from __future__ import annotations
from typing import List, Tuple
import os, json
import numpy as np
import faiss  # type: ignore
from ..utils import l2n

class IVFPQ:
    """IVF-PQ ANN with exact residual re-rank of top-200."""
    def __init__(self, d: int, nlist: int, m: int, nbits: int, nprobe: int = 8):
        self.d, self.nlist, self.m, self.nbits, self.nprobe = int(d), int(nlist), int(m), int(nbits), int(nprobe)
        quantizer = faiss.IndexFlatIP(self.d)
        self.index = faiss.IndexIVFPQ(quantizer, self.d, self.nlist, self.m, self.nbits)
        self.index.nprobe = self.nprobe
        self._ids: List[str] = []
        self._D: np.ndarray | None = None

    def train_add(self, D: np.ndarray, ids: List[str]) -> None:
        if D.dtype != np.float32: D = D.astype(np.float32)
        Dn = l2n(D, axis=1)
        if len(ids) != Dn.shape[0]:
            raise ValueError("ids length mismatch")
        if Dn.shape[0] == 0:
            return

        if not self.index.is_trained:
            train = Dn
            needed = max(self.nlist, int(self.index.pq.ksub))
            if train.shape[0] < needed:
                reps = int(np.ceil(needed / train.shape[0]))
                train = np.tile(train, (reps, 1))[:needed]
            self.index.train(train)
        self.index.add(Dn)
        if self._D is None:
            self._D = Dn.copy()
        else:
            self._D = np.concatenate([self._D, Dn], axis=0)
        self._ids.extend(ids)

    def search(self, q: np.ndarray, k: int) -> Tuple[List[str], np.ndarray]:
        if self._D is None: return [], np.array([], dtype=np.float32)
        qn = l2n(q.astype(np.float32), axis=None)[None, :]
        ncand = max(200, k * 5, 1)
        sims, I = self.index.search(qn, min(ncand, len(self._ids)))
        cand = I[0]
        cand = cand[cand >= 0]
        if cand.size == 0:
            return [], np.array([], dtype=np.float32)
        sims_exact = (self._D[cand] @ qn[0]).astype(np.float32)
        order = np.argsort(-sims_exact)[:k]
        chosen = cand[order]
        scores = sims_exact[order]
        return [self._ids[i] for i in chosen], scores

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "ivfpq.faiss"))
        if self._D is not None:
            np.save(os.path.join(dirpath, "D_norm.npy"), self._D)
        with open(os.path.join(dirpath, "ids.txt"), "w", encoding="utf-8") as f:
            for s in self._ids: f.write(s+"\n")
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"d": self.d, "kind": "ivfpq", "nlist": self.nlist, "m": self.m, "nbits": self.nbits, "nprobe": self.nprobe},
                f
            )

    @classmethod
    def load(cls, dirpath: str) -> "IVFPQ":
        with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["d"], meta["nlist"], meta["m"], meta["nbits"], meta["nprobe"])
        obj.index = faiss.read_index(os.path.join(dirpath, "ivfpq.faiss"))
        Dp = os.path.join(dirpath, "D_norm.npy")
        if os.path.exists(Dp):
            import numpy as np
            obj._D = np.load(Dp).astype(np.float32)
        with open(os.path.join(dirpath, "ids.txt"), "r", encoding="utf-8") as f:
            obj._ids = [ln.strip() for ln in f if ln.strip()]
        return obj
