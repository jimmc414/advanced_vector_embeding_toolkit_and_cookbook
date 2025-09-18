from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import l2n

def build_knn(D: np.ndarray, k: int) -> csr_matrix:
    """Build symmetric kNN graph with cosine weights, no self-loops."""
    Dn = l2n(D, axis=1)
    S = (Dn @ Dn.T).astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    k = int(min(max(1, k), n-1))
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = S[np.arange(n)[:, None], idx].ravel()
    # make symmetric by max
    M = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    M = M.maximum(M.T)
    M.setdiag(0.0)
    M.eliminate_zeros()
    return M

def ppr(adj: csr_matrix, seed_scores: np.ndarray, alpha: float, iters: int = 20) -> np.ndarray:
    """Personalized PageRank with power iteration on row-normalized graph."""
    assert adj.shape[0] == adj.shape[1]
    n = adj.shape[0]
    # row-normalize
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-12)
    P = adj.multiply(1.0 / deg[:, None]).tocsr()
    s = seed_scores.astype(np.float32).copy()
    s = np.maximum(s, 0)
    s /= (s.sum() + 1e-12)
    r = s.copy()
    for _ in range(int(iters)):
        r = float(alpha) * s + (1.0 - float(alpha)) * (P.T @ r)
    return r.astype(np.float32)
