from __future__ import annotations
from typing import List, Tuple
import numpy as np
from ..utils import l2n

def _cos_many(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    qn = l2n(q.astype(np.float32), axis=None)
    Dn = l2n(D, axis=1)
    return (Dn @ qn).astype(np.float32)

def directional(q: np.ndarray, v_dir: np.ndarray, D: np.ndarray, alpha: float) -> List[Tuple[int, float]]:
    q_shift = l2n(q + float(alpha) * v_dir, axis=None)
    s = _cos_many(q_shift, D)
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order]

def contrastive(q: np.ndarray, v_neg: np.ndarray, D: np.ndarray, lam: float) -> List[Tuple[int, float]]:
    lam = float(lam)
    s = _cos_many(q, D) - lam * _cos_many(v_neg, D)
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order]

def compose_and(qs: List[np.ndarray]) -> np.ndarray:
    if len(qs) == 0: raise ValueError("compose_and needs at least one vector")
    v = np.zeros_like(qs[0], dtype=np.float32)
    for q in qs: v += q.astype(np.float32)
    return l2n(v, axis=None)

def compose_or(scores: List[np.ndarray]) -> np.ndarray:
    if len(scores) == 0: raise ValueError("compose_or needs at least one score array")
    out = scores[0].astype(np.float32)
    for s in scores[1:]:
        out = np.maximum(out, s.astype(np.float32))
    return out

def analogical(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return l2n(b - a + c, axis=None)

def mmr(q: np.ndarray, D: np.ndarray, k: int, lam: float) -> List[int]:
    k = int(k); lam = float(lam)
    sims_q = _cos_many(q, D)
    chosen: List[int] = []
    pool = np.arange(D.shape[0], dtype=int)
    for _ in range(min(k, len(pool))):
        rem = np.setdiff1d(pool, np.array(chosen, dtype=int), assume_unique=True)
        if len(chosen) == 0:
            i = int(rem[np.argmax(sims_q[rem])])
        else:
            Dn = l2n(D, axis=1)
            S = Dn[chosen]
            div = (Dn[rem] @ S.T).max(axis=1)
            score = lam * sims_q[rem] - (1 - lam) * div
            i = int(rem[np.argmax(score)])
        chosen.append(i)
    return chosen

def temporal(scores: np.ndarray, ages_days: np.ndarray, gamma: float) -> np.ndarray:
    return (scores.astype(np.float32) * np.exp(-float(gamma) * ages_days.astype(np.float32))).astype(np.float32)

def personalize(q: np.ndarray, u: np.ndarray, D: np.ndarray, beta: float) -> np.ndarray:
    return (_cos_many(q, D) + float(beta) * _cos_many(u, D)).astype(np.float32)

def cone_filter(q: np.ndarray, D: np.ndarray, cos_min: float) -> List[int]:
    s = _cos_many(q, D)
    return [int(i) for i in np.where(s >= float(cos_min))[0].tolist()]

def polytope_filter(D: np.ndarray, constraints: List[Tuple[np.ndarray, float]]) -> List[int]:
    Dn = l2n(D, axis=1)
    keep = np.ones(Dn.shape[0], dtype=bool)
    for v, thr in constraints:
        v = l2n(v.astype(np.float32), axis=None)
        keep &= (Dn @ v) >= float(thr)
    return [int(i) for i in np.where(keep)[0].tolist()]

def mahalanobis_diag(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    diff = (u.astype(np.float32) - v.astype(np.float32))
    if np.any(w < 0): raise ValueError("weights must be nonnegative")
    return float(np.sqrt(np.sum((diff * np.sqrt(w.astype(np.float32)))**2, dtype=np.float32)))
