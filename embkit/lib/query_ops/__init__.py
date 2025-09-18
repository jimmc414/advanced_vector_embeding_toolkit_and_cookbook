from __future__ import annotations
from typing import List, Sequence, Tuple
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


def directional_search(q: np.ndarray, v_dir: np.ndarray, D: np.ndarray, alpha: float = 0.5) -> List[int]:
    """Return row indices sorted by cosine to ``q + alpha * v_dir``."""
    ranked = directional(q, v_dir, D, alpha)
    return [idx for idx, _ in ranked]

def contrastive(q: np.ndarray, v_neg: np.ndarray, D: np.ndarray, lam: float) -> List[Tuple[int, float]]:
    lam = float(lam)
    s = _cos_many(q, D) - lam * _cos_many(v_neg, D)
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order]


def contrastive_score(q: np.ndarray, v_neg: np.ndarray, d: np.ndarray, lam: float = 1.0) -> float:
    """Return contrastive score for a single document vector ``d``."""
    qn = l2n(q.astype(np.float32), axis=None)
    nn = l2n(v_neg.astype(np.float32), axis=None)
    dn = l2n(d.astype(np.float32), axis=None)
    return float(dn @ qn - float(lam) * (dn @ nn))

def compose_and(qs: Sequence[np.ndarray]) -> np.ndarray:
    if len(qs) == 0: raise ValueError("compose_and needs at least one vector")
    v = np.zeros_like(qs[0], dtype=np.float32)
    for q in qs: v += q.astype(np.float32)
    return l2n(v, axis=None)

def compose_or(scores: Sequence[np.ndarray]) -> np.ndarray:
    if len(scores) == 0: raise ValueError("compose_or needs at least one score array")
    out = scores[0].astype(np.float32)
    for s in scores[1:]:
        out = np.maximum(out, s.astype(np.float32))
    return out


def facet_subsearch(query_terms: Sequence[np.ndarray], doc_vectors: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    """Return docs that repeatedly appear across facet sub-searches."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(query_terms) == 0:
        raise ValueError("query_terms must be non-empty")
    Dn = l2n(doc_vectors.astype(np.float32), axis=1)
    if Dn.shape[0] == 0:
        return []
    agg: dict[int, float] = {}
    for term in query_terms:
        sims = Dn @ l2n(term.astype(np.float32), axis=None)
        idx = np.argpartition(-sims, min(top_k, sims.shape[0]-1))[:top_k]
        for i in idx:
            agg[i] = agg.get(int(i), 0.0) + float(sims[i])
    ranked = sorted(agg.items(), key=lambda x: -x[1])
    return ranked[:min(top_k, len(ranked))]

def analogical(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return l2n(b - a + c, axis=None)


def analogical_query(a: str, b: str, c: str, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Convenience helper mirroring the classic analogy ``b - a + c``."""
    return analogical(embeddings[a], embeddings[b], embeddings[c])

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


def mmr_select(q: np.ndarray, D: np.ndarray, k: int = 10, lam: float = 0.7) -> List[int]:
    """Wrapper returning indices picked by maximal marginal relevance."""
    return mmr(q, D, k, lam)

def temporal(scores: np.ndarray, ages_days: np.ndarray, gamma: float) -> np.ndarray:
    return (scores.astype(np.float32) * np.exp(-float(gamma) * ages_days.astype(np.float32))).astype(np.float32)


def temporal_score(score: float, age_days: float, gamma: float = 0.01) -> float:
    """Apply exponential decay to a single score."""
    return float(temporal(np.array([score], dtype=np.float32), np.array([age_days], dtype=np.float32), gamma)[0])

def personalize(q: np.ndarray, u: np.ndarray, D: np.ndarray, beta: float) -> np.ndarray:
    return (_cos_many(q, D) + float(beta) * _cos_many(u, D)).astype(np.float32)


def personalized_score(q: np.ndarray, u: np.ndarray, d: np.ndarray, beta: float = 0.3) -> float:
    """Return personalized score for a single document vector ``d``."""
    return float(personalize(q, u, d[None, :], beta)[0])

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


def late_interaction_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT-style MaxSim scoring for multi-vector interactions."""
    if query_vecs.ndim != 2 or doc_vecs.ndim != 2:
        raise ValueError("expected 2D arrays for query and document vectors")
    Q = l2n(query_vecs.astype(np.float32), axis=1)
    Dn = l2n(doc_vecs.astype(np.float32), axis=1)
    score = 0.0
    for q in Q:
        sims = Dn @ q
        score += float(np.max(sims))
    return float(score)


def subspace_similarity(q: np.ndarray, d: np.ndarray, mask: np.ndarray) -> float:
    """Cosine similarity restricted to an attribute mask."""
    if mask.shape != q.shape:
        raise ValueError("mask must match vector shape")
    q_sub = q.astype(np.float32) * mask.astype(np.float32)
    d_sub = d.astype(np.float32) * mask.astype(np.float32)
    if not np.any(q_sub) or not np.any(d_sub):
        return 0.0
    return float(l2n(q_sub, axis=None) @ l2n(d_sub, axis=None))


def hybrid_score_mix(bm25_scores: Sequence[float], dense_scores: Sequence[float], weight: float = 0.5) -> List[float]:
    """Fuse aligned sparse and dense scores using linear interpolation."""
    if len(bm25_scores) != len(dense_scores):
        raise ValueError("score arrays must align")
    bm25 = np.array(bm25_scores, dtype=np.float32)
    dense = np.array(dense_scores, dtype=np.float32)
    if bm25.size == 0:
        return []
    bm25_norm = bm25 / (bm25.max() + 1e-6)
    dense_norm = dense / (dense.max() + 1e-6)
    w = float(weight)
    fused = w * dense_norm + (1.0 - w) * bm25_norm
    return fused.astype(np.float32).tolist()
