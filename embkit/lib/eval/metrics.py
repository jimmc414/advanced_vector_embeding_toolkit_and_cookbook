from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Iterable

def recall_at_k(rels: List[int], ranked: List[int], k: int) -> float:
    if k <= 0: return 0.0
    topk = set(ranked[:k])
    hit = sum(1 for r in rels if r in topk)
    return float(hit / max(1, len(rels)))

def dcg_at_k(gains: List[int], k: int) -> float:
    g = np.array(gains[:k], dtype=np.float32)
    if g.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * discounts))

def ndcg_at_k(ranked_rels: List[int], k: int = 10) -> float:
    dcg = dcg_at_k(ranked_rels, k)
    ideal = dcg_at_k(sorted(ranked_rels, reverse=True), k)
    return float(dcg / ideal) if ideal > 0 else 0.0

def mrr(ranked: List[int], rel_set: set[int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in rel_set: return 1.0 / i
    return 0.0

def diversity_cosine(D_top: np.ndarray) -> float:
    if D_top.shape[0] < 2: return 0.0
    Dn = D_top / np.linalg.norm(D_top, axis=1, keepdims=True).clip(min=1e-12)
    S = Dn @ Dn.T
    n = D_top.shape[0]
    upper = S[np.triu_indices(n, k=1)]
    return float(1.0 - float(np.mean(upper))) if upper.size else 0.0

def time_weighted_ndcg(rels: List[int], ages_days: List[float], k: int, gamma: float) -> float:
    # gains times exp(-gamma * age)
    gains = np.array(rels[:k], dtype=np.float32) * np.exp(-gamma * np.array(ages_days[:k], dtype=np.float32))
    ideal = np.sort(gains)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, min(k, gains.size) + 2))
    dcg = float(np.sum(gains[:discounts.size] * discounts))
    idcg = float(np.sum(ideal[:discounts.size] * discounts))
    return float(dcg / idcg) if idcg > 0 else 0.0

def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = y_true.astype(np.float32).ravel()
    pr = p.astype(np.float32).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (pr >= bins[i]) & (pr < bins[i+1] if i+1 < len(bins)-1 else pr <= bins[i+1])
        if m.sum() == 0: continue
        conf = float(pr[m].mean())
        acc = float(y[m].mean())
        ece += (m.sum() / len(pr)) * abs(conf - acc)
    return float(ece)

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = y_true.astype(np.float32).ravel()
    pr = p.astype(np.float32).ravel()
    return float(np.mean((pr - y) ** 2))

def compute_all(labels, rankings, times, ages) -> Dict[str, float]:
    # Minimal aggregator for demo; users can extend.
    # labels: dict[qid] -> set(relevant doc ids)
    # rankings: dict[qid] -> list[doc ids]
    # times: dict[qid] -> float latency ms
    # ages: dict[qid] -> list[age_days per doc]
    out: Dict[str, float] = {}
    rec10 = []
    nd10 = []
    mrrs = []
    lat = []
    for qid, ranked in rankings.items():
        rels = labels.get(qid, set())
        rec10.append(recall_at_k(list(rels), ranked, 10))
        rbin = [1 if d in rels else 0 for d in ranked]
        nd10.append(ndcg_at_k(rbin, 10))
        mrrs.append(mrr(ranked, rels))
        lat.append(float(times.get(qid, 0.0)))
    out["Recall@10"] = float(np.mean(rec10)) if rec10 else 0.0
    out["nDCG@10"] = float(np.mean(nd10)) if nd10 else 0.0
    out["MRR"] = float(np.mean(mrrs)) if mrrs else 0.0
    out["Latency.p50.ms"] = float(np.median(lat)) if lat else 0.0
    out["Latency.p95.ms"] = float(np.percentile(lat, 95)) if lat else 0.0
    return out
