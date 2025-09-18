import numpy as np
import pytest

from embkit.lib.query_ops import (
    analogical_query,
    cone_filter,
    contrastive_score,
    directional,
    directional_search,
    facet_subsearch,
    hybrid_score_mix,
    late_interaction_score,
    mahalanobis_diag,
    mmr_select,
    personalize,
    personalized_score,
    polytope_filter,
    subspace_similarity,
    temporal,
    temporal_score,
)
from embkit.lib.utils import l2n


def test_directional_shifts_rank():
    q = l2n(np.array([1.0, 0.0], dtype=np.float32), axis=None)
    vdir = l2n(np.array([0.0, 1.0], dtype=np.float32), axis=None)
    D = l2n(np.array([[0.9, 0.1], [0.4, 0.9]], dtype=np.float32), axis=1)
    base = directional(q, np.zeros_like(vdir), D, alpha=0.0)
    with_dir = directional(q, vdir, D, alpha=0.8)
    assert base[0][0] == 0
    assert with_dir[0][0] == 1  # direction pushes doc1 up
    reordered = directional_search(q, vdir, D, alpha=0.8)
    assert reordered[0] == 1


def test_cone_excludes_wide_angle():
    q = np.array([1.0, 0.0], dtype=np.float32)
    D = l2n(np.array([[0.95, 0.1], [0.5, 0.5], [-0.9, 0.0]], dtype=np.float32), axis=1)
    keep = cone_filter(q, D, cos_min=0.8)
    assert keep == [0]  # only near q


def test_polytope_intersection():
    D = l2n(np.array([[0.9, 0.9], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32), axis=1)
    c1 = (np.array([1.0, 0.0], dtype=np.float32), 0.7)
    c2 = (np.array([0.0, 1.0], dtype=np.float32), 0.7)
    keep = polytope_filter(D, [c1, c2])
    assert keep == [0]


def test_contrastive_penalizes_negative_concept():
    q = np.array([0.7, 0.7], dtype=np.float32)
    neg = np.array([0.0, 1.0], dtype=np.float32)
    doc_ok = np.array([0.6, 0.0], dtype=np.float32)
    doc_bad = np.array([0.1, 0.9], dtype=np.float32)
    s_ok = contrastive_score(q, neg, doc_ok, lam=0.5)
    s_bad = contrastive_score(q, neg, doc_bad, lam=0.5)
    assert s_ok > s_bad


def test_facet_subsearch_prioritizes_multi_facet_docs():
    docs = np.array([[0.9, 0.8], [0.95, 0.1], [0.1, 0.95], [0.5, 0.5]], dtype=np.float32)
    ml = np.array([1.0, 0.0], dtype=np.float32)
    health = np.array([0.0, 1.0], dtype=np.float32)
    ranked = facet_subsearch([ml, health], docs, top_k=3)
    top_ids = [idx for idx, _ in ranked[:2]]
    assert 0 in top_ids
    assert 3 in top_ids


def test_analogical_query_returns_expected_vector():
    emb = {
        "king": np.array([1.0, 1.0], dtype=np.float32),
        "queen": np.array([-1.0, 1.0], dtype=np.float32),
        "man": np.array([1.0, 0.0], dtype=np.float32),
        "woman": np.array([-1.0, 0.0], dtype=np.float32),
    }
    vec = analogical_query("king", "queen", "man", emb)
    sim_to_queen = float(vec @ l2n(emb["queen"], axis=None))
    sim_to_king = float(vec @ l2n(emb["king"], axis=None))
    assert sim_to_queen > sim_to_king


def test_mmr_select_promotes_diversity():
    rng = np.random.default_rng(0)
    q = l2n(rng.normal(size=5).astype(np.float32), axis=None)
    doc1 = q + 0.1
    doc2 = q + 0.1
    doc3 = q * 0.8
    D = l2n(np.vstack([doc1, doc2, doc3]).astype(np.float32), axis=1)
    selected = mmr_select(q, D, k=2, lam=0.5)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert any(i in selected for i in (0, 1))
    assert 2 in selected


def test_temporal_decay_prefers_recent_docs():
    scores = np.array([0.8, 0.8], dtype=np.float32)
    ages = np.array([10.0, 100.0], dtype=np.float32)
    adjusted = temporal(scores, ages, gamma=0.02)
    assert adjusted[0] > adjusted[1]
    assert temporal_score(0.8, 10.0, gamma=0.02) == pytest.approx(adjusted[0])


def test_personalization_boosts_profile_matches():
    q = np.array([0.0, 1.0], dtype=np.float32)
    user = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    scores = personalize(q, user, docs, beta=0.5)
    assert scores[0] > scores[1]
    assert personalized_score(q, user, docs[0], beta=0.5) == pytest.approx(scores[0])


def test_late_interaction_score_rewards_term_coverage():
    q_tokens = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc1 = np.array([[0.9, 0.1], [0.1, 0.95]], dtype=np.float32)
    doc2 = np.array([[0.88, 0.0], [0.88, 0.0]], dtype=np.float32)
    s1 = late_interaction_score(q_tokens, doc1)
    s2 = late_interaction_score(q_tokens, doc2)
    assert s1 > s2


def test_subspace_similarity_filters_by_attribute():
    mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q = np.array([0.9, 0.2, 0.1], dtype=np.float32)
    doc_match = np.array([0.8, -0.3, 0.0], dtype=np.float32)
    doc_mismatch = np.array([-0.8, 0.4, 0.2], dtype=np.float32)
    assert subspace_similarity(q, doc_match, mask) > subspace_similarity(q, doc_mismatch, mask)


def test_hybrid_score_mix_handles_extreme_weights():
    bm25 = [2.0, 1.5, 0.5]
    dense = [0.2, 0.7, 0.9]
    all_sparse = hybrid_score_mix(bm25, dense, weight=0.0)
    all_dense = hybrid_score_mix(bm25, dense, weight=1.0)
    even = hybrid_score_mix(bm25, dense, weight=0.5)
    assert np.argmax(all_sparse) == 0
    assert np.argmax(all_dense) == 2
    assert len(even) == 3


def test_mahalanobis_diag_requires_nonnegative_weights():
    u = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 1.0], dtype=np.float32)
    w = np.array([1.0, 4.0], dtype=np.float32)
    assert mahalanobis_diag(u, v, w) > 0
    with pytest.raises(ValueError):
        mahalanobis_diag(u, v, np.array([1.0, -1.0], dtype=np.float32))
