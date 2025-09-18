import numpy as np

from embkit.lib.training.active import margin_uncertainty, select_uncertain_queries
from embkit.lib.training.hard_negative import mine_hard_negatives, triplet_margin
from embkit.lib.training.robust import fgsm_perturb, generate_synonym_variants


def test_generate_synonym_variants_produces_paraphrases():
    query = "capital of france"
    synonyms = {"capital": ["chief city"], "france": ["the french republic"]}
    variants = generate_synonym_variants(query, synonyms)
    assert "chief city of france" in variants
    assert any("french republic" in v for v in variants)


def test_fgsm_perturb_returns_normalized_vector():
    emb = np.array([1.0, 0.0], dtype=np.float32)
    grad = np.array([0.5, -0.3], dtype=np.float32)
    perturbed = fgsm_perturb(emb, grad, epsilon=0.1)
    assert np.allclose(np.linalg.norm(perturbed), 1.0, atol=1e-6)


def test_mine_hard_negatives_skips_positive():
    ranked = ["doc_pos", "doc_neg1", "doc_neg2"]
    hard = mine_hard_negatives(ranked, positives=["doc_pos"], limit=2)
    assert hard == ["doc_neg1", "doc_neg2"]


def test_triplet_margin_zero_when_pos_stronger():
    q = np.array([1.0, 0.0], dtype=np.float32)
    pos = np.array([1.0, 0.0], dtype=np.float32)
    neg = np.array([0.0, 1.0], dtype=np.float32)
    assert triplet_margin(q, pos, neg, margin=0.2) == 0.0


def test_select_uncertain_queries_uses_margin():
    scores = {"q1": [0.9, 0.85, 0.2], "q2": [0.7, 0.1, 0.05]}
    selected = select_uncertain_queries(scores, threshold=0.1)
    assert selected == ["q1"]
    assert margin_uncertainty(scores["q1"]) < margin_uncertainty(scores["q2"])
