import numpy as np
from embkit.lib.query_ops import directional
from embkit.lib.safety.pii import pii_contains, pii_redact, pii_filter_results
from embkit.lib.safety.repellors import apply_repellors
from embkit.lib.utils import l2n

def test_pii_regex():
    assert pii_contains("Contact: john.doe@example.com")
    assert pii_contains("SSN 123-45-6789")
    assert not pii_contains("No sensitive data")
    red = pii_redact("Mail me at jane@site.org, SSN 123-45-6789, phone +1 (212) 555-1212")
    assert "[REDACTED_EMAIL]" in red and "[REDACTED_SSN]" in red and "[REDACTED_PHONE]" in red


def test_pii_filter_results_masks_snippets():
    rows = [{"id": "1", "snippet": "Call me at 212-555-1212"}, {"id": "2", "snippet": "Clean"}]
    filtered = pii_filter_results(rows)
    assert filtered[0]["snippet"] == "[REDACTED]"
    assert filtered[1]["snippet"] == "Clean"

def test_repellor_penalty():
    D = l2n(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), axis=1)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    B = l2n(np.array([[1.0, 0.0]], dtype=np.float32), axis=1)  # block first direction
    penalized = apply_repellors(scores, D, B, lam=0.5)
    assert penalized[0] < penalized[1]


def test_directional_scores_feed_repellor_penalty():
    q = l2n(np.array([1.0, 0.0], dtype=np.float32), axis=None)
    vdir = l2n(np.array([0.0, 1.0], dtype=np.float32), axis=None)
    D = l2n(np.array([[0.9, 0.1], [0.3, 0.95], [0.7, 0.2]], dtype=np.float32), axis=1)
    id_list = ["doc0", "doc1", "doc2"]
    ranked_base = directional(q, np.zeros_like(vdir), D, alpha=0.0)
    ids = [id_list[idx] for idx, _ in ranked_base]
    scores = np.array([score for _, score in ranked_base], dtype=np.float32)
    ranked_dir = directional(q, vdir, D, alpha=0.8)
    cand_index = {doc_id: idx for idx, doc_id in enumerate(ids)}
    re_sc = []
    for row_idx, score in ranked_dir:
        if 0 <= row_idx < len(id_list):
            doc_id = id_list[row_idx]
            if doc_id in cand_index:
                re_sc.append((cand_index[doc_id], float(score)))
    re_sc.sort(key=lambda x: -x[1])
    top_k = 2
    top = re_sc[: min(top_k, len(re_sc))]
    ids = [ids[idx] for idx, _ in top]
    scores = np.array([score for _, score in top], dtype=np.float32)

    id_to_row = {doc_id: i for i, doc_id in enumerate(id_list)}
    repellors = D[2:3]
    rows = [id_to_row.get(doc_id, -1) for doc_id in ids]
    valid = [i for i, row in enumerate(rows) if row >= 0]
    lam = 0.5
    if valid:
        D_subset = D[np.array([rows[i] for i in valid], dtype=int)]
        penalized = apply_repellors(scores[valid], D_subset, repellors, lam=lam)
        scores = scores.astype(np.float32)
        for idx_local, sc in zip(valid, penalized):
            scores[idx_local] = sc
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        ids = [ids[i] for i in order]
        scores = scores[order]

    assert ids[0] == "doc2"
    expected_scores = np.array([score for _, score in ranked_dir[:top_k]], dtype=np.float32)
    expected_rows = np.array([idx for idx, _ in ranked_dir[:top_k]], dtype=int)
    expected = apply_repellors(expected_scores, D[expected_rows], repellors, lam=lam)
    np.testing.assert_allclose(scores, expected[: scores.shape[0]], rtol=1e-5, atol=1e-6)
