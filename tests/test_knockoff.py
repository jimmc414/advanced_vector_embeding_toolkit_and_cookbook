import numpy as np

from embkit.lib.analysis.knockoff import knockoff_adjust, knockoff_scores


def test_knockoff_adjust_removes_attribute_component():
    doc = np.array([0.2, 0.8], dtype=np.float32)
    attr = np.array([0.0, 1.0], dtype=np.float32)
    adjusted = knockoff_adjust(doc, attr, remove=True)
    assert np.allclose(adjusted[1], 0.0, atol=1e-6)


def test_knockoff_scores_reduces_alignment():
    q = np.array([0.0, 1.0], dtype=np.float32)
    doc = np.array([[0.1, 0.9]], dtype=np.float32)
    attr = np.array([0.0, 1.0], dtype=np.float32)
    base, adj = knockoff_scores(q, doc, attr, remove=True)
    assert adj[0] < base[0]
