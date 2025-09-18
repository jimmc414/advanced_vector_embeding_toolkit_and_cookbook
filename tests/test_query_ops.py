import numpy as np
from embkit.lib.query_ops import directional, cone_filter, polytope_filter
from embkit.lib.utils import l2n

def test_directional_shifts_rank():
    q = l2n(np.array([1.0, 0.0], dtype=np.float32), axis=None)
    vdir = l2n(np.array([0.0, 1.0], dtype=np.float32), axis=None)
    D = l2n(np.array([[0.9, 0.1], [0.4, 0.9]], dtype=np.float32), axis=1)
    base = directional(q, np.zeros_like(vdir), D, alpha=0.0)
    with_dir = directional(q, vdir, D, alpha=0.8)
    assert base[0][0] == 0
    assert with_dir[0][0] == 1  # direction pushes doc1 up

def test_cone_excludes_wide_angle():
    q = np.array([1.0, 0.0], dtype=np.float32)
    D = l2n(np.array([[0.95, 0.1], [0.5, 0.5], [-0.9, 0.0]], dtype=np.float32), axis=1)
    keep = cone_filter(q, D, cos_min=0.8)
    assert keep == [0]  # only near q

def test_polytope_intersection():
    D = l2n(np.array([[0.9,0.9], [0.9,0.1], [0.1,0.9]], dtype=np.float32), axis=1)
    c1 = (np.array([1.0, 0.0], dtype=np.float32), 0.7)
    c2 = (np.array([0.0, 1.0], dtype=np.float32), 0.7)
    keep = polytope_filter(D, [c1, c2])
    assert keep == [0]
