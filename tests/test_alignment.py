import numpy as np

from embkit.lib.index.alignment import align_vectors, alignment_error, solve_linear_map


def test_alignment_recovers_linear_transform():
    rng = np.random.default_rng(0)
    true_W = rng.standard_normal((4, 4)).astype(np.float32)
    X_src = rng.standard_normal((32, 4)).astype(np.float32)
    X_tgt = X_src @ true_W

    W = solve_linear_map(X_src, X_tgt)
    err = alignment_error(X_src, X_tgt, W)
    assert err < 1e-4

    new = rng.standard_normal((1, 4)).astype(np.float32)
    mapped = align_vectors(new, W)
    target = new @ true_W
    assert np.allclose(mapped, target, atol=1e-4)
