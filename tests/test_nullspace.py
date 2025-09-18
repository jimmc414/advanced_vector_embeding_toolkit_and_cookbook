import numpy as np

from embkit.lib.analysis.nullspace import remove_direction, remove_directions


def test_remove_direction_aligns_actor_pairs():
    gender = np.array([1.0, 0.0], dtype=np.float32)
    actor = np.array([0.9, 0.7], dtype=np.float32)
    actress = np.array([-0.9, 0.7], dtype=np.float32)
    actor_db = remove_direction(actor, gender)
    actress_db = remove_direction(actress, gender)
    assert np.allclose(actor_db, actress_db, atol=1e-6)


def test_remove_directions_batch_matches_single():
    direction = np.array([1.0, 1.0], dtype=np.float32)
    vectors = np.stack([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])
    projected = remove_directions(vectors, direction)
    expected = np.stack([remove_direction(v, direction) for v in vectors])
    assert np.allclose(projected, expected, atol=1e-6)
