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


def test_remove_directions_repeated_direction_zeroes_component():
    vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    repeated = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    projected = remove_directions(vectors, repeated)
    expected = np.zeros_like(vectors)
    assert np.allclose(projected, expected, atol=1e-6)


def test_remove_directions_linearly_dependent_matches_unique():
    vectors = np.array(
        [
            [1.0, 2.0, -0.5],
            [-3.0, 0.5, 1.5],
        ],
        dtype=np.float32,
    )
    dependent = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear with the first direction
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    unique = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    projected_dependent = remove_directions(vectors, dependent)
    projected_unique = remove_directions(vectors, unique)
    assert np.allclose(projected_dependent, projected_unique, atol=1e-6)
