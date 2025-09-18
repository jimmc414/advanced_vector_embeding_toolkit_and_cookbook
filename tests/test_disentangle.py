import numpy as np
import pytest

from embkit.lib.analysis.disentangle import merge_embedding, split_embedding, swap_subspace


def test_split_and_merge_round_trip():
    vec = np.arange(6, dtype=np.float32)
    parts = split_embedding(vec, [2, 4])
    assert len(parts) == 2
    merged = merge_embedding(parts)
    assert np.allclose(vec, merged)


def test_swap_subspace_replaces_component():
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([9, 8, 7, 6], dtype=np.float32)
    swapped = swap_subspace(a, b, [2, 2], index=1)
    assert np.allclose(swapped, np.array([1, 2, 7, 6], dtype=np.float32))
    with pytest.raises(IndexError):
        swap_subspace(a, b, [2, 2], index=3)
