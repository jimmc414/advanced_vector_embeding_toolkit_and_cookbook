import numpy as np

from embkit.lib.index.router import merge_fanout, nearest_centroids, route_and_search


def test_nearest_centroid_prefers_closest():
    centroids = {
        "news": np.array([1.0, 0.0], dtype=np.float32),
        "papers": np.array([0.2, 0.9], dtype=np.float32),
    }
    q = np.array([0.1, 0.95], dtype=np.float32)
    ranked = nearest_centroids(q, centroids, top_n=2)
    assert ranked[0][0] == "papers"
    assert len(ranked) == 2


def test_route_and_search_hits_expected_index():
    centroids = {
        "news": np.array([1.0, 0.0], dtype=np.float32),
        "papers": np.array([0.2, 0.9], dtype=np.float32),
    }
    q = np.array([0.2, 0.98], dtype=np.float32)

    def make_search(name):
        def fn(query, k):
            base = 1.0 if name == "papers" else 0.5
            ids = [f"{name}_{i}" for i in range(k)]
            scores = [base - 0.01 * i for i in range(k)]
            return ids, scores

        return fn

    searchers = {name: make_search(name) for name in centroids}
    routed = route_and_search(q, centroids, searchers, k=3)
    assert all(r[0] == "papers" for r in routed)

    # merge fanout with mixed batches keeps global top order
    batch1 = route_and_search(q, centroids, searchers, k=2, fanout=1)
    batch2 = route_and_search(q, centroids, searchers, k=2, fanout=2)
    merged = merge_fanout([batch1, batch2], k=2)
    assert len(merged) == 2
    assert merged[0][2] >= merged[1][2]
