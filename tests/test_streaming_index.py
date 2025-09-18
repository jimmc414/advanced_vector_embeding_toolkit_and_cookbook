import time

import numpy as np

from embkit.lib.index.streaming import StreamingIndex


def test_streaming_index_add_and_search():
    idx = StreamingIndex(dim=2)
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = ["doc1", "doc2"]
    ts = [time.time(), time.time()]
    idx.add(vecs, ids, ts)
    results = idx.search(np.array([1.0, 0.1], dtype=np.float32), k=1)
    assert results[0][0] == "doc1"
    assert idx.size == 2


def test_prune_expired_removes_old_docs():
    idx = StreamingIndex(dim=2)
    now = time.time()
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = ["recent", "old"]
    ts = [now, now - 1000]
    idx.add(vecs, ids, ts)
    idx.prune_expired(current_ts=now, ttl_seconds=500)
    assert idx.size == 1
    assert idx.search(np.array([0.0, 1.0], dtype=np.float32), k=1)[0][0] == "recent"
