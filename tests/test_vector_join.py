from embkit.lib.query_ops import vector_join, vector_join_and


def test_vector_join_and_returns_intersection():
    results = [
        [("docAB", 0.9), ("docA", 0.8)],
        [("docAB", 0.95), ("docB", 0.7)],
    ]
    joined = vector_join_and(results)
    assert joined[0][0] == "docAB"
    assert joined[0][2] == 2


def test_vector_join_fallback_when_empty():
    results = [
        [("docA", 0.9)],
        [("docB", 0.95)],
    ]
    joined = vector_join(results)
    assert len(joined) == 2
    assert joined[0][2] == 1
