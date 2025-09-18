from embkit.lib.query_ops import merge_generative_dense


def test_merge_generative_dense_prefers_shared_hits():
    generative = [("docA", 0.9), ("docB", 0.6)]
    dense = {"docA": 0.8, "docC": 0.95}
    fused = merge_generative_dense(generative, dense, weight=0.6, top_k=3)
    assert fused[0][0] == "docA"
    ids = [doc for doc, _ in fused]
    assert "docC" in ids


def test_merge_generative_dense_handles_empty_inputs():
    assert merge_generative_dense([], {}, weight=0.5) == []
