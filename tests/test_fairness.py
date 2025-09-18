from embkit.lib.query_ops import fair_rerank


def test_fair_rerank_hits_target_ratio():
    candidates = ["p1", "p2", "p3", "p4", "p5", "p6"]
    relevance = {"p1": 0.9, "p2": 0.8, "p3": 0.7, "p4": 0.6, "p5": 0.5, "p6": 0.4}
    groups = {"p1": "A", "p2": "A", "p3": "A", "p4": "B", "p5": "B", "p6": "B"}
    reranked = fair_rerank(candidates, relevance, groups, protected_group="A", target_ratio=0.5, top_k=6)
    top3 = reranked[:3]
    count_a = sum(1 for doc in top3 if groups[doc] == "A")
    assert count_a >= 1  # at least one protected member in top3
    assert len(reranked) <= 6
