from embkit.lib.models.mixture import KeywordExpertRouter, combine_expert_embeddings


def test_keyword_router_prefers_matching_expert():
    router = KeywordExpertRouter(
        expert_keywords={"code": ["def", "class"], "text": ["news", "article"]},
        default_expert="text",
    )
    ranked = router.route("How do I write a def in Python?", top_n=2)
    assert ranked[0][0] == "code"
    assert len(ranked) == 2


def test_combine_expert_embeddings_weighted_average():
    embeddings = [
        ("text", [1.0, 0.0]),
        ("code", [0.0, 1.0]),
    ]
    weights = {"text": 0.75, "code": 0.25}
    combined = combine_expert_embeddings(embeddings, weights)
    assert combined == [0.75, 0.25]
