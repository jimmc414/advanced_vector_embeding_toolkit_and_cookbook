from embkit.lib.analysis.counterfactual import generate_counterfactuals, rank_delta


def test_generate_counterfactuals_swaps_facets():
    query = "top universities in Germany"
    mapping = {"Germany": ["Europe", "Asia"]}
    variants = generate_counterfactuals(query, mapping)
    assert "Europe" in variants[0]
    assert len(variants) == 2


def test_rank_delta_reports_changes():
    original = ["a", "b", "c"]
    variant = ["b", "a", "d"]
    delta = rank_delta(original, variant)
    assert delta["b"] == 1  # improved by one position
    assert delta["a"] == -1  # dropped by one position
