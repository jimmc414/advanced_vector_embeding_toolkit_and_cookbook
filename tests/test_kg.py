import networkx as nx

from embkit.lib.graph.kg import fuse_with_graph, meta_path_scores


def test_meta_path_scores_rewards_connected_nodes():
    g = nx.Graph()
    g.add_node("gene:egfr", type="gene")
    g.add_node("drug:tk_inhibitor", type="drug")
    g.add_node("paper:1", type="paper")
    g.add_edge("gene:egfr", "drug:tk_inhibitor")
    g.add_edge("drug:tk_inhibitor", "paper:1")
    scores = meta_path_scores(g, ["gene:egfr"], ["gene", "drug", "paper"], decay=0.5)
    assert scores.get("paper:1", 0.0) > 0


def test_fuse_with_graph_boosts_graph_nodes():
    base = {"paper:1": 0.4, "paper:2": 0.5}
    kg = {"paper:1": 1.0}
    fused = fuse_with_graph(base, kg, weight=0.3)
    assert fused[0][0] == "paper:1"
