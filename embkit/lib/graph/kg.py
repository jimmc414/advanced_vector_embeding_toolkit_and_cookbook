from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence

import networkx as nx


def meta_path_scores(
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    meta_path: Sequence[str],
    decay: float = 0.6,
) -> Dict[str, float]:
    """Traverse ``meta_path`` type sequence and score reachable nodes."""
    if not meta_path:
        raise ValueError("meta_path must be non-empty")
    scores: Dict[str, float] = {}
    queue = deque()
    for seed in seed_nodes:
        queue.append((seed, 0, 1.0))
    while queue:
        node, depth, weight = queue.popleft()
        node_type = graph.nodes[node].get("type")
        if node_type != meta_path[depth]:
            continue
        if depth == len(meta_path) - 1:
            scores[node] = scores.get(node, 0.0) + weight
            continue
        for nbr in graph.neighbors(node):
            queue.append((nbr, depth + 1, weight * decay))
    return scores


def fuse_with_graph(
    base_scores: Mapping[str, float],
    kg_scores: Mapping[str, float],
    weight: float = 0.3,
) -> List[tuple[str, float]]:
    """Combine dense scores with KG-derived boosts."""
    weight = float(weight)
    fused: Dict[str, float] = {}
    for doc, score in base_scores.items():
        fused[doc] = float(score)
    for doc, score in kg_scores.items():
        fused[doc] = fused.get(doc, 0.0) + weight * float(score)
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return ranked
