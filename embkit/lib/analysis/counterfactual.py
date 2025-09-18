from __future__ import annotations

from typing import Dict, List, Sequence


def generate_counterfactuals(query: str, facet_map: Dict[str, Sequence[str]]) -> List[str]:
    """Swap known facet values in ``query`` to create counterfactual variations."""
    variants: List[str] = []
    for facet, replacements in facet_map.items():
        if facet in query:
            for repl in replacements:
                if repl == facet:
                    continue
                variants.append(query.replace(facet, repl))
    return variants


def rank_delta(original: Sequence[str], counterfactual: Sequence[str]) -> Dict[str, int]:
    """Compute rank change for documents across two result lists."""
    pos = {doc: idx for idx, doc in enumerate(original)}
    delta: Dict[str, int] = {}
    for idx, doc in enumerate(counterfactual):
        if doc in pos:
            delta[doc] = pos[doc] - idx
    return delta
