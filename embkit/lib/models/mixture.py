from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class KeywordExpertRouter:
    """Route queries to experts based on keyword matches."""

    expert_keywords: Mapping[str, Sequence[str]]
    default_expert: str

    def score(self, query: str) -> Dict[str, int]:
        lowered = query.lower()
        scores: Dict[str, int] = {name: 0 for name in self.expert_keywords}
        for name, keywords in self.expert_keywords.items():
            scores[name] = sum(1 for kw in keywords if kw.lower() in lowered)
        return scores

    def route(self, query: str, top_n: int = 1) -> List[Tuple[str, float]]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if all(score == 0 for _, score in ranked):
            return [(self.default_expert, 0.0)]
        return [(name, float(score)) for name, score in ranked[: top_n]]


def combine_expert_embeddings(
    embeddings: Iterable[Tuple[str, Sequence[float]]],
    weights: Mapping[str, float],
) -> List[float]:
    """Combine expert embeddings with routing weights."""
    combined: List[float] | None = None
    total_weight = 0.0
    for expert, vector in embeddings:
        w = float(weights.get(expert, 0.0))
        if w <= 0.0:
            continue
        total_weight += w
        if combined is None:
            combined = [float(x) * w for x in vector]
        else:
            if len(combined) != len(vector):
                raise ValueError("embedding dimensions must match")
            for i, val in enumerate(vector):
                combined[i] += float(val) * w
    if combined is None or total_weight == 0.0:
        raise ValueError("no embeddings received with positive weight")
    return [val / total_weight for val in combined]
