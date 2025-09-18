from __future__ import annotations

from typing import Sequence

from .compression import summarize_sentences


def build_memory_prompt(query: str, neighbors: Sequence[dict]) -> str:
    """Append summarized neighbor hints to the query prompt."""
    if not neighbors:
        return query
    hints = []
    for item in neighbors[:5]:
        summary = item.get("summary") or summarize_sentences([item.get("text", "")], max_sentences=1)
        if summary:
            hints.append(f"[MEM] {summary.strip()}")
    tail = "\n".join(hints)
    return f"{query}\n{tail}" if tail else query
