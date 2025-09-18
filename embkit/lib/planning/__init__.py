from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Protocol, Sequence

__all__ = [
    "Document",
    "planned_search",
    "multi_hop_retrieve",
    "extract_born_in",
    "extract_person_name",
]


@dataclass
class Document:
    """Minimal representation returned by planning helpers."""
    id: str
    text: str
    score: float = 0.0


class SearchProtocol(Protocol):
    def __call__(self, query: str, k: int = 10) -> Sequence[Document]:
        ...


def extract_born_in(query: str) -> tuple[str, str] | None:
    """Return (entity_query, location) if the query contains "born in" pattern."""
    match = re.search(r"(.+?)\s+born in\s+([\w\s]+)$", query, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def planned_search(query: str, search_fn: SearchProtocol, k: int = 10) -> List[Document]:
    """Plan simple two-stage queries such as "presidents born in New York"."""
    parts = extract_born_in(query)
    if not parts:
        return list(search_fn(query, k=k))
    entity_query, location = parts
    candidates = list(search_fn(entity_query, k=max(k, 50)))
    filtered = [doc for doc in candidates if location.lower() in doc.text.lower()]
    return filtered[:k] if filtered else candidates[:k]


def extract_person_name(text: str) -> str | None:
    match = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
    return match.group(1) if match else None


def multi_hop_retrieve(question: str, search_fn: SearchProtocol, k: int = 10) -> List[Document]:
    """Perform a naive two-hop retrieval plan for questions with intermediate entities."""
    born = re.search(r"born in (\d{4})", question, flags=re.IGNORECASE)
    president = re.search(r"president of ([^\(\)]+)", question, flags=re.IGNORECASE)
    if born and president:
        year = born.group(1)
        country = president.group(1).strip()
        first_query = f"president of {country} born in {year}"
        first_docs = list(search_fn(first_query, k=max(5, k)))
        for doc in first_docs:
            person = extract_person_name(doc.text)
            if not person:
                continue
            remainder = question.split("?")[0].split(")")[-1].strip()
            follow = f"{person} {remainder}".strip()
            second_docs = list(search_fn(follow, k=k))
            if second_docs:
                return second_docs
    return list(search_fn(question, k=k))
