from __future__ import annotations

import re
from typing import Iterable, List, MutableMapping

_RE_EMAIL = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_PHONE = re.compile(r"\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b")

__all__ = ["pii_contains", "pii_redact", "pii_filter_results"]


def pii_contains(text: str) -> bool:
    return any(r.search(text) for r in (_RE_EMAIL, _RE_SSN, _RE_PHONE))


def pii_redact(text: str) -> str:
    t = _RE_EMAIL.sub("[REDACTED_EMAIL]", text)
    t = _RE_SSN.sub("[REDACTED_SSN]", t)
    t = _RE_PHONE.sub("[REDACTED_PHONE]", t)
    return t


def pii_filter_results(results: Iterable[MutableMapping[str, str]], field: str = "snippet", redaction: str = "[REDACTED]") -> List[MutableMapping[str, str]]:
    """Redact or drop snippets containing PII."""
    filtered: List[MutableMapping[str, str]] = []
    for row in results:
        value = row.get(field, "")
        if not isinstance(value, str):
            filtered.append(row)
            continue
        if pii_contains(value):
            row = dict(row)
            row[field] = redaction
        filtered.append(row)
    return filtered
