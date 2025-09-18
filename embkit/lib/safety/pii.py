from __future__ import annotations
import re

_RE_EMAIL = re.compile(r'\b[^@\s]+@[^@\s]+\.[^@\s]+\b')
_RE_SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_RE_PHONE = re.compile(r'\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b')

def pii_contains(text: str) -> bool:
    return any(r.search(text) for r in (_RE_EMAIL, _RE_SSN, _RE_PHONE))

def pii_redact(text: str) -> str:
    t = _RE_EMAIL.sub("[REDACTED_EMAIL]", text)
    t = _RE_SSN.sub("[REDACTED_SSN]", t)
    t = _RE_PHONE.sub("[REDACTED_PHONE]", t)
    return t
