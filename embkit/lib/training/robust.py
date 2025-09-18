from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from ..utils import l2n

__all__ = ["generate_synonym_variants", "fgsm_perturb"]


def generate_synonym_variants(query: str, synonym_map: Dict[str, Iterable[str]]) -> List[str]:
    """Return paraphrased queries by swapping tokens with provided synonyms."""
    tokens = query.split()
    variants: List[str] = []
    for i, tok in enumerate(tokens):
        synonyms = synonym_map.get(tok.lower())
        if not synonyms:
            continue
        for syn in synonyms:
            replaced = tokens.copy()
            replaced[i] = syn
            variant = " ".join(replaced)
            if variant != query:
                variants.append(variant)
    return variants


def fgsm_perturb(embedding: np.ndarray, gradient: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply an FGSM-style perturbation and re-normalize the embedding."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    emb = np.asarray(embedding, dtype=np.float32)
    grad = np.asarray(gradient, dtype=np.float32)
    step = epsilon * np.sign(grad)
    return l2n(emb + step, axis=None)
