from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .dummy import DummyEncoder

try:  # Optional Hugging Face support
    from .hf import (
        MODEL_REGISTRY,
        HuggingFaceEncoder,
        HuggingFaceModelConfig,
        resolve_model_name,
    )
except Exception:  # pragma: no cover - optional dependency
    MODEL_REGISTRY = {}
    HuggingFaceEncoder = None  # type: ignore
    HuggingFaceModelConfig = None  # type: ignore
    resolve_model_name = lambda name: name  # type: ignore


class QueryDocumentEncoder(Protocol):
    """Common interface for encoders supporting batch query/document encoding."""

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


def create_encoder(model_cfg, seed: int, *, dimension: int | None = None) -> QueryDocumentEncoder:
    """Instantiate an encoder based on the provided model configuration."""

    provider = getattr(model_cfg, "provider", "dummy")
    if provider == "dummy":
        d = dimension if dimension is not None else getattr(model_cfg, "dim", 32)
        return DummyEncoder(d=int(d), seed=seed)
    if provider == "huggingface":
        if HuggingFaceEncoder is None or HuggingFaceModelConfig is None:
            raise RuntimeError(
                "Hugging Face dependencies are not available. Install torch, transformers, "
                "sentence-transformers, and huggingface-hub."
            )
        cfg = HuggingFaceModelConfig(
            name_or_path=resolve_model_name(getattr(model_cfg, "name")),
            batch_size=getattr(model_cfg, "batch_size", 32),
            max_length=getattr(model_cfg, "max_length", 512),
            cache_dir=getattr(model_cfg, "cache_dir", None),
            revision=getattr(model_cfg, "revision", None),
            seed=seed,
        )
        return HuggingFaceEncoder(cfg)
    raise ValueError(f"Unknown model provider: {provider}")


__all__ = [
    "DummyEncoder",
    "HuggingFaceEncoder",
    "MODEL_REGISTRY",
    "QueryDocumentEncoder",
    "create_encoder",
    "resolve_model_name",
]
