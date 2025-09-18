from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..utils import l2n, set_determinism


MODEL_REGISTRY = {
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
    "mxbai-large": "mixedbread-ai/mxbai-embed-large-v1",
    "sfr-mistral": "Salesforce/SFR-Embedding-Mistral",
    "gist-embedding": "avsolatorio/GIST-Embedding-v0",
    "gte-large": "thenlper/gte-large",
}


@dataclass
class HuggingFaceModelConfig:
    name_or_path: str
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    seed: int = 42


class HuggingFaceEncoder:
    """Sentence-Transformer backed encoder operating purely on CPU."""

    def __init__(self, cfg: HuggingFaceModelConfig):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise RuntimeError(
                "torch is required for HuggingFaceEncoder; install torch>=2.1"
            ) from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for HuggingFaceEncoder"
            ) from exc

        set_determinism(cfg.seed)
        torch.manual_seed(cfg.seed)
        try:  # pragma: no cover - backend availability varies
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.set_num_threads(1)

        model_id = MODEL_REGISTRY.get(cfg.name_or_path, cfg.name_or_path)
        init_kwargs = {"device": "cpu"}
        if cfg.cache_dir:
            init_kwargs["cache_folder"] = cfg.cache_dir
        if cfg.revision:
            init_kwargs["model_kwargs"] = {"revision": cfg.revision}
        self._model = SentenceTransformer(model_id, **init_kwargs)
        if cfg.max_length:
            try:
                self._model.max_seq_length = int(cfg.max_length)
            except AttributeError:  # pragma: no cover - older st versions
                pass

        self.batch_size = int(cfg.batch_size)
        dim_getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        self.d = 0
        if callable(dim_getter):
            try:
                maybe_dim = dim_getter()
            except TypeError:  # pragma: no cover - signature differences
                maybe_dim = dim_getter
            if maybe_dim:
                self.d = int(maybe_dim)
        if self.d <= 0:
            # Fallback: run a dummy encode to infer dimensionality
            dummy = self._model.encode([""])
            dummy_arr = np.asarray(dummy, dtype=np.float32)
            if dummy_arr.ndim != 2:
                dummy_arr = np.atleast_2d(dummy_arr)
            self.d = int(dummy_arr.shape[1])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    def encode_document(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.d), dtype=np.float32)
        embeds = self._model.encode(
            texts_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        arr = np.asarray(embeds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return l2n(arr, axis=1)


def resolve_model_name(name: str) -> str:
    """Return the full Hugging Face identifier for an alias."""

    return MODEL_REGISTRY.get(name, name)
