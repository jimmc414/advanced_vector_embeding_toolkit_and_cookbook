from __future__ import annotations

import numpy as np

from ..utils import l2n

__all__ = ["apply_secure_transform", "dp_gaussian_noise"]


def apply_secure_transform(embedding: np.ndarray, transform: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
    """Apply a client-side transform (e.g., random projection) and optional noise."""
    emb = np.asarray(embedding, dtype=np.float32)
    mat = np.asarray(transform, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("transform must be 2D")
    if emb.shape[-1] != mat.shape[1]:
        raise ValueError("transform dimensions do not align with embedding")
    projected = mat @ emb
    if noise_scale > 0:
        projected = projected + np.random.normal(scale=noise_scale, size=projected.shape).astype(np.float32)
    return l2n(projected, axis=None)


def dp_gaussian_noise(gradients: np.ndarray, clip_norm: float, noise_multiplier: float) -> np.ndarray:
    """Clip ``gradients`` and add Gaussian noise for DP-SGD style updates."""
    grads = np.asarray(gradients, dtype=np.float32)
    norm = float(np.linalg.norm(grads))
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    scale = min(1.0, clip_norm / (norm + 1e-12))
    clipped = grads * scale
    if noise_multiplier > 0:
        noise = np.random.normal(scale=noise_multiplier * clip_norm, size=clipped.shape).astype(np.float32)
        clipped = clipped + noise
    return clipped.astype(np.float32)
