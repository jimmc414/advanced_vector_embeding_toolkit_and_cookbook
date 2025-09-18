from __future__ import annotations

import numpy as np

__all__ = ["hotelling_t2", "detect_drift"]


def hotelling_t2(batch: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray, regularizer: float = 1e-6) -> float:
    """Compute Hotelling's T^2 statistic between ``batch`` mean and reference statistics."""
    X = np.asarray(batch, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("batch must be 2D")
    mean = X.mean(axis=0)
    diff = mean - np.asarray(ref_mean, dtype=np.float32)
    cov = np.asarray(ref_cov, dtype=np.float32)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("ref_cov must be square")
    cov_reg = cov + np.eye(cov.shape[0], dtype=np.float32) * float(regularizer)
    try:
        inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ref_cov must be invertible") from exc
    score = float(diff.T @ inv @ diff)
    return score


def detect_drift(batch: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray, threshold: float, regularizer: float = 1e-6) -> bool:
    """Return ``True`` when the Hotelling's T^2 statistic exceeds ``threshold``."""
    score = hotelling_t2(batch, ref_mean, ref_cov, regularizer=regularizer)
    return bool(score > threshold)
