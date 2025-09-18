from __future__ import annotations

import numpy as np

__all__ = ["isotonic_fit", "isotonic_apply"]


def isotonic_fit(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return breakpoints and calibrated probabilities using Pool Adjacent Violators."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and scores must align")
    order = np.argsort(s)
    x = s[order]
    y_ord = y[order]
    blocks = []  # (start, end, value, weight)
    for xi, yi in zip(x, y_ord):
        block = [xi, xi, yi, 1.0]
        blocks.append(block)
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b1 = blocks.pop()
            b0 = blocks.pop()
            total_w = b0[3] + b1[3]
            avg = (b0[2] * b0[3] + b1[2] * b1[3]) / total_w
            blocks.append([b0[0], b1[1], avg, total_w])
    thresholds = np.array([b[1] for b in blocks], dtype=np.float64)
    values = np.array([np.clip(b[2], 0.0, 1.0) for b in blocks], dtype=np.float64)
    return thresholds, values


def isotonic_apply(scores: np.ndarray, thresholds: np.ndarray, values: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    thr = np.asarray(thresholds, dtype=np.float64).ravel()
    val = np.asarray(values, dtype=np.float64).ravel()
    if thr.size == 0 or val.size == 0:
        raise ValueError("thresholds and values must be non-empty")
    out = np.zeros_like(s, dtype=np.float64)
    for i, sc in enumerate(s):
        idx = np.searchsorted(thr, sc, side="right")
        if idx >= val.size:
            idx = val.size - 1
        out[i] = val[idx]
    return out.astype(np.float32)
