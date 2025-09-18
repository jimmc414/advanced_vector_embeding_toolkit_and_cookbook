from __future__ import annotations

import numpy as np


def project_to_ball(vec: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Project ``vec`` inside the open Poincare ball if the norm exceeds one."""
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm >= 1.0:
        v = v / (norm + eps) * (1.0 - eps)
    return v.astype(np.float32)


def poincare_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Return the geodesic distance between ``u`` and ``v`` in the Poincare ball."""
    u = project_to_ball(u)
    v = project_to_ball(v)
    nu = float(np.sum(u * u))
    nv = float(np.sum(v * v))
    diff = float(np.sum((u - v) * (u - v)))
    denom = max((1.0 - nu) * (1.0 - nv), 1e-12)
    arg = 1.0 + 2.0 * diff / denom
    arg = max(arg, 1.0 + 1e-9)
    return float(np.arccosh(arg))
