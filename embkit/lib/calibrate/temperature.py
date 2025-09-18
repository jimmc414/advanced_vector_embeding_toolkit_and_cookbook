from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from ..eval.metrics import expected_calibration_error

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def temperature_fit(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Fit T>0 to minimize NLL of sigmoid(logits/T)."""
    y = y_true.astype(np.float64).ravel()
    z = logits.astype(np.float64).ravel()
    if y.shape != z.shape: raise ValueError("y_true and logits must align")

    def nll(T: float) -> float:
        T = max(T, 1e-6)
        p = _sigmoid(z / T)
        # avoid log(0)
        p = np.clip(p, 1e-6, 1-1e-6)
        return float(-(y * np.log(p) + (1-y) * np.log(1-p)).mean())

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(res.x)

    def _ece_for(temp: float) -> float:
        temp = max(temp, 1e-6)
        probs = _sigmoid(z / temp)
        return expected_calibration_error(y, probs)

    baseline = _ece_for(1.0)
    best_T = T_opt
    best_ece = _ece_for(T_opt)
    if best_ece > baseline:
        best_T = 1.0
        best_ece = baseline
        grid = np.logspace(-2, 1.5, num=40)
        for cand in grid:
            ece = _ece_for(cand)
            if ece < best_ece:
                best_ece = ece
                best_T = float(cand)
    return float(best_T)

def temperature_apply(logits: np.ndarray, T: float) -> np.ndarray:
    return _sigmoid(logits.astype(np.float64) / float(T)).astype(np.float32)
