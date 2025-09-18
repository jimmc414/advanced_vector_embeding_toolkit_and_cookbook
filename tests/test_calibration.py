import numpy as np
from embkit.lib.calibrate.temperature import temperature_fit, temperature_apply
from embkit.lib.eval.metrics import expected_calibration_error

def test_temperature_reduces_ece():
    y = np.array([1,0,1,0,1,0,1,0], dtype=np.float32)
    logits = np.array([3.0, 2.5, 2.0, -1.0, 4.0, -2.0, 1.5, -0.5], dtype=np.float32)
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    ece_raw = expected_calibration_error(y, p_raw, n_bins=10)
    T = temperature_fit(y, logits)
    p_cal = temperature_apply(logits, T)
    ece_cal = expected_calibration_error(y, p_cal, n_bins=10)
    assert ece_cal <= ece_raw
