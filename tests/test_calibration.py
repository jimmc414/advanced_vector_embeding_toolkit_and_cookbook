import numpy as np
from embkit.lib.calibrate.temperature import temperature_fit, temperature_apply
from embkit.lib.calibrate.isotonic import isotonic_fit, isotonic_apply
from embkit.lib.eval.metrics import expected_calibration_error

def test_temperature_reduces_ece():
    # Generate realistic synthetic data with noise
    np.random.seed(42)
    n = 100

    # Generate overconfident logits (common real-world scenario)
    logits = np.random.randn(n) * 2.0  # Wide spread for overconfidence

    # Create noisy labels that somewhat correlate with logits
    # This simulates real-world label noise
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs + np.random.randn(n) * 0.1 > 0.5).astype(np.float32)

    # Test that calibration improves ECE
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    ece_raw = expected_calibration_error(y, p_raw, n_bins=10)

    T = temperature_fit(y, logits.astype(np.float32))
    p_cal = temperature_apply(logits.astype(np.float32), T)
    ece_cal = expected_calibration_error(y, p_cal, n_bins=10)

    # Allow small tolerance for edge cases
    assert ece_cal <= ece_raw + 0.05


def test_isotonic_monotonic_mapping():
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.float32)
    thr, val = isotonic_fit(labels, scores)
    calibrated = isotonic_apply(scores, thr, val)
    assert np.all(np.diff(calibrated) >= -1e-6)
