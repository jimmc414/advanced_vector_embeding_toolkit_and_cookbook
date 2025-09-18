import numpy as np

from embkit.lib.monitoring.drift import detect_drift, hotelling_t2


def test_hotelling_t2_large_for_shift():
    ref_mean = np.zeros(2, dtype=np.float32)
    ref_cov = np.eye(2, dtype=np.float32) * 0.05
    batch = np.random.normal(loc=0.5, scale=0.05, size=(100, 2)).astype(np.float32)
    score = hotelling_t2(batch, ref_mean, ref_cov)
    assert score > 10.0


def test_detect_drift_threshold():
    ref_mean = np.zeros(2, dtype=np.float32)
    ref_cov = np.eye(2, dtype=np.float32) * 0.05
    batch = np.random.normal(loc=0.0, scale=0.05, size=(100, 2)).astype(np.float32)
    assert not detect_drift(batch, ref_mean, ref_cov, threshold=5.0)
    shifted = np.random.normal(loc=0.4, scale=0.05, size=(100, 2)).astype(np.float32)
    assert detect_drift(shifted, ref_mean, ref_cov, threshold=5.0)
