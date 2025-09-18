import numpy as np

from embkit.lib.safety.privacy import apply_secure_transform, dp_gaussian_noise


def test_apply_secure_transform_normalizes_projection():
    np.random.seed(0)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    transform = np.random.normal(size=(2, 2)).astype(np.float32)
    projected = apply_secure_transform(emb, transform, noise_scale=1e-6)
    assert np.allclose(np.linalg.norm(projected), 1.0, atol=1e-4)


def test_dp_gaussian_noise_clips_gradients():
    np.random.seed(1)
    grads = np.array([10.0, 0.0], dtype=np.float32)
    noisy = dp_gaussian_noise(grads, clip_norm=1.0, noise_multiplier=0.0)
    assert np.linalg.norm(noisy) <= 1.0 + 1e-6
