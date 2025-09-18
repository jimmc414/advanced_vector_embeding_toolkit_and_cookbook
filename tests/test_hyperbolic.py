import numpy as np

from embkit.lib.utils.hyperbolic import poincare_distance, project_to_ball


def test_poincare_distance_increases_near_boundary():
    center = np.array([0.1, 0.1], dtype=np.float32)
    boundary = np.array([0.95, 0.0], dtype=np.float32)
    center = project_to_ball(center)
    boundary = project_to_ball(boundary)
    d_center = poincare_distance(center, center + np.array([0.05, 0.0], dtype=np.float32))
    d_boundary = poincare_distance(boundary, project_to_ball(boundary + np.array([0.05, 0.0], dtype=np.float32)))
    assert d_boundary > d_center


def test_project_to_ball_clamps_norm():
    vec = np.array([2.0, 0.0], dtype=np.float32)
    projected = project_to_ball(vec)
    assert np.linalg.norm(projected) < 1.0
