from __future__ import annotations

import numpy as np

from posementor.utils.math3d import compute_joint_angle, mpjpe


def test_compute_joint_angle_right_angle() -> None:
    points = np.zeros((1, 17, 3), dtype=np.float32)
    points[0, 5] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    points[0, 7] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    points[0, 9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    angle = compute_joint_angle(points, 5, 7, 9)
    assert np.isclose(float(angle[0]), 90.0, atol=1e-3)


def test_mpjpe_meter_to_mm() -> None:
    pred = np.zeros((2, 17, 3), dtype=np.float32)
    gt = np.zeros((2, 17, 3), dtype=np.float32)
    gt[..., 0] = 0.01

    value = mpjpe(pred, gt, to_mm=True)
    assert 9.0 <= value <= 11.0
