from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanFilter1D:
    process_var: float = 1e-4
    measure_var: float = 1e-2

    def __post_init__(self) -> None:
        self.x = 0.0
        self.p = 1.0

    def update(self, z: float) -> float:
        # 预测
        self.p += self.process_var

        # 更新
        k = self.p / (self.p + self.measure_var)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class KeypointKalmanSmoother:
    """对每个关节点 x/y 分别做 1D Kalman，适合实时流式平滑。"""

    def __init__(self, num_joints: int = 17, process_var: float = 1e-4, measure_var: float = 3e-3):
        self.num_joints = num_joints
        self.filters = [
            [
                KalmanFilter1D(process_var=process_var, measure_var=measure_var),
                KalmanFilter1D(process_var=process_var, measure_var=measure_var),
            ]
            for _ in range(num_joints)
        ]
        self.is_initialized = False

    def reset(self) -> None:
        for joint_filters in self.filters:
            for f in joint_filters:
                f.x = 0.0
                f.p = 1.0
        self.is_initialized = False

    def __call__(self, keypoints2d: np.ndarray) -> np.ndarray:
        smoothed = keypoints2d.copy()

        if not self.is_initialized:
            for j in range(self.num_joints):
                self.filters[j][0].x = float(keypoints2d[j, 0])
                self.filters[j][1].x = float(keypoints2d[j, 1])
            self.is_initialized = True
            return smoothed

        for j in range(self.num_joints):
            smoothed[j, 0] = self.filters[j][0].update(float(keypoints2d[j, 0]))
            smoothed[j, 1] = self.filters[j][1].update(float(keypoints2d[j, 1]))
        return smoothed
