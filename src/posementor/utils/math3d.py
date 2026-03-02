from __future__ import annotations

import numpy as np

from posementor.utils.joints import ANGLE_DEFS, JOINT_NAMES


def safe_norm(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.linalg.norm(vec, axis=-1, keepdims=True).clip(min=eps)


def compute_joint_angle(points3d: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """计算夹角 ∠ABC，输入支持 [..., J, 3]。"""
    ba = points3d[..., a, :] - points3d[..., b, :]
    bc = points3d[..., c, :] - points3d[..., b, :]
    ba_u = ba / safe_norm(ba)
    bc_u = bc / safe_norm(bc)
    cos_val = np.sum(ba_u * bc_u, axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def compute_angle_dict(points3d: np.ndarray) -> dict[str, np.ndarray]:
    angles: dict[str, np.ndarray] = {}
    for angle_name, (a, b, c) in ANGLE_DEFS.items():
        angles[angle_name] = compute_joint_angle(points3d, a, b, c)
    return angles


def mpjpe(pred3d: np.ndarray, gt3d: np.ndarray, to_mm: bool = True) -> float:
    """Mean Per Joint Position Error，默认输出 mm。"""
    err = np.linalg.norm(pred3d - gt3d, axis=-1)
    mean_err = float(np.mean(err))
    if not to_mm:
        return mean_err

    # AIST++ 常见单位为米，若值明显过大则视为已是 mm。
    if mean_err < 10:
        return mean_err * 1000.0
    return mean_err


def per_joint_error_mm(pred3d: np.ndarray, gt3d: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(pred3d - gt3d, axis=-1)
    if float(np.mean(err)) < 10:
        return err * 1000.0
    return err


def center_pose(points3d: np.ndarray) -> np.ndarray:
    """以髋中心对齐，降低相机平移误差影响。"""
    left_hip = points3d[..., JOINT_NAMES.index("left_hip"), :]
    right_hip = points3d[..., JOINT_NAMES.index("right_hip"), :]
    root = (left_hip + right_hip) / 2.0
    return points3d - root[..., None, :]


def normalize_2d_points(keypoints2d: np.ndarray, image_w: float, image_h: float) -> np.ndarray:
    """将像素坐标归一化到 [-1, 1]，置信度不变。"""
    normed = keypoints2d.copy()
    normed[..., 0] = (normed[..., 0] / max(image_w, 1e-6)) * 2.0 - 1.0
    normed[..., 1] = (normed[..., 1] / max(image_h, 1e-6)) * 2.0 - 1.0
    return normed
