from __future__ import annotations

import logging

import numpy as np

from posementor.utils.joints import ANGLE_DEFS, JOINT_NAMES

logger = logging.getLogger(__name__)


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


def _auto_detect_unit_scale(mean_err: float) -> float:
    """根据误差量级推断单位：<10 视为米制需乘 1000，否则视为已是毫米。"""
    # 阈值 10: AIST++ 米制数据 MPJPE 通常在 0.02~2.0m 之间
    if mean_err < 10:
        logger.debug("MPJPE=%.4f 推断为米制，乘以 1000 转换为 mm", mean_err)
        return 1000.0
    return 1.0


def mpjpe(pred3d: np.ndarray, gt3d: np.ndarray, to_mm: bool = True) -> float:
    """Mean Per Joint Position Error，默认输出 mm。"""
    err = np.linalg.norm(pred3d - gt3d, axis=-1)
    mean_err = float(np.nanmean(err))
    if not np.isfinite(mean_err):
        logger.warning("MPJPE 计算结果非有限值 (NaN/Inf)，返回 0.0")
        return 0.0
    if not to_mm:
        return mean_err
    return mean_err * _auto_detect_unit_scale(mean_err)


def per_joint_error_mm(pred3d: np.ndarray, gt3d: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(pred3d - gt3d, axis=-1)
    scale = _auto_detect_unit_scale(float(np.nanmean(err)))
    return err * scale


_LEFT_HIP_IDX = JOINT_NAMES.index("left_hip")
_RIGHT_HIP_IDX = JOINT_NAMES.index("right_hip")


def center_pose(points3d: np.ndarray) -> np.ndarray:
    """以髋中心对齐，降低相机平移误差影响。"""
    left_hip = points3d[..., _LEFT_HIP_IDX, :]
    right_hip = points3d[..., _RIGHT_HIP_IDX, :]
    root = (left_hip + right_hip) / 2.0
    return points3d - root[..., None, :]


def normalize_2d_points(keypoints2d: np.ndarray, image_w: float, image_h: float) -> np.ndarray:
    """将像素坐标归一化到 [-1, 1]，置信度不变。"""
    normed = keypoints2d.copy()
    normed[..., 0] = (normed[..., 0] / max(image_w, 1e-6)) * 2.0 - 1.0
    normed[..., 1] = (normed[..., 1] / max(image_h, 1e-6)) * 2.0 - 1.0
    return normed
