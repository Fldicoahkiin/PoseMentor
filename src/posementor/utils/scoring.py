from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from posementor.utils.joints import JOINT_ADVICE, JOINT_NAMES
from posementor.utils.math3d import compute_angle_dict, mpjpe, per_joint_error_mm


@dataclass(slots=True)
class ScoreDetail:
    score: float
    mpjpe_mm: float
    angle_error_deg: float
    worst_joint: str
    joint_errors_mm: dict[str, float]
    advice_text: str


def _joint_flatten(frames3d: np.ndarray) -> np.ndarray:
    """将 [T, J, 3] 展平用于 DTW。"""
    return frames3d.reshape(frames3d.shape[0], -1)


def dtw_align_indices(query3d: np.ndarray, ref3d: np.ndarray) -> tuple[list[int], list[int], float]:
    q_vec = _joint_flatten(query3d)
    r_vec = _joint_flatten(ref3d)
    distance, path = fastdtw(q_vec, r_vec, dist=euclidean)
    q_idx = [p[0] for p in path]
    r_idx = [p[1] for p in path]
    return q_idx, r_idx, float(distance)


def compute_angle_error_deg(pred3d: np.ndarray, ref3d: np.ndarray) -> float:
    pred_angle = compute_angle_dict(pred3d)
    ref_angle = compute_angle_dict(ref3d)
    all_err: list[float] = []
    for name in pred_angle:
        all_err.extend(np.abs(pred_angle[name] - ref_angle[name]).tolist())
    return float(np.mean(all_err)) if all_err else 0.0


def make_advice(worst_joint: str, signed_angle_diff: float) -> str:
    template = JOINT_ADVICE.get(worst_joint)
    if template is None:
        return "动作节奏接近模板，请继续保持。"
    if signed_angle_diff > 0:
        return template.less_flex_text
    return template.more_flex_text


def score_from_errors(mpjpe_mm_val: float, angle_error_deg: float) -> float:
    # Demo 阶段使用线性融合，便于解释评分依据。
    score = 100.0 - 1.2 * max(0.0, mpjpe_mm_val - 18.0) - 2.0 * max(0.0, angle_error_deg - 4.0)
    return float(np.clip(score, 0.0, 100.0))


def evaluate_aligned_sequence(pred3d: np.ndarray, ref3d: np.ndarray) -> ScoreDetail:
    """对齐后计算 MPJPE + 角度误差 + 关节级别错误。"""
    mpjpe_mm_val = mpjpe(pred3d, ref3d, to_mm=True)
    angle_error_deg = compute_angle_error_deg(pred3d, ref3d)

    joint_errors = per_joint_error_mm(pred3d, ref3d)
    mean_joint_err = np.mean(joint_errors, axis=0)
    worst_idx = int(np.argmax(mean_joint_err))
    worst_joint = JOINT_NAMES[worst_idx]

    pred_angle = compute_angle_dict(pred3d)
    ref_angle = compute_angle_dict(ref3d)
    signed_angle_diff = 0.0
    if worst_joint in pred_angle:
        signed_angle_diff = float(np.mean(pred_angle[worst_joint] - ref_angle[worst_joint]))

    advice_text = make_advice(worst_joint, signed_angle_diff)

    score = score_from_errors(mpjpe_mm_val=mpjpe_mm_val, angle_error_deg=angle_error_deg)

    joint_errors_mm = {name: float(mean_joint_err[i]) for i, name in enumerate(JOINT_NAMES)}
    return ScoreDetail(
        score=score,
        mpjpe_mm=mpjpe_mm_val,
        angle_error_deg=angle_error_deg,
        worst_joint=worst_joint,
        joint_errors_mm=joint_errors_mm,
        advice_text=advice_text,
    )
