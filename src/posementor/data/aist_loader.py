from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from posementor.utils.io import load_pickle_or_npz, try_get_array


@dataclass(slots=True)
class AISTSequence:
    seq_id: str
    style: str
    joints3d: np.ndarray  # [T, 17, 3]
    fps: int


def infer_style_from_seq_id(seq_id: str) -> str:
    """
    AIST++ 序列名常见格式：gBR_sBM_cAll_d04_mBR0_ch01
    这里直接取第一段 gXX 作为舞种标签。
    """
    token = seq_id.split("_")[0]
    return token


def _reshape_to_tjc(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"无法识别的 3D 关键点维度: {arr.shape}")
    return arr.astype(np.float32)


def _select_17_joints(joints3d: np.ndarray) -> np.ndarray:
    if joints3d.shape[1] == 17:
        return joints3d
    if joints3d.shape[1] > 17:
        # Demo 阶段优先兼容：默认取前 17 个关节，后续可替换为 SMPL 精确映射。
        return joints3d[:, :17, :]
    raise ValueError(f"关节数量不足 17: {joints3d.shape}")


def load_gt3d_file(file_path: Path, default_fps: int = 60) -> AISTSequence:
    data = load_pickle_or_npz(file_path)
    arr = try_get_array(
        data,
        candidates=[
            "keypoints3d",
            "joints3d",
            "kp3d",
            "positions_3d",
            "joint3d",
            "data",
        ],
    )
    if arr is None:
        raise ValueError(f"找不到 3D 关键点字段: {file_path}")

    joints3d = _select_17_joints(_reshape_to_tjc(arr))

    fps_val = data.get("fps", default_fps)
    try:
        fps = int(np.array(fps_val).item())
    except Exception:  # noqa: BLE001
        fps = default_fps

    seq_id = file_path.stem
    style = infer_style_from_seq_id(seq_id)
    return AISTSequence(seq_id=seq_id, style=style, joints3d=joints3d, fps=fps)


def find_gt3d_files(annotations_root: Path) -> list[Path]:
    if not annotations_root.exists():
        return []

    # AIST++ fullset 标准 3D 文件在 keypoints3d 目录，优先精确匹配，避免误扫 motions。
    keypoints3d_files: list[Path] = []
    for suffix in ["*.pkl", "*.pickle", "*.npz"]:
        keypoints3d_files.extend(annotations_root.rglob(f"keypoints3d/{suffix}"))
    if keypoints3d_files:
        return sorted(set(keypoints3d_files))

    # 兼容非标准命名目录，尝试仅扫描明显是 3D 标注的目录名。
    candidates: list[Path] = []
    for folder_name in ["joints3d", "pose3d", "positions3d"]:
        for suffix in ["*.pkl", "*.pickle", "*.npz"]:
            candidates.extend(annotations_root.rglob(f"{folder_name}/{suffix}"))
    if candidates:
        return sorted(set(candidates))

    # 最后兜底：若目录结构未知，递归扫描全部标注文件。
    for suffix in ["*.pkl", "*.pickle", "*.npz"]:
        candidates.extend(annotations_root.rglob(suffix))

    return sorted(set(candidates))
