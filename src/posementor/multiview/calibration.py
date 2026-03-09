from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from posementor.utils.io import load_yaml


@dataclass(slots=True)
class CameraCalibration:
    name: str
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    rvec: np.ndarray

    @property
    def projection(self) -> np.ndarray:
        return self.intrinsic @ np.hstack([self.rotation, self.translation.reshape(3, 1)])

    @property
    def normalized_projection(self) -> np.ndarray:
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])


@dataclass(slots=True)
class CalibrationRig:
    name: str
    world_unit: str
    cameras: dict[str, CameraCalibration]

    def camera(self, name: str) -> CameraCalibration:
        key = Path(name).stem.lower()
        if key not in self.cameras:
            raise KeyError(f"标定文件缺少机位: {name}")
        return self.cameras[key]



def _as_matrix(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} 形状错误，期望 {shape}，实际 {arr.shape}")
    return arr



def _as_vector(value: object, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} 长度错误，期望 {length}，实际 {arr.size}")
    return arr



def load_calibration_rig(path: Path) -> CalibrationRig:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"标定文件格式错误: {path}")

    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, dict) or not raw_cameras:
        raise ValueError(f"标定文件缺少 cameras: {path}")

    cameras: dict[str, CameraCalibration] = {}
    for camera_name, raw in raw_cameras.items():
        if not isinstance(raw, dict):
            raise ValueError(f"机位配置错误: {camera_name}")
        key = str(camera_name).strip().lower()
        image_size_raw = raw.get("image_size", [0, 0])
        image_size_vec = _as_vector(image_size_raw, f"{camera_name}.image_size", 2)
        image_size = (int(image_size_vec[0]), int(image_size_vec[1]))
        intrinsic = _as_matrix(raw.get("intrinsic"), f"{camera_name}.intrinsic", (3, 3))
        distortion = np.asarray(
            raw.get("distortion", [0, 0, 0, 0, 0]),
            dtype=np.float32,
        ).reshape(-1)
        rotation = _as_matrix(raw.get("rotation"), f"{camera_name}.rotation", (3, 3))
        translation = _as_vector(raw.get("translation"), f"{camera_name}.translation", 3)
        rvec, _ = cv2.Rodrigues(rotation)
        cameras[key] = CameraCalibration(
            name=key,
            image_size=image_size,
            intrinsic=intrinsic,
            distortion=distortion,
            rotation=rotation,
            translation=translation,
            rvec=rvec.reshape(3),
        )

    return CalibrationRig(
        name=str(payload.get("name", path.stem)).strip() or path.stem,
        world_unit=str(payload.get("world_unit", "meter")).strip() or "meter",
        cameras=cameras,
    )
