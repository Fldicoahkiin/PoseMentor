from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from posementor.utils.io import ensure_dir, load_pickle_or_npz

CAMERA_TOKEN_PATTERN = re.compile(r"_c(\d+)_", re.IGNORECASE)
AIST_CAMERA_COUNT = 9
AIST_DEFAULT_FPS = 60.0
AIST_TIMESTAMP_UNIT = 1_000_000.0
ALIGNMENT_SEARCH_RADIUS = 480
ALIGNMENT_SAMPLE_STEP = 4
ALIGNMENT_CONF_THRES = 0.1
ALIGNMENT_MIN_VISIBLE_JOINTS = 6
ALIGNMENT_MIN_SAMPLED_FRAMES = 24
ALIGNMENT_MODE = "aist_official_projection"


@dataclass(slots=True)
class AISTCameraGeometry:
    camera_id: str
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def to_payload(self) -> dict[str, object]:
        fx = float(self.intrinsic[0, 0])
        fy = float(self.intrinsic[1, 1])
        cx = float(self.intrinsic[0, 2])
        cy = float(self.intrinsic[1, 2])
        return {
            "camera_id": self.camera_id,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "focal_length_px": [round(fx, 4), round(fy, 4)],
            "principal_point_px": [round(cx, 4), round(cy, 4)],
            "intrinsic": np.round(self.intrinsic, 6).tolist(),
            "distortion": np.round(self.distortion, 6).tolist(),
            "rotation": np.round(self.rotation, 6).tolist(),
            "translation": np.round(self.translation, 6).tolist(),
        }


@lru_cache(maxsize=1)
def _load_mapping(mapping_path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            seq_id, setting_name = line.split(maxsplit=1)
            rows[seq_id] = setting_name
    return rows


@lru_cache(maxsize=32)
def _load_camera_settings(setting_path: str) -> dict[str, AISTCameraGeometry]:
    payload = json.loads(Path(setting_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"AIST 相机参数格式错误: {setting_path}")

    cameras: dict[str, AISTCameraGeometry] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        camera_id = str(raw.get("name", "")).strip().lower()
        if not camera_id:
            continue
        size_arr = np.asarray(raw.get("size", [0, 0]), dtype=np.int32).reshape(-1)
        image_size = (int(size_arr[0]), int(size_arr[1])) if size_arr.size >= 2 else (0, 0)
        intrinsic = np.asarray(raw.get("matrix", []), dtype=np.float32).reshape(3, 3)
        distortion = np.asarray(
            raw.get("distortions", [0, 0, 0, 0, 0]),
            dtype=np.float32,
        ).reshape(-1)
        rotation = np.asarray(raw.get("rotation", [0, 0, 0]), dtype=np.float32).reshape(3)
        translation = np.asarray(raw.get("translation", [0, 0, 0]), dtype=np.float32).reshape(3)
        cameras[camera_id] = AISTCameraGeometry(
            camera_id=camera_id,
            image_size=image_size,
            intrinsic=intrinsic,
            distortion=distortion,
            rotation=rotation,
            translation=translation,
        )
    if not cameras:
        raise ValueError(f"AIST 相机参数为空: {setting_path}")
    return cameras


@lru_cache(maxsize=64)
def _load_group_tracks(
    group_seq_id: str,
    annotations_root: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    root = Path(annotations_root)
    keypoints2d_file = root / "keypoints2d" / f"{group_seq_id}.pkl"
    keypoints3d_file = root / "keypoints3d" / f"{group_seq_id}.pkl"
    if not keypoints2d_file.exists():
        raise FileNotFoundError(f"AIST 2D 标注不存在: {keypoints2d_file}")
    if not keypoints3d_file.exists():
        raise FileNotFoundError(f"AIST 3D 标注不存在: {keypoints3d_file}")

    raw_2d = load_pickle_or_npz(keypoints2d_file)
    raw_3d = load_pickle_or_npz(keypoints3d_file)

    keypoints2d = np.asarray(raw_2d.get("keypoints2d"), dtype=np.float32)
    if keypoints2d.ndim != 4 or keypoints2d.shape[-1] < 3:
        raise ValueError(f"AIST 2D 标注维度错误: {keypoints2d.shape}")
    timestamps = np.asarray(raw_2d.get("timestamps", []), dtype=np.float64).reshape(-1)

    if "keypoints3d_optim" in raw_3d:
        keypoints3d = np.asarray(raw_3d["keypoints3d_optim"], dtype=np.float32)
    else:
        keypoints3d = np.asarray(raw_3d.get("keypoints3d"), dtype=np.float32)
    if keypoints3d.ndim != 3 or keypoints3d.shape[-1] != 3:
        raise ValueError(f"AIST 3D 标注维度错误: {keypoints3d.shape}")

    fps = _estimate_timeline_fps(timestamps)
    return keypoints2d, keypoints3d, timestamps, fps


@lru_cache(maxsize=64)
def _project_group_joints3d(
    group_seq_id: str,
    annotations_root: str,
    setting_path: str,
) -> dict[str, np.ndarray]:
    keypoints2d, keypoints3d, _timestamps, _fps = _load_group_tracks(group_seq_id, annotations_root)
    del keypoints2d
    cameras = _load_camera_settings(setting_path)
    projections: dict[str, np.ndarray] = {}
    for camera_id, camera in cameras.items():
        projected_frames: list[np.ndarray] = []
        rvec = camera.rotation.reshape(3, 1)
        tvec = camera.translation.reshape(3, 1)
        for frame in keypoints3d:
            projected, _ = cv2.projectPoints(
                frame,
                rvec,
                tvec,
                camera.intrinsic,
                camera.distortion,
            )
            projected_frames.append(projected.reshape(-1, 2).astype(np.float32))
        projections[camera_id] = np.stack(projected_frames, axis=0)
    return projections


def load_aist_alignment_meta(
    group_seq_id: str,
    annotations_root: str,
    cache_dir: str,
    refresh: bool = False,
) -> dict[str, object]:
    root = Path(annotations_root)
    cache_path = ensure_dir(Path(cache_dir)) / f"{group_seq_id}_alignment.json"
    keypoints2d_file = root / "keypoints2d" / f"{group_seq_id}.pkl"
    keypoints3d_file = root / "keypoints3d" / f"{group_seq_id}.pkl"
    mapping_path = root / "cameras" / "mapping.txt"
    if not keypoints2d_file.exists() or not keypoints3d_file.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"AIST 对齐依赖不存在: {group_seq_id}")

    mapping = _load_mapping(str(mapping_path))
    setting_name = mapping.get(group_seq_id)
    if not setting_name:
        raise KeyError(f"AIST 未找到相机方案映射: {group_seq_id}")
    setting_file = root / "cameras" / f"{setting_name}.json"
    if not setting_file.exists():
        raise FileNotFoundError(f"AIST 相机参数文件不存在: {setting_file}")

    dep_mtime = max(
        keypoints2d_file.stat().st_mtime_ns,
        keypoints3d_file.stat().st_mtime_ns,
        mapping_path.stat().st_mtime_ns,
        setting_file.stat().st_mtime_ns,
        Path(__file__).stat().st_mtime_ns,
    )
    if cache_path.exists() and not refresh and cache_path.stat().st_mtime_ns >= dep_mtime:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    keypoints2d, keypoints3d, _timestamps, fps = _load_group_tracks(group_seq_id, annotations_root)
    projected = _project_group_joints3d(group_seq_id, annotations_root, str(setting_file))
    cameras = _load_camera_settings(str(setting_file))

    official_frame_total = int(min(keypoints2d.shape[1], keypoints3d.shape[0]))
    camera_offsets: dict[str, int] = {}
    camera_sync_error_px: dict[str, float] = {}
    camera_frame_count: dict[str, int] = {}
    camera_geometry: dict[str, object] = {}

    view_count = min(AIST_CAMERA_COUNT, keypoints2d.shape[0])
    for camera_index in range(view_count):
        camera_id = f"c{camera_index + 1:02d}"
        if camera_id not in cameras or camera_id not in projected:
            continue
        observed_xy = keypoints2d[camera_index, :official_frame_total, :, :2].astype(np.float32)
        observed_conf = keypoints2d[camera_index, :official_frame_total, :, 2].astype(np.float32)
        best_shift, best_error = _search_best_shift(
            projected_xy=projected[camera_id][:official_frame_total],
            observed_xy=observed_xy,
            observed_conf=observed_conf,
        )
        camera_offsets[camera_id] = int(best_shift)
        camera_sync_error_px[camera_id] = round(float(best_error), 4)
        camera_frame_count[camera_id] = int(observed_xy.shape[0])
        camera_geometry[camera_id] = cameras[camera_id].to_payload()

    payload = {
        "dataset_id": "aistpp",
        "group_seq_id": group_seq_id,
        "mode": ALIGNMENT_MODE,
        "setting_name": setting_name,
        "setting_file": str(setting_file),
        "timeline_fps": round(float(fps), 6),
        "timeline_frame_count": official_frame_total,
        "alignment_search_radius": ALIGNMENT_SEARCH_RADIUS,
        "alignment_sample_step": ALIGNMENT_SAMPLE_STEP,
        "camera_offsets": camera_offsets,
        "camera_sync_error_px": camera_sync_error_px,
        "camera_frame_count": camera_frame_count,
        "camera_geometry": camera_geometry,
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def resolve_group_seq_id(video_stem: str) -> str:
    return CAMERA_TOKEN_PATTERN.sub("_cAll_", video_stem)


def extract_camera_id(video_stem: str) -> str | None:
    matched = CAMERA_TOKEN_PATTERN.search(video_stem)
    if not matched:
        return None
    return f"c{int(matched.group(1)):02d}"


def collect_group_video_paths(video_root: Path, group_seq_id: str) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for camera_index in range(1, AIST_CAMERA_COUNT + 1):
        camera_id = f"c{camera_index:02d}"
        candidate = video_root / f"{group_seq_id.replace('_cAll_', f'_{camera_id}_')}.mp4"
        if candidate.exists() and candidate.is_file():
            rows[camera_id] = candidate
    return rows


def read_video_stats(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or AIST_DEFAULT_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def plan_group_preview(
    alignment: dict[str, object],
    video_stats: dict[str, dict[str, float | int]],
    joints3d_frame_total: int,
    camera_ids: list[str] | None = None,
) -> dict[str, object]:
    camera_offsets_all = {
        str(key): int(value)
        for key, value in dict(alignment.get("camera_offsets", {})).items()
    }
    camera_sync_error_all = {
        str(key): float(value)
        for key, value in dict(alignment.get("camera_sync_error_px", {})).items()
    }
    camera_frame_total_all = {
        str(key): int(value)
        for key, value in dict(alignment.get("camera_frame_count", {})).items()
    }
    camera_geometry_all = {
        str(key): value
        for key, value in dict(alignment.get("camera_geometry", {})).items()
    }
    available = sorted(camera_ids or video_stats.keys())
    available = [
        camera_id
        for camera_id in available
        if camera_id in video_stats and camera_id in camera_offsets_all
    ]
    if not available:
        raise ValueError("当前分组没有可用的 AIST 多机位素材")

    min_offset = min(camera_offsets_all[camera_id] for camera_id in available)
    timeline_start_frame = max(0, -min_offset)
    frame_limits = [max(0, int(joints3d_frame_total) - timeline_start_frame)]
    camera_trim_start: dict[str, int] = {}
    camera_offsets: dict[str, int] = {}
    camera_sync_error_px: dict[str, float] = {}
    camera_frame_count: dict[str, int] = {}
    camera_geometry: dict[str, object] = {}

    for camera_id in available:
        offset = camera_offsets_all[camera_id]
        trim_start = timeline_start_frame + offset
        local_frames = int(video_stats[camera_id].get("frames", 0))
        official_frames = camera_frame_total_all.get(camera_id, local_frames)
        usable_frames = min(local_frames, official_frames) - trim_start
        frame_limits.append(max(0, usable_frames))
        camera_trim_start[camera_id] = int(trim_start)
        camera_offsets[camera_id] = int(offset)
        camera_sync_error_px[camera_id] = round(camera_sync_error_all.get(camera_id, 0.0), 4)
        camera_frame_count[camera_id] = int(min(local_frames, official_frames))
        geometry_payload = camera_geometry_all.get(camera_id)
        if geometry_payload is not None:
            camera_geometry[camera_id] = geometry_payload

    frame_total = max(0, min(frame_limits))
    return {
        "dataset_id": "aistpp",
        "group_seq_id": str(alignment.get("group_seq_id", "")),
        "mode": str(alignment.get("mode", ALIGNMENT_MODE)),
        "setting_name": str(alignment.get("setting_name", "")),
        "setting_file": str(alignment.get("setting_file", "")),
        "timeline_fps": float(alignment.get("timeline_fps", AIST_DEFAULT_FPS)),
        "timeline_frame_count": int(alignment.get("timeline_frame_count", joints3d_frame_total)),
        "timeline_start_frame": int(timeline_start_frame),
        "frame_total": int(frame_total),
        "available_cameras": available,
        "camera_offsets": camera_offsets,
        "camera_trim_start": camera_trim_start,
        "camera_sync_error_px": camera_sync_error_px,
        "camera_frame_count": camera_frame_count,
        "camera_geometry": camera_geometry,
    }


def load_group_keypoints2d(group_seq_id: str, annotations_root: str) -> tuple[np.ndarray, float]:
    keypoints2d, _keypoints3d, _timestamps, fps = _load_group_tracks(group_seq_id, annotations_root)
    return keypoints2d, fps


def _estimate_timeline_fps(timestamps: np.ndarray) -> float:
    if timestamps.size <= 1:
        return AIST_DEFAULT_FPS
    deltas = np.diff(timestamps.astype(np.float64))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return AIST_DEFAULT_FPS
    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        return AIST_DEFAULT_FPS
    return AIST_TIMESTAMP_UNIT / median_delta


def _search_best_shift(
    projected_xy: np.ndarray,
    observed_xy: np.ndarray,
    observed_conf: np.ndarray,
) -> tuple[int, float]:
    best_shift = 0
    best_error = float("inf")
    for shift in range(-ALIGNMENT_SEARCH_RADIUS, ALIGNMENT_SEARCH_RADIUS + 1):
        error = _score_shift(
            projected_xy=projected_xy,
            observed_xy=observed_xy,
            observed_conf=observed_conf,
            shift=shift,
        )
        if error < best_error:
            best_shift = shift
            best_error = error
    return best_shift, best_error


def _score_shift(
    projected_xy: np.ndarray,
    observed_xy: np.ndarray,
    observed_conf: np.ndarray,
    shift: int,
) -> float:
    start_proj = max(0, -shift)
    start_obs = start_proj + shift
    max_length = min(projected_xy.shape[0] - start_proj, observed_xy.shape[0] - start_obs)
    if max_length <= 0:
        return float("inf")

    sampled_proj = projected_xy[start_proj : start_proj + max_length : ALIGNMENT_SAMPLE_STEP]
    sampled_obs = observed_xy[start_obs : start_obs + max_length : ALIGNMENT_SAMPLE_STEP]
    sampled_conf = observed_conf[start_obs : start_obs + max_length : ALIGNMENT_SAMPLE_STEP]
    if sampled_proj.shape[0] < ALIGNMENT_MIN_SAMPLED_FRAMES:
        return float("inf")

    visible_mask = sampled_conf > ALIGNMENT_CONF_THRES
    valid_frame_mask = np.sum(visible_mask, axis=1) >= ALIGNMENT_MIN_VISIBLE_JOINTS
    if not np.any(valid_frame_mask):
        return float("inf")

    diff = sampled_proj[valid_frame_mask] - sampled_obs[valid_frame_mask]
    joint_error = np.linalg.norm(diff, axis=-1)
    joint_error = np.where(visible_mask[valid_frame_mask], joint_error, np.nan)
    frame_error = np.nanmean(joint_error, axis=1)
    if frame_error.size == 0 or np.all(np.isnan(frame_error)):
        return float("inf")
    return float(np.nanmean(frame_error))
