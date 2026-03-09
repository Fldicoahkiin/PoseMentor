from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from posementor.multiview.calibration import CalibrationRig
from posementor.multiview.naming import build_seq_id_from_rel_path
from posementor.utils.io import ensure_dir, load_yaml, write_csv


@dataclass(slots=True)
class TriangulationConfig:
    min_views: int = 2
    min_confidence: float = 0.2
    max_reprojection_error_px: float = 18.0
    smooth_alpha: float = 0.58


@dataclass(slots=True)
class Observation:
    camera_name: str
    xy: np.ndarray
    confidence: float



def load_triangulation_config(config_path: Path) -> TriangulationConfig:
    payload = load_yaml(config_path)
    if not isinstance(payload, dict):
        return TriangulationConfig()
    raw = payload.get("triangulation", {})
    if not isinstance(raw, dict):
        return TriangulationConfig()
    return TriangulationConfig(
        min_views=max(2, int(raw.get("min_views", 2))),
        min_confidence=float(raw.get("min_confidence", 0.2)),
        max_reprojection_error_px=float(raw.get("max_reprojection_error_px", 18.0)),
        smooth_alpha=float(raw.get("smooth_alpha", 0.58)),
    )



def load_session_tracks(
    pose2d_root: Path,
    session_name: str,
    camera_files: list[str],
) -> dict[str, np.ndarray]:
    tracks: dict[str, np.ndarray] = {}
    for camera_file in camera_files:
        camera_name = Path(camera_file).stem.lower()
        seq_id = build_seq_id_from_rel_path(f"{session_name}/{camera_name}.mp4")
        file_path = pose2d_root / f"{seq_id}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 2D 关键点文件: {file_path}")
        with np.load(file_path) as data:
            if "keypoints2d" not in data.files:
                raise KeyError(f"{file_path} 缺少 keypoints2d 字段")
            tracks[camera_name] = data["keypoints2d"].astype(np.float32)
    return tracks



def _smooth_valid_points(points: np.ndarray, valid_mask: np.ndarray, alpha: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    smoothed = points.copy()
    joint_count = points.shape[1]
    for joint_idx in range(joint_count):
        last_valid: np.ndarray | None = None
        for frame_idx in range(points.shape[0]):
            if not valid_mask[frame_idx, joint_idx]:
                continue
            current = points[frame_idx, joint_idx]
            if last_valid is None:
                smoothed[frame_idx, joint_idx] = current
                last_valid = current.copy()
                continue
            blended = alpha * last_valid + (1.0 - alpha) * current
            smoothed[frame_idx, joint_idx] = blended
            last_valid = blended
    return smoothed



def _undistort_point(camera, point_xy: np.ndarray) -> np.ndarray:
    normalized = cv2.undistortPoints(
        np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2),
        camera.intrinsic,
        camera.distortion,
    )
    return normalized.reshape(2).astype(np.float32)



def _solve_dlt(observations: list[Observation], rig: CalibrationRig) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for obs in observations:
        camera = rig.camera(obs.camera_name)
        undistorted = _undistort_point(camera, obs.xy)
        projection = camera.normalized_projection
        rows.append(undistorted[0] * projection[2] - projection[0])
        rows.append(undistorted[1] * projection[2] - projection[1])
    if len(rows) < 4:
        return None
    design = np.stack(rows, axis=0)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    point_h = vh[-1]
    if abs(float(point_h[3])) < 1e-6:
        return None
    point = point_h[:3] / point_h[3]
    return point.astype(np.float32)



def _reprojection_errors(
    point3d: np.ndarray,
    observations: list[Observation],
    rig: CalibrationRig,
) -> np.ndarray:
    errors: list[float] = []
    point = np.asarray(point3d, dtype=np.float32).reshape(1, 1, 3)
    for obs in observations:
        camera = rig.camera(obs.camera_name)
        projected, _ = cv2.projectPoints(
            point,
            camera.rvec,
            camera.translation.reshape(3, 1),
            camera.intrinsic,
            camera.distortion,
        )
        pixel = projected.reshape(2)
        errors.append(float(np.linalg.norm(pixel - obs.xy)))
    return np.asarray(errors, dtype=np.float32)



def triangulate_observations(
    observations: list[Observation],
    rig: CalibrationRig,
    config: TriangulationConfig,
) -> tuple[np.ndarray | None, float, list[str]]:
    active = [obs for obs in observations if obs.confidence >= config.min_confidence]
    used: list[str] = []
    while len(active) >= config.min_views:
        point = _solve_dlt(active, rig)
        if point is None:
            return None, 0.0, []
        errors = _reprojection_errors(point, active, rig)
        if errors.size == 0:
            return None, 0.0, []
        worst_idx = int(np.argmax(errors))
        worst_error = float(errors[worst_idx])
        if worst_error <= config.max_reprojection_error_px:
            used = [obs.camera_name for obs in active]
            return point, float(np.mean(errors)), used
        if len(active) == config.min_views:
            return None, worst_error, []
        active.pop(worst_idx)
    return None, 0.0, []



def triangulate_session_tracks(
    tracks: dict[str, np.ndarray],
    rig: CalibrationRig,
    config: TriangulationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_total = min(track.shape[0] for track in tracks.values())
    joint_count = min(track.shape[1] for track in tracks.values())
    joints3d = np.full((frame_total, joint_count, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros((frame_total, joint_count), dtype=bool)
    reprojection_errors = np.full((frame_total, joint_count), np.nan, dtype=np.float32)
    used_views = np.zeros((frame_total, joint_count), dtype=np.int16)

    for frame_idx in range(frame_total):
        for joint_idx in range(joint_count):
            observations: list[Observation] = []
            for camera_name, track in tracks.items():
                joint = track[frame_idx, joint_idx]
                if joint.shape[0] < 3:
                    continue
                if not np.isfinite(joint[:3]).all():
                    continue
                confidence = float(joint[2])
                if confidence < config.min_confidence:
                    continue
                observations.append(
                    Observation(
                        camera_name=camera_name,
                        xy=joint[:2].astype(np.float32),
                        confidence=confidence,
                    )
                )
            point3d, reproj_error, used = triangulate_observations(observations, rig, config)
            if point3d is None:
                continue
            joints3d[frame_idx, joint_idx] = point3d
            valid_mask[frame_idx, joint_idx] = True
            reprojection_errors[frame_idx, joint_idx] = reproj_error
            used_views[frame_idx, joint_idx] = len(used)

    smoothed = _smooth_valid_points(joints3d, valid_mask, alpha=config.smooth_alpha)
    summary = {
        "frame_count": int(frame_total),
        "joint_count": int(joint_count),
        "valid_joint_ratio": round(float(valid_mask.mean()), 4),
        "mean_reprojection_error_px": round(float(np.nanmean(reprojection_errors)), 4)
        if np.isfinite(reprojection_errors).any()
        else None,
        "min_required_views": int(config.min_views),
        "max_reprojection_error_px": float(config.max_reprojection_error_px),
    }
    return smoothed, {
        "valid_mask": valid_mask,
        "reprojection_error": reprojection_errors,
        "used_views": used_views,
        "summary": summary,
    }



def export_session_gt3d(
    gt3d_root: Path,
    session_name: str,
    camera_files: list[str],
    joints3d: np.ndarray,
    details: dict[str, Any],
) -> list[str]:
    ensure_dir(gt3d_root)
    output_ids: list[str] = []
    valid_mask = details["valid_mask"].astype(np.uint8)
    reprojection_error = np.nan_to_num(details["reprojection_error"], nan=-1.0).astype(np.float32)
    used_views = details["used_views"].astype(np.int16)
    for camera_file in camera_files:
        camera_name = Path(camera_file).stem.lower()
        seq_id = build_seq_id_from_rel_path(f"{session_name}/{camera_name}.mp4")
        out_file = gt3d_root / f"{seq_id}.npz"
        np.savez_compressed(
            out_file,
            joints3d=joints3d.astype(np.float32),
            valid_mask=valid_mask,
            reprojection_error_px=reprojection_error,
            used_views=used_views,
            session_id=np.array(session_name),
            camera_id=np.array(camera_name),
        )
        output_ids.append(seq_id)
    return output_ids



def write_triangulation_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    manifest = ensure_dir(output_root) / "gt3d_manifest.csv"
    serialized_rows = [
        {
            "session": row["session"],
            "frames": row["frames"],
            "joint_count": row["joint_count"],
            "valid_joint_ratio": row["valid_joint_ratio"],
            "mean_reprojection_error_px": row["mean_reprojection_error_px"],
            "seq_ids": row["seq_ids"],
        }
        for row in rows
    ]
    write_csv(
        manifest,
        rows=serialized_rows,
        fieldnames=[
            "session",
            "frames",
            "joint_count",
            "valid_joint_ratio",
            "mean_reprojection_error_px",
            "seq_ids",
        ],
    )
