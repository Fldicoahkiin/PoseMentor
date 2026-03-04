from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from posementor.utils.joints import SKELETON_EDGES
from posementor.utils.visualize import draw_pose_2d

YAW_DEG = 35.0
PITCH_DEG = 18.0
CANVAS_BG = 242
MARGIN = 44.0
ROOT_LEFT_HIP = 11
ROOT_RIGHT_HIP = 12
CAMERA_TOKEN_PATTERN = re.compile(r"_c\d+_")
TEMPORAL_SMOOTH_ALPHA = 0.72

def find_sequence_id(
    yolo2d_dir: Path,
    video_stem: str,
    source_video_name: str,
) -> str | None:
    direct = yolo2d_dir / f"{video_stem}.npz"
    if direct.exists():
        return video_stem

    call_stem = CAMERA_TOKEN_PATTERN.sub("_cAll_", video_stem)
    if call_stem != video_stem:
        call_npz = yolo2d_dir / f"{call_stem}.npz"
        if call_npz.exists():
            return call_stem

    for file_path in sorted(yolo2d_dir.glob("*.npz")):
        try:
            with np.load(file_path) as data:
                if "source_video_name" not in data.files:
                    continue
                value = str(np.asarray(data["source_video_name"]).reshape(-1)[0])
                if value == source_video_name:
                    return file_path.stem
        except Exception:
            continue
    return None


def _project_frame(points3d: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(YAW_DEG)
    pitch = np.deg2rad(PITCH_DEG)
    rot_y = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    root = (points3d[ROOT_LEFT_HIP] + points3d[ROOT_RIGHT_HIP]) * 0.5
    centered = points3d - root[None, :]
    rotated = centered @ rot_y.T @ rot_x.T
    return rotated.astype(np.float32)


def _normalize_xy(
    points: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale_x = (width - MARGIN * 2.0) / span[0]
    scale_y = (height - MARGIN * 2.0) / span[1]
    scale = float(min(scale_x, scale_y))
    mapped = (points - min_xy[None, :]) * scale + MARGIN
    mapped[:, 1] = float(height) - mapped[:, 1]
    return mapped


def _smooth_sequence(points: np.ndarray, alpha: float = TEMPORAL_SMOOTH_ALPHA) -> np.ndarray:
    if points.ndim != 3 or points.shape[0] <= 1:
        return points
    smoothed = points.copy()
    for frame_idx in range(1, points.shape[0]):
        smoothed[frame_idx] = alpha * smoothed[frame_idx - 1] + (1.0 - alpha) * points[frame_idx]
    return smoothed


def _draw_scene_grid(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    top = int(height * 0.62)
    bottom = int(height * 0.94)
    left = int(width * 0.14)
    right = int(width * 0.86)

    cv2.rectangle(
        canvas,
        (left, top),
        (right, bottom),
        (236, 236, 236),
        -1,
    )
    for idx in range(1, 7):
        x = int(left + (right - left) * idx / 7)
        cv2.line(canvas, (x, top), (x, bottom), (214, 214, 214), 1, lineType=cv2.LINE_AA)
    for idx in range(1, 4):
        y = int(top + (bottom - top) * idx / 4)
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (right, bottom), (198, 198, 198), 1, lineType=cv2.LINE_AA)


def _draw_3d_frame(
    projected_xyz: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    width: int,
    height: int,
    frame_idx: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), CANVAS_BG, dtype=np.uint8)
    _draw_scene_grid(canvas)
    points_xy = _normalize_xy(
        projected_xyz[:, :2],
        min_xy=min_xy,
        max_xy=max_xy,
        width=width,
        height=height,
    )
    depth = projected_xyz[:, 2]
    depth_min = float(np.min(depth))
    depth_span = float(np.max(depth) - depth_min + 1e-6)

    for a, b in SKELETON_EDGES:
        z_norm = (float(depth[a] + depth[b]) * 0.5 - depth_min) / depth_span
        color = (
            int(44 + 60 * z_norm),
            int(96 + 70 * (1.0 - z_norm)),
            int(178 + 48 * (1.0 - z_norm)),
        )
        pa = tuple(np.round(points_xy[a]).astype(int).tolist())
        pb = tuple(np.round(points_xy[b]).astype(int).tolist())
        cv2.line(canvas, pa, pb, (210, 210, 210), 6, lineType=cv2.LINE_AA)
        cv2.line(canvas, pa, pb, color, 3, lineType=cv2.LINE_AA)

    for idx in range(points_xy.shape[0]):
        z_norm = (float(depth[idx]) - depth_min) / depth_span
        color = (
            int(64 + 68 * z_norm),
            int(122 + 74 * (1.0 - z_norm)),
            int(198 + 42 * (1.0 - z_norm)),
        )
        p = tuple(np.round(points_xy[idx]).astype(int).tolist())
        cv2.circle(canvas, p, 6, (226, 226, 226), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p, 4, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"3D Skeleton   frame={frame_idx}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (36, 36, 36),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _write_browser_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    frame_fps = max(1.0, fps)

    with tempfile.TemporaryDirectory(prefix="pose_preview_", dir=str(path.parent)) as tmp_dir:
        raw_path = Path(tmp_dir) / "raw.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_fps,
            (width, height),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            command = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-profile:v",
                "baseline",
                "-level",
                "3.1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            try:
                subprocess.run(command, check=True)
                return
            except Exception:
                pass

        shutil.copy2(raw_path, path)


def render_pose_preview_videos(
    source_video: Path,
    keypoints2d: np.ndarray,
    joints3d: np.ndarray,
    output_2d: Path,
    output_3d: Path,
) -> dict[str, float]:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)

    frame_total = int(min(len(keypoints2d), len(joints3d)))
    if frame_total <= 0:
        cap.release()
        raise RuntimeError("视频与关键点帧数不匹配，无法生成预览。")

    joints3d_use = _smooth_sequence(joints3d[:frame_total].astype(np.float32))
    projected_all = np.stack(
        [_project_frame(joints3d_use[idx]) for idx in range(frame_total)],
        axis=0,
    )
    flat_xy = projected_all[:, :, :2].reshape(-1, 2)
    min_xy = np.percentile(flat_xy, 2, axis=0).astype(np.float32)
    max_xy = np.percentile(flat_xy, 98, axis=0).astype(np.float32)
    max_xy = np.maximum(max_xy, min_xy + 1e-3)

    frames_2d: list[np.ndarray] = []
    frames_3d: list[np.ndarray] = []
    for frame_idx in range(frame_total):
        ok, frame = cap.read()
        if not ok:
            break

        kp = keypoints2d[frame_idx].astype(np.float32)
        frame_2d = draw_pose_2d(frame, kp, conf_thres=0.05)
        cv2.putText(
            frame_2d,
            f"2D Skeleton frame={frame_idx}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (24, 24, 24),
            2,
            lineType=cv2.LINE_AA,
        )
        frames_2d.append(frame_2d)

        frame_3d = _draw_3d_frame(
            projected_all[frame_idx],
            min_xy=min_xy,
            max_xy=max_xy,
            width=width,
            height=height,
            frame_idx=frame_idx,
        )
        frames_3d.append(frame_3d)

    cap.release()
    _write_browser_mp4(output_2d, frames_2d, fps)
    _write_browser_mp4(output_3d, frames_3d, fps)
    return {"fps": fps, "frames": float(len(frames_2d))}
