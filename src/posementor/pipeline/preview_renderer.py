from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from posementor.multiview.naming import build_seq_id_from_rel_path
from posementor.utils.joints import SKELETON_EDGES
from posementor.utils.visualize import draw_pose_2d

YAW_DEG = 35.0
PITCH_DEG = 18.0
CANVAS_BG = 242
MARGIN = 24.0
ROOT_LEFT_HIP = 11
ROOT_RIGHT_HIP = 12
CAMERA_TOKEN_PATTERN = re.compile(r"_c\d+_")
TEMPORAL_SMOOTH_ALPHA = 0.72
PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
AXIS_LENGTH_PX = 56.0
GRID_TOP_RATIO = 0.60
GRID_BOTTOM_RATIO = 0.94
GRID_LEFT_RATIO = 0.10
GRID_RIGHT_RATIO = 0.90
POSE3D_JSON_DECIMALS = 4
POSE2D_JSON_DECIMALS = 2
POSE3D_PREVIEW_VERSION = 1


def find_sequence_id(
    yolo2d_dir: Path,
    video_stem: str,
    source_video_name: str,
    source_video_rel: str = "",
) -> str | None:
    direct = yolo2d_dir / f"{video_stem}.npz"
    if direct.exists():
        return video_stem

    if source_video_rel:
        rel_seq_id = build_seq_id_from_rel_path(source_video_rel)
        rel_file = yolo2d_dir / f"{rel_seq_id}.npz"
        if rel_file.exists():
            return rel_seq_id

    call_stem = CAMERA_TOKEN_PATTERN.sub("_cAll_", video_stem)
    if call_stem != video_stem:
        call_npz = yolo2d_dir / f"{call_stem}.npz"
        if call_npz.exists():
            return call_stem

    for file_path in sorted(yolo2d_dir.glob("*.npz")):
        try:
            with np.load(file_path) as data:
                if source_video_rel and "source_video_rel" in data.files:
                    rel_value = str(np.asarray(data["source_video_rel"]).reshape(-1)[0])
                    if rel_value == source_video_rel:
                        return file_path.stem
                if "source_video_name" not in data.files:
                    continue
                value = str(np.asarray(data["source_video_name"]).reshape(-1)[0])
                if value == source_video_name:
                    return file_path.stem
        except Exception:
            continue
    return None


def _build_rotation_matrices() -> tuple[np.ndarray, np.ndarray]:
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
    return rot_y, rot_x


def _project_frame(points3d: np.ndarray) -> np.ndarray:
    rot_y, rot_x = _build_rotation_matrices()
    root = (points3d[ROOT_LEFT_HIP] + points3d[ROOT_RIGHT_HIP]) * 0.5
    centered = points3d - root[None, :]
    rotated = centered @ rot_y.T @ rot_x.T
    return rotated.astype(np.float32)


def _fit_preview_size(width: int, height: int) -> tuple[int, int]:
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    scale = min(PREVIEW_MAX_WIDTH / src_w, PREVIEW_MAX_HEIGHT / src_h, 1.0)
    target_w = max(2, int(round(src_w * scale)))
    target_h = max(2, int(round(src_h * scale)))
    if target_w % 2 == 1:
        target_w -= 1
    if target_h % 2 == 1:
        target_h -= 1
    return max(2, target_w), max(2, target_h)


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
    fitted_width = float(span[0]) * scale
    fitted_height = float(span[1]) * scale
    offset_x = (float(width) - fitted_width) * 0.5
    offset_y = (float(height) - fitted_height) * 0.5

    mapped = np.empty((points.shape[0], 2), dtype=np.float32)
    mapped[:, 0] = (points[:, 0] - float(min_xy[0])) * scale + offset_x
    mapped[:, 1] = (points[:, 1] - float(min_xy[1])) * scale + offset_y
    mapped[:, 1] = float(height) - mapped[:, 1]
    return mapped


def _smooth_sequence(points: np.ndarray, alpha: float = TEMPORAL_SMOOTH_ALPHA) -> np.ndarray:
    if points.ndim != 3 or points.shape[0] <= 1:
        return points
    smoothed = points.copy()
    for frame_idx in range(1, points.shape[0]):
        smoothed[frame_idx] = alpha * smoothed[frame_idx - 1] + (1.0 - alpha) * points[frame_idx]
    return smoothed


def build_pose3d_preview_data(
    joints3d: np.ndarray,
    fps: float,
    frame_total: int | None = None,
) -> dict[str, object]:
    if joints3d.ndim != 3 or joints3d.shape[-1] != 3:
        raise ValueError("joints3d 形状必须为 [T, J, 3]")

    available_frames = int(joints3d.shape[0])
    use_frames = (
        available_frames
        if frame_total is None
        else min(available_frames, max(0, int(frame_total)))
    )
    if use_frames <= 0:
        raise ValueError("joints3d 为空，无法导出 3D 预览数据")

    joints3d_use = _smooth_sequence(joints3d[:use_frames].astype(np.float32))
    root = (joints3d_use[:, ROOT_LEFT_HIP, :] + joints3d_use[:, ROOT_RIGHT_HIP, :]) * 0.5
    centered = joints3d_use - root[:, None, :]
    flat_points = centered.reshape(-1, 3)
    min_xyz = np.min(flat_points, axis=0).astype(np.float32)
    max_xyz = np.max(flat_points, axis=0).astype(np.float32)
    max_radius = float(np.max(np.linalg.norm(flat_points, axis=1)))

    rounded = np.round(centered, POSE3D_JSON_DECIMALS).astype(np.float32)
    return {
        "preview_version": POSE3D_PREVIEW_VERSION,
        "fps": float(max(1.0, fps)),
        "frame_count": int(use_frames),
        "joint_count": int(joints3d_use.shape[1]),
        "edges": [[int(a), int(b)] for a, b in SKELETON_EDGES],
        "bounds": {
            "min": rounded_triplet(min_xyz),
            "max": rounded_triplet(max_xyz),
            "floor_y": round(float(min_xyz[1]), POSE3D_JSON_DECIMALS),
            "max_radius": round(max_radius, POSE3D_JSON_DECIMALS),
            "default_yaw_deg": YAW_DEG,
            "default_pitch_deg": PITCH_DEG,
        },
        "joints3d": rounded.tolist(),
    }


def rounded_triplet(values: np.ndarray) -> list[float]:
    return [round(float(value), POSE3D_JSON_DECIMALS) for value in values[:3]]


def build_pose2d_preview_data(
    keypoints2d: np.ndarray,
    fps: float,
    frame_width: int,
    frame_height: int,
    frame_total: int | None = None,
) -> dict[str, object]:
    if keypoints2d.ndim != 3 or keypoints2d.shape[-1] < 2:
        raise ValueError("keypoints2d 形状必须为 [T, J, C]，且至少包含 xy 坐标")

    available_frames = int(keypoints2d.shape[0])
    use_frames = (
        available_frames
        if frame_total is None
        else min(available_frames, max(0, int(frame_total)))
    )
    if use_frames <= 0:
        raise ValueError("keypoints2d 为空，无法导出 2D 预览数据")

    keypoints_use = keypoints2d[:use_frames].astype(np.float32).copy()
    if keypoints_use.shape[-1] >= 3:
        keypoints_use[..., 2] = np.clip(keypoints_use[..., 2], 0.0, 1.0)
    rounded = np.round(keypoints_use, POSE2D_JSON_DECIMALS).astype(np.float32)
    return {
        "fps": float(max(1.0, fps)),
        "frame_count": int(use_frames),
        "joint_count": int(keypoints_use.shape[1]),
        "frame_width": int(max(1, frame_width)),
        "frame_height": int(max(1, frame_height)),
        "edges": [[int(a), int(b)] for a, b in SKELETON_EDGES],
        "keypoints2d": rounded.tolist(),
    }


def _draw_scene_grid(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    horizon = int(height * GRID_TOP_RATIO)
    bottom = int(height * GRID_BOTTOM_RATIO)
    left = int(width * GRID_LEFT_RATIO)
    right = int(width * GRID_RIGHT_RATIO)
    center_x = int((left + right) * 0.5)

    polygon = np.array(
        [
            [left, bottom],
            [right, bottom],
            [int(width * 0.72), horizon],
            [int(width * 0.28), horizon],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, polygon, (234, 234, 234), lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [polygon], True, (202, 202, 202), 1, lineType=cv2.LINE_AA)

    for idx in range(1, 8):
        x = int(left + (right - left) * idx / 8)
        cv2.line(canvas, (x, bottom), (center_x, horizon), (212, 212, 212), 1, lineType=cv2.LINE_AA)
    for idx in range(1, 5):
        ratio = idx / 5.0
        y = int(bottom - (bottom - horizon) * ratio)
        line_left = int(left + (center_x - left) * ratio)
        line_right = int(right - (right - center_x) * ratio)
        cv2.line(canvas, (line_left, y), (line_right, y), (218, 218, 218), 1, lineType=cv2.LINE_AA)


def _draw_axis_gizmo(canvas: np.ndarray) -> None:
    rot_y, rot_x = _build_rotation_matrices()
    axis_points = np.eye(3, dtype=np.float32) @ rot_y.T @ rot_x.T
    axis_colors = {
        "X": (74, 106, 224),
        "Y": (92, 166, 96),
        "Z": (214, 132, 68),
    }
    height, width = canvas.shape[:2]
    panel_top_left = (24, 20)
    panel_bottom_right = (152, 146)
    origin = np.array([72.0, 114.0], dtype=np.float32)

    cv2.rectangle(
        canvas,
        panel_top_left,
        panel_bottom_right,
        (246, 246, 246),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        panel_top_left,
        panel_bottom_right,
        (214, 214, 214),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "XYZ",
        (36, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (44, 44, 44),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "view",
        (92, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (116, 116, 116),
        1,
        lineType=cv2.LINE_AA,
    )

    for label, axis_vector in zip(("X", "Y", "Z"), axis_points, strict=True):
        direction = np.array([axis_vector[0], -axis_vector[1]], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        screen_dir = direction / norm
        end = origin + screen_dir * AXIS_LENGTH_PX
        start_xy = tuple(np.round(origin).astype(int).tolist())
        end_xy = tuple(np.round(end).astype(int).tolist())
        color = axis_colors[label]
        cv2.arrowedLine(canvas, start_xy, end_xy, color, 2, tipLength=0.16, line_type=cv2.LINE_AA)
        label_xy = tuple(np.round(end + screen_dir * 10.0).astype(int).tolist())
        cv2.putText(
            canvas,
            label,
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    cv2.circle(
        canvas,
        tuple(np.round(origin).astype(int).tolist()),
        4,
        (80, 80, 80),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "X lateral  Y vertical  Z depth",
        (28, min(height - 18, 138)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (104, 104, 104),
        1,
        lineType=cv2.LINE_AA,
    )


def _decorate_source_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (16, 16), (244, 56), (249, 249, 249), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (16, 16), (244, 56), (216, 216, 216), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Source frame={frame_idx}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (36, 36, 36),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


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
    _draw_axis_gizmo(canvas)

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
            int(68 + 88 * (1.0 - z_norm)),
            int(122 + 64 * (1.0 - abs(z_norm - 0.5) * 2.0)),
            int(188 + 36 * z_norm),
        )
        pa = tuple(np.round(points_xy[a]).astype(int).tolist())
        pb = tuple(np.round(points_xy[b]).astype(int).tolist())
        cv2.line(canvas, pa, pb, (220, 220, 220), 6, lineType=cv2.LINE_AA)
        cv2.line(canvas, pa, pb, color, 3, lineType=cv2.LINE_AA)

    for idx in range(points_xy.shape[0]):
        z_norm = (float(depth[idx]) - depth_min) / depth_span
        color = (
            int(72 + 82 * (1.0 - z_norm)),
            int(126 + 60 * (1.0 - abs(z_norm - 0.5) * 2.0)),
            int(198 + 32 * z_norm),
        )
        point_xy = tuple(np.round(points_xy[idx]).astype(int).tolist())
        cv2.circle(canvas, point_xy, 6, (230, 230, 230), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, point_xy, 4, color, -1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (20, 20), (312, 56), (248, 248, 248), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (20, 20), (312, 56), (216, 216, 216), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"3D Skeleton fusion   frame={frame_idx}",
        (32, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
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
    output_source: Path | None = None,
) -> dict[str, float]:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
    target_width, target_height = _fit_preview_size(source_width, source_height)
    scale_x = target_width / max(1, source_width)
    scale_y = target_height / max(1, source_height)

    frame_total = int(min(len(keypoints2d), len(joints3d)))
    if frame_total <= 0:
        cap.release()
        raise RuntimeError("视频与关键点帧数不匹配，无法生成预览。")

    joints3d_use = _smooth_sequence(joints3d[:frame_total].astype(np.float32))
    projected_all = np.stack(
        [_project_frame(joints3d_use[idx]) for idx in range(frame_total)],
        axis=0,
    )
    xy_all = projected_all[:, :, :2]
    frame_min = np.min(xy_all, axis=1)
    frame_max = np.max(xy_all, axis=1)
    frame_center = (frame_min + frame_max) * 0.5
    frame_span = np.maximum(frame_max - frame_min, 1e-4)
    center_xy = np.median(frame_center, axis=0).astype(np.float32)
    span_xy = np.percentile(frame_span, 75, axis=0).astype(np.float32)
    span_xy = np.maximum(span_xy * 1.35, np.array([0.45, 0.72], dtype=np.float32))
    min_xy = center_xy - span_xy * 0.5
    max_xy = center_xy + span_xy * 0.5

    frames_source: list[np.ndarray] = []
    frames_2d: list[np.ndarray] = []
    frames_3d: list[np.ndarray] = []
    for frame_idx in range(frame_total):
        ok, frame = cap.read()
        if not ok:
            break

        frame_resized = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        if output_source is not None:
            frames_source.append(_decorate_source_frame(frame_resized, frame_idx))

        kp = keypoints2d[frame_idx].astype(np.float32).copy()
        kp[:, 0] *= scale_x
        kp[:, 1] *= scale_y
        frame_2d = draw_pose_2d(frame_resized.copy(), kp, conf_thres=0.05)
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
            width=target_width,
            height=target_height,
            frame_idx=frame_idx,
        )
        frames_3d.append(frame_3d)

    cap.release()
    if output_source is not None:
        _write_browser_mp4(output_source, frames_source, fps)
    _write_browser_mp4(output_2d, frames_2d, fps)
    _write_browser_mp4(output_3d, frames_3d, fps)
    return {"fps": fps, "frames": float(len(frames_2d))}
