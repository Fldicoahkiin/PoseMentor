from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoStats:
    path: Path
    fps: float
    frames: int
    width: int
    height: int
    motion_start: int


def _read_video_meta(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frames, width, height


def detect_motion_start(path: Path, scan_frames: int = 300, motion_ratio: float = 2.8) -> int:
    """基于帧差能量估计动作起始帧，用于多机位粗对齐。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")

    prev_gray = None
    scores: list[float] = []

    frame_idx = 0
    while frame_idx < scan_frames:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
            scores.append(score)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if len(scores) < 6:
        return 0

    base = float(np.median(scores[: min(30, len(scores))]))
    threshold = max(2.0, base * motion_ratio)

    for idx, score in enumerate(scores):
        if score >= threshold:
            return idx
    return 0


def analyze_videos(video_paths: list[Path], scan_frames: int, motion_ratio: float) -> list[VideoStats]:
    stats: list[VideoStats] = []
    for path in video_paths:
        fps, frames, width, height = _read_video_meta(path)
        motion_start = detect_motion_start(path, scan_frames=scan_frames, motion_ratio=motion_ratio)
        stats.append(
            VideoStats(
                path=path,
                fps=fps,
                frames=frames,
                width=width,
                height=height,
                motion_start=motion_start,
            )
        )
    return stats


def compute_offsets(stats: list[VideoStats]) -> dict[str, int]:
    min_start = min(item.motion_start for item in stats)
    return {item.path.stem: int(item.motion_start - min_start) for item in stats}
