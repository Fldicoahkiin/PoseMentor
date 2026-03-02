from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from posementor.utils.io import ensure_dir, write_csv


@dataclass(slots=True)
class SyncSpec:
    target_fps: float
    target_width: int
    target_height: int
    max_frames: int


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    return cap


def sync_and_export_session(
    session_name: str,
    input_paths: list[Path],
    offsets: dict[str, int],
    output_dir: Path,
    spec: SyncSpec,
) -> dict[str, object]:
    """将四机位视频按 offset 对齐后，导出统一尺寸/帧率的视频。"""
    ensure_dir(output_dir)

    caps = [_open_capture(path) for path in input_paths]
    try:
        available_lengths = []
        for cap, path in zip(caps, input_paths, strict=False):
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            available = total - offsets[path.stem]
            available_lengths.append(max(0, available))

        valid_len = min(available_lengths)
        if spec.max_frames > 0:
            valid_len = min(valid_len, spec.max_frames)

        if valid_len <= 0:
            raise RuntimeError(f"session={session_name} 无可对齐帧")

        out_files: list[str] = []

        for cap, path in zip(caps, input_paths, strict=False):
            offset = offsets[path.stem]
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(offset))

            out_path = output_dir / f"{path.stem}.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                spec.target_fps,
                (spec.target_width, spec.target_height),
            )

            for _ in range(valid_len):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(
                    frame,
                    (spec.target_width, spec.target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                writer.write(frame)
            writer.release()
            out_files.append(str(out_path))

        meta = {
            "session": session_name,
            "frames": valid_len,
            "target_fps": spec.target_fps,
            "target_size": [spec.target_width, spec.target_height],
            "offsets": offsets,
            "videos": out_files,
        }

        (output_dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta
    finally:
        for cap in caps:
            cap.release()


def write_multiview_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest = output_root / "multiview_manifest.csv"
    fieldnames = ["session", "frames", "target_fps", "offsets", "videos"]

    serialized_rows: list[dict[str, object]] = []
    for row in rows:
        serialized_rows.append(
            {
                "session": row["session"],
                "frames": row["frames"],
                "target_fps": row["target_fps"],
                "offsets": json.dumps(row["offsets"], ensure_ascii=False),
                "videos": json.dumps(row["videos"], ensure_ascii=False),
            }
        )

    write_csv(manifest, rows=serialized_rows, fieldnames=fieldnames)
