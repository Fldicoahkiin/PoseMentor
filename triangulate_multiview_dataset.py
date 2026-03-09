#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posementor.multiview.calibration import load_calibration_rig
from posementor.multiview.triangulation import (
    export_session_gt3d,
    load_session_tracks,
    load_triangulation_config,
    triangulate_session_tracks,
    write_triangulation_manifest,
)
from posementor.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于标定参数将四机位 2D 关键点三角化为 3D 真值"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/multiview.yaml"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="覆盖配置中的 calibration_file",
    )
    parser.add_argument("--limit-sessions", type=int, default=0, help="仅处理前 N 个 session")
    return parser.parse_args()


def _resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path



def _discover_sessions(output_root: Path) -> list[Path]:
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "session_meta.json").exists()
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return payload



def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    base_dir = args.config.resolve().parent.parent

    output_root = _resolve_path(
        str(cfg.get("output_root", "data/processed/multiview")),
        base=base_dir,
    )
    processed_root = _resolve_path(str(cfg.get("processed_root", output_root)), base=base_dir)
    pose2d_root = _resolve_path(
        str(cfg.get("pose2d_root", processed_root / "yolo2d")),
        base=base_dir,
    )
    gt3d_root = _resolve_path(str(cfg.get("gt3d_root", processed_root / "gt3d")), base=base_dir)

    calibration_text = str(args.calibration or cfg.get("calibration_file", "")).strip()
    if not calibration_text:
        raise ValueError("请提供 --calibration 或在 configs/multiview.yaml 中设置 calibration_file")
    calibration_file = _resolve_path(calibration_text, base=base_dir)
    if not calibration_file.exists():
        raise FileNotFoundError(f"标定文件不存在: {calibration_file}")

    camera_files = [str(item) for item in cfg.get("camera_files", [])]
    if not camera_files:
        raise ValueError("multiview 配置缺少 camera_files")

    sessions = _discover_sessions(output_root)
    if args.limit_sessions > 0:
        sessions = sessions[: args.limit_sessions]
    if not sessions:
        raise RuntimeError(f"未在 {output_root} 发现已格式化的 session")

    rig = load_calibration_rig(calibration_file)
    tri_config = load_triangulation_config(args.config)
    ensure_dir(gt3d_root)

    manifest_rows: list[dict[str, Any]] = []
    for index, session_dir in enumerate(sessions, start=1):
        session_name = session_dir.name
        print(f"[INFO] ({index}/{len(sessions)}) 三角化 session: {session_name}")
        tracks = load_session_tracks(
            pose2d_root=pose2d_root,
            session_name=session_name,
            camera_files=camera_files,
        )
        joints3d, details = triangulate_session_tracks(tracks=tracks, rig=rig, config=tri_config)
        seq_ids = export_session_gt3d(
            gt3d_root=gt3d_root,
            session_name=session_name,
            camera_files=camera_files,
            joints3d=joints3d,
            details=details,
        )
        summary = details["summary"]
        summary_payload = {
            "session": session_name,
            "calibration": rig.name,
            **summary,
            "seq_ids": seq_ids,
        }
        (gt3d_root / f"{session_name}__summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "session": session_name,
                "frames": summary["frame_count"],
                "joint_count": summary["joint_count"],
                "valid_joint_ratio": summary["valid_joint_ratio"],
                "mean_reprojection_error_px": summary["mean_reprojection_error_px"],
                "seq_ids": json.dumps(seq_ids, ensure_ascii=False),
            }
        )
        print(
            f"[DONE] {session_name}: valid_ratio={summary['valid_joint_ratio']} "
            f"reproj={summary['mean_reprojection_error_px']}"
        )

    write_triangulation_manifest(gt3d_root, manifest_rows)
    print(f"[DONE] 3D 真值输出目录: {gt3d_root}")


if __name__ == "__main__":
    main()
