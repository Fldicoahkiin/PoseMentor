#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from posementor.multiview.alignment import analyze_videos, compute_offsets
from posementor.multiview.formatter import SyncSpec, sync_and_export_session, write_multiview_manifest
from posementor.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四机位视频对齐、处理与格式化")
    parser.add_argument("--config", type=Path, default=Path("configs/multiview.yaml"))
    parser.add_argument("--limit-sessions", type=int, default=0, help="仅处理前 N 个 session")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_sessions(input_root: Path, camera_files: list[str]) -> list[Path]:
    sessions: list[Path] = []
    for session_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        if all((session_dir / cam).exists() for cam in camera_files):
            sessions.append(session_dir)
    return sessions


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_root = Path(cfg["input_root"])
    output_root = ensure_dir(Path(cfg["output_root"]))
    camera_files = [str(x) for x in cfg["camera_files"]]

    if not input_root.exists():
        raise FileNotFoundError(f"找不到输入目录: {input_root}")

    sessions = discover_sessions(input_root, camera_files)
    if args.limit_sessions > 0:
        sessions = sessions[: args.limit_sessions]

    if not sessions:
        raise RuntimeError(
            "未发现有效 session。请确认每个 session 目录内包含配置中的 4 个 camera_files。"
        )

    align_cfg = cfg["alignment"]
    format_cfg = cfg["format"]
    spec = SyncSpec(
        target_fps=float(format_cfg["target_fps"]),
        target_width=int(format_cfg["target_width"]),
        target_height=int(format_cfg["target_height"]),
        max_frames=int(format_cfg["max_frames"]),
    )

    manifest_rows: list[dict[str, object]] = []

    for idx, session_dir in enumerate(sessions, start=1):
        session_name = session_dir.name
        video_paths = [session_dir / cam for cam in camera_files]

        print(f"[INFO] ({idx}/{len(sessions)}) 分析 session: {session_name}")
        stats = analyze_videos(
            video_paths=video_paths,
            scan_frames=int(align_cfg["scan_frames"]),
            motion_ratio=float(align_cfg["motion_ratio"]),
        )

        offsets = compute_offsets(stats)
        out_dir = ensure_dir(output_root / session_name)

        meta = sync_and_export_session(
            session_name=session_name,
            input_paths=video_paths,
            offsets=offsets,
            output_dir=out_dir,
            spec=spec,
        )
        manifest_rows.append(meta)
        print(f"[DONE] {session_name}: frames={meta['frames']} offsets={meta['offsets']}")

    write_multiview_manifest(output_root=output_root, rows=manifest_rows)
    print(f"[DONE] 清单输出: {output_root / 'multiview_manifest.csv'}")


if __name__ == "__main__":
    main()
