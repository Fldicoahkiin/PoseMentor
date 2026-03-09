from __future__ import annotations

from pathlib import Path, PurePosixPath


def build_seq_id_from_rel_path(rel_path: str) -> str:
    rel = PurePosixPath(rel_path)
    return "__".join(rel.with_suffix("").parts)


def build_video_seq_id(video_root: Path, video_path: Path) -> str:
    try:
        rel = video_path.relative_to(video_root).as_posix()
    except ValueError:
        rel = video_path.name
    return build_seq_id_from_rel_path(rel)


def build_video_rel_path(video_root: Path, video_path: Path) -> str:
    try:
        return video_path.relative_to(video_root).as_posix()
    except ValueError:
        return video_path.name
