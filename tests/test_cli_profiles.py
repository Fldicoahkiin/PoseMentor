from __future__ import annotations

from pathlib import Path

from posementor.cli import _inspect_aist_videos, _profile_satisfied


def test_inspect_aist_videos_collects_group_and_camera(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    (video_root / "gBR_sBM_c01_d04_mBR0_ch01.mp4").write_bytes(b"1")
    (video_root / "gBR_sBM_c02_d04_mBR0_ch01.mp4").write_bytes(b"1")
    (video_root / "gBR_sBM_c03_d04_mBR0_ch01.mp4").write_bytes(b"1")
    stats = _inspect_aist_videos(video_root)
    assert stats["file_count"] == 3
    assert stats["group_count"] == 1
    assert stats["camera_ids"] == ["c01", "c02", "c03"]


def test_profile_satisfied_requires_cameras_and_group_count() -> None:
    stats = {"camera_ids": ["c01", "c02", "c03", "c04", "c05"], "group_count": 80}
    assert _profile_satisfied(stats, "mv5_standard")
    assert not _profile_satisfied(stats, "mv9_core")
