from __future__ import annotations

from pathlib import Path

from posementor.multiview.alignment import VideoStats, compute_offsets


def test_compute_offsets() -> None:
    stats = [
        VideoStats(Path("front.mp4"), fps=30.0, frames=300, width=1280, height=720, motion_start=10),
        VideoStats(Path("left.mp4"), fps=30.0, frames=300, width=1280, height=720, motion_start=25),
        VideoStats(Path("right.mp4"), fps=30.0, frames=300, width=1280, height=720, motion_start=13),
        VideoStats(Path("back.mp4"), fps=30.0, frames=300, width=1280, height=720, motion_start=10),
    ]

    offsets = compute_offsets(stats)
    assert offsets == {
        "front": 0,
        "left": 15,
        "right": 3,
        "back": 0,
    }
