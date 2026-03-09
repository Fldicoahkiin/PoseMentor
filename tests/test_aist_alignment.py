from __future__ import annotations

import numpy as np

from posementor.data.aist_alignment import (
    _search_best_shift,
    extract_camera_id,
    plan_group_preview,
    resolve_group_seq_id,
)


def test_resolve_group_seq_id_and_camera_id() -> None:
    stem = "gBR_sBM_c05_d04_mBR0_ch01"
    assert resolve_group_seq_id(stem) == "gBR_sBM_cAll_d04_mBR0_ch01"
    assert extract_camera_id(stem) == "c05"



def test_search_best_shift_matches_known_offset() -> None:
    frame_count = 180
    joint_count = 17
    projected_xy = np.zeros((frame_count, joint_count, 2), dtype=np.float32)
    for frame_idx in range(frame_count):
        for joint_idx in range(joint_count):
            projected_xy[frame_idx, joint_idx, 0] = frame_idx * 1.8 + joint_idx * 2.0
            projected_xy[frame_idx, joint_idx, 1] = frame_idx * 0.9 + joint_idx * 1.5

    shift = 12
    observed_xy = projected_xy.copy()
    observed_xy[shift:] = projected_xy[:-shift]
    observed_xy[:shift] = projected_xy[0]
    observed_conf = np.ones((frame_count, joint_count), dtype=np.float32)

    best_shift, best_error = _search_best_shift(
        projected_xy=projected_xy,
        observed_xy=observed_xy,
        observed_conf=observed_conf,
    )
    assert best_shift == shift
    assert best_error < 1e-4



def test_plan_group_preview_uses_camera_offsets() -> None:
    alignment = {
        "group_seq_id": "gBR_sBM_cAll_d04_mBR0_ch01",
        "mode": "aist_official_projection",
        "setting_name": "setting1",
        "setting_file": "setting1.json",
        "timeline_fps": 59.9988,
        "timeline_frame_count": 720,
        "camera_offsets": {"c01": 0, "c02": 1, "c05": 91},
        "camera_sync_error_px": {"c01": 3.0, "c02": 3.1, "c05": 25.4},
        "camera_frame_count": {"c01": 720, "c02": 720, "c05": 720},
        "camera_geometry": {
            "c01": {"camera_id": "c01"},
            "c02": {"camera_id": "c02"},
            "c05": {"camera_id": "c05"},
        },
    }
    video_stats = {
        "c01": {"frames": 720, "fps": 60.0, "width": 1920, "height": 1080},
        "c02": {"frames": 720, "fps": 60.0, "width": 1920, "height": 1080},
        "c05": {"frames": 720, "fps": 60.0, "width": 1920, "height": 1080},
    }

    plan = plan_group_preview(
        alignment=alignment,
        video_stats=video_stats,
        joints3d_frame_total=720,
    )
    assert plan["timeline_start_frame"] == 0
    assert plan["frame_total"] == 629
    assert plan["camera_trim_start"] == {"c01": 0, "c02": 1, "c05": 91}
    assert plan["available_cameras"] == ["c01", "c02", "c05"]
