from __future__ import annotations

from pathlib import Path

import numpy as np

from extract_pose_aist2d import _resolve_source_video_name, _select_main_person


def test_select_main_person_choose_best_camera_by_det_scores() -> None:
    keypoints = np.zeros((9, 12, 17, 3), dtype=np.float32)
    det_scores = np.full((9, 12), 0.1, dtype=np.float32)
    det_scores[4, :] = 0.9
    data = {"keypoints2d": keypoints, "det_scores": det_scores}

    selected, camera_id = _select_main_person(
        data=data,
        file_path=Path("demo.pkl"),
        preferred_camera_idx=None,
    )
    assert selected.shape == (12, 17, 3)
    assert camera_id == "c05"


def test_select_main_person_respects_preferred_camera() -> None:
    keypoints = np.zeros((9, 8, 17, 3), dtype=np.float32)
    det_scores = np.full((9, 8), 0.3, dtype=np.float32)
    det_scores[6, :] = 0.95
    data = {"keypoints2d": keypoints, "det_scores": det_scores}

    _, camera_id = _select_main_person(
        data=data,
        file_path=Path("demo.pkl"),
        preferred_camera_idx=1,
    )
    assert camera_id == "c02"


def test_resolve_source_video_name_from_call() -> None:
    source = _resolve_source_video_name("gBR_sBM_cAll_d04_mBR0_ch01", "c07")
    assert source == "gBR_sBM_c07_d04_mBR0_ch01.mp4"

