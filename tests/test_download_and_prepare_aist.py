from __future__ import annotations

from download_and_prepare_aist import _parse_camera_ids, _select_video_stems


def test_parse_camera_ids_normalizes_and_deduplicates() -> None:
    assert _parse_camera_ids("c1,c01, 02, c03,invalid") == ["c01", "c02", "c03"]


def test_select_video_stems_requires_multiview_groups() -> None:
    video_ids = [
        "gBR_sBM_c01_d04_mBR0_ch01.mp4",
        "gBR_sBM_c02_d04_mBR0_ch01.mp4",
        "gBR_sBM_c03_d04_mBR0_ch01.mp4",
        "gPO_sBM_c01_d04_mPO0_ch01.mp4",
        "gPO_sBM_c02_d04_mPO0_ch01.mp4",
    ]

    stems = _select_video_stems(
        video_ids=video_ids,
        group_limit=0,
        video_limit=0,
        camera_ids=["c01", "c02", "c03"],
        min_cameras_per_group=3,
    )
    assert stems == [
        "gBR_sBM_c01_d04_mBR0_ch01",
        "gBR_sBM_c02_d04_mBR0_ch01",
        "gBR_sBM_c03_d04_mBR0_ch01",
    ]


def test_select_video_stems_respects_group_limit() -> None:
    video_ids = [
        "gAA_sBM_c01_d01_mAA0_ch01.mp4",
        "gAA_sBM_c02_d01_mAA0_ch01.mp4",
        "gAA_sBM_c03_d01_mAA0_ch01.mp4",
        "gBB_sBM_c01_d01_mBB0_ch01.mp4",
        "gBB_sBM_c02_d01_mBB0_ch01.mp4",
        "gBB_sBM_c03_d01_mBB0_ch01.mp4",
    ]
    stems = _select_video_stems(
        video_ids=video_ids,
        group_limit=1,
        video_limit=0,
        camera_ids=["c01", "c02", "c03"],
        min_cameras_per_group=3,
    )
    assert len(stems) == 3
