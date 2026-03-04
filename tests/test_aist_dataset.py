from __future__ import annotations

import numpy as np
from pathlib import Path

from posementor.data.aist_dataset import (
    AISTLiftDataset,
    SequencePair,
    _resolve_video_candidate,
    compute_2d_norm_stats,
)


def test_compute_2d_norm_stats_handles_nan() -> None:
    kp2d = np.zeros((4, 17, 3), dtype=np.float32)
    kp2d[..., 0] = 100.0
    kp2d[..., 1] = 200.0
    kp2d[0, 0, 0] = np.nan
    gt3d = np.zeros((4, 17, 3), dtype=np.float32)

    pairs = [SequencePair(seq_id="gBR_x", style="gBR", kp2d=kp2d, gt3d=gt3d)]
    mean_2d, std_2d = compute_2d_norm_stats(pairs)

    assert np.isfinite(mean_2d).all()
    assert np.isfinite(std_2d).all()
    assert float(std_2d.min()) > 0.0


def test_dataset_masks_invalid_3d_points() -> None:
    kp2d = np.zeros((100, 17, 3), dtype=np.float32)
    kp2d[..., 2] = 1.0
    gt3d = np.zeros((100, 17, 3), dtype=np.float32)
    gt3d[10, 5, :] = np.nan

    pair = SequencePair(seq_id="gPO_x", style="gPO", kp2d=kp2d, gt3d=gt3d)
    mean_2d = np.zeros((1, 1, 2), dtype=np.float32)
    std_2d = np.ones((1, 1, 2), dtype=np.float32)

    ds = AISTLiftDataset(
        pairs=[pair],
        seq_len=81,
        sample_stride=10,
        mean_2d=mean_2d,
        std_2d=std_2d,
    )
    sample = ds[0]

    assert np.isfinite(sample["gt3d"].numpy()).all()
    assert np.isfinite(sample["kp2d"].numpy()).all()
    assert (sample["conf"].numpy() >= 0.0).all()
    assert (sample["conf"].numpy() <= 1.0).all()


def test_resolve_video_candidate_prefers_source_video_name(tmp_path: Path) -> None:
    videos_root = tmp_path / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    explicit_video = videos_root / "explicit.mp4"
    explicit_video.write_bytes(b"demo")
    fallback_video = videos_root / "gBR_sBM_c01_d04_mBR0_ch01.mp4"
    fallback_video.write_bytes(b"demo")

    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(
        npz_path,
        keypoints2d=np.zeros((10, 17, 3), dtype=np.float32),
        source_video_name=np.array("explicit.mp4"),
        camera_id=np.array("c01"),
    )
    with np.load(npz_path) as data:
        resolved = _resolve_video_candidate(
            videos_root=videos_root,
            seq_id="gBR_sBM_cAll_d04_mBR0_ch01",
            k_data=data,
        )
    assert resolved == explicit_video


def test_resolve_video_candidate_supports_call_replacement(tmp_path: Path) -> None:
    videos_root = tmp_path / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    replaced_video = videos_root / "gBR_sBM_c03_d04_mBR0_ch01.mp4"
    replaced_video.write_bytes(b"demo")

    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(
        npz_path,
        keypoints2d=np.zeros((10, 17, 3), dtype=np.float32),
        camera_id=np.array("c03"),
    )
    with np.load(npz_path) as data:
        resolved = _resolve_video_candidate(
            videos_root=videos_root,
            seq_id="gBR_sBM_cAll_d04_mBR0_ch01",
            k_data=data,
        )
    assert resolved == replaced_video
