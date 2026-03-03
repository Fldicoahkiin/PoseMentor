from __future__ import annotations

import numpy as np

from posementor.data.aist_dataset import AISTLiftDataset, SequencePair, compute_2d_norm_stats


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

