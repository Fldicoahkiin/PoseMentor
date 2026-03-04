from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class SequencePair:
    seq_id: str
    style: str
    kp2d: np.ndarray  # [T, 17, 3]
    gt3d: np.ndarray  # [T, 17, 3]
    video_path: Path | None = None


class AISTLiftDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        pairs: list[SequencePair],
        seq_len: int,
        sample_stride: int,
        mean_2d: np.ndarray,
        std_2d: np.ndarray,
    ) -> None:
        self.pairs = pairs
        self.seq_len = seq_len
        self.sample_stride = sample_stride
        self.mean_2d = mean_2d.astype(np.float32)
        self.std_2d = std_2d.astype(np.float32)
        self.indices: list[tuple[int, int]] = []

        for pair_idx, pair in enumerate(self.pairs):
            total = pair.kp2d.shape[0]
            if total < self.seq_len:
                continue
            max_start = total - self.seq_len
            for st in range(0, max_start + 1, self.sample_stride):
                self.indices.append((pair_idx, st))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair_idx, st = self.indices[index]
        pair = self.pairs[pair_idx]

        kp2d = pair.kp2d[st : st + self.seq_len, :, :2].astype(np.float32)
        conf = pair.kp2d[st : st + self.seq_len, :, 2:3].astype(np.float32)
        gt3d_raw = pair.gt3d[st : st + self.seq_len].astype(np.float32)

        valid3d = np.isfinite(gt3d_raw).all(axis=-1, keepdims=True).astype(np.float32)
        gt3d = np.nan_to_num(gt3d_raw, nan=0.0, posinf=0.0, neginf=0.0)
        kp2d = np.nan_to_num(kp2d, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.clip(conf, 0.0, 1.0) * valid3d

        kp2d = (kp2d - self.mean_2d) / self.std_2d

        return {
            "kp2d": torch.from_numpy(kp2d),
            "conf": torch.from_numpy(conf),
            "gt3d": torch.from_numpy(gt3d),
            "seq_id": pair.seq_id,
            "start_idx": torch.tensor(st, dtype=torch.int64),
            "video_path": str(pair.video_path) if pair.video_path is not None else "",
        }


def _split_by_hash(seq_id: str, val_ratio: float) -> str:
    digest = hashlib.md5(seq_id.encode("utf-8")).hexdigest()  # noqa: S324
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def load_sequence_pairs(
    yolo_dir: Path,
    gt_dir: Path,
    val_ratio: float,
    split: str,
    videos_root: Path | None = None,
) -> list[SequencePair]:
    assert split in {"train", "val"}
    pairs: list[SequencePair] = []

    yolo_files = {f.stem: f for f in yolo_dir.glob("*.npz")}
    gt_files = {f.stem: f for f in gt_dir.glob("*.npz")}

    common = sorted(set(yolo_files) & set(gt_files))
    for seq_id in common:
        if _split_by_hash(seq_id=seq_id, val_ratio=val_ratio) != split:
            continue

        with np.load(yolo_files[seq_id]) as k_data:
            kp2d = k_data["keypoints2d"].astype(np.float32)
            if "style" in k_data.files:
                style_val = k_data["style"]
                style = str(style_val.item() if np.ndim(style_val) == 0 else style_val)
            else:
                style = "unknown"
        with np.load(gt_files[seq_id]) as g_data:
            gt3d = g_data["joints3d"].astype(np.float32)

        video_path: Path | None = None
        if videos_root is not None:
            candidate = videos_root / f"{seq_id}.mp4"
            if candidate.exists():
                video_path = candidate

        frames = min(len(kp2d), len(gt3d))
        if frames < 24:
            continue

        pairs.append(
            SequencePair(
                seq_id=seq_id,
                style=style,
                kp2d=kp2d[:frames],
                gt3d=gt3d[:frames],
                video_path=video_path,
            )
        )

    return pairs


def compute_2d_norm_stats(pairs: list[SequencePair]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        mean_2d = np.zeros((1, 1, 2), dtype=np.float32)
        std_2d = np.ones((1, 1, 2), dtype=np.float32)
        return mean_2d, std_2d

    all_points = np.concatenate([pair.kp2d[:, :, :2] for pair in pairs], axis=0).astype(np.float32)
    mean_2d = np.nanmean(all_points, axis=(0, 1), keepdims=True)
    std_2d = np.nanstd(all_points, axis=(0, 1), keepdims=True)

    mean_2d = np.nan_to_num(mean_2d, nan=0.0, posinf=0.0, neginf=0.0)
    std_2d = np.nan_to_num(std_2d, nan=1.0, posinf=1.0, neginf=1.0)
    std_2d = np.clip(std_2d, 1e-6, None)
    return mean_2d.astype(np.float32), std_2d.astype(np.float32)
