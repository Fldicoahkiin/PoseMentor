#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from posementor.utils.io import ensure_dir, load_pickle_or_npz, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 AIST++ 官方 2D 注释提取训练输入关键点")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--max-files", type=int, default=0, help="仅处理前 N 个文件，0 表示全部")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser.parse_args()


def _infer_fps(timestamps: np.ndarray, default_fps: float = 60.0) -> float:
    if timestamps.size < 2:
        return default_fps
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return default_fps
    median_delta = float(np.median(diffs))
    # AIST++ timestamps 是微秒单位。
    fps = 1_000_000.0 / median_delta
    if not np.isfinite(fps) or fps <= 1.0:
        return default_fps
    return fps


def _select_main_person(data: dict[str, np.ndarray], file_path: Path) -> np.ndarray:
    keypoints = np.asarray(data.get("keypoints2d"))
    if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
        return keypoints.astype(np.float32)

    if keypoints.ndim != 4 or keypoints.shape[-1] != 3:
        raise ValueError(f"不支持的 keypoints2d 形状: {keypoints.shape} ({file_path.name})")

    person_num, frame_num, _, _ = keypoints.shape
    if person_num == 1:
        return keypoints[0].astype(np.float32)

    det_scores = np.asarray(data.get("det_scores")) if data.get("det_scores") is not None else None
    if det_scores is not None and det_scores.shape == (person_num, frame_num):
        best_person = np.argmax(det_scores, axis=0)
    else:
        # 无 det_scores 时，回退到每帧平均关键点置信度最高的人体。
        avg_conf = np.nanmean(keypoints[..., 2], axis=-1)
        best_person = np.argmax(avg_conf, axis=0)

    frame_ids = np.arange(frame_num)
    selected = keypoints[best_person, frame_ids]
    return selected.astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    annotations_root = Path(cfg["aist_root"]) / cfg.get("annotations_subdir", "annotations")
    processed_root = Path(cfg["processed_root"])
    out_root = ensure_dir(processed_root / "yolo2d")

    keypoints2d_dir = annotations_root / "aist_plusplus_final" / "keypoints2d"
    if not keypoints2d_dir.exists():
        raise FileNotFoundError(f"找不到 AIST++ 2D 注释目录: {keypoints2d_dir}")

    files = sorted(keypoints2d_dir.glob("*.pkl"))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise RuntimeError(f"目录中没有可处理文件: {keypoints2d_dir}")

    print(f"[INFO] 待处理 2D 注释文件: {len(files)}")
    for idx, file_path in enumerate(files, start=1):
        seq_id = file_path.stem
        out_file = out_root / f"{seq_id}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}/{len(files)}) {seq_id} 已存在")
            continue

        try:
            data = load_pickle_or_npz(file_path)
            keypoints2d = _select_main_person(data=data, file_path=file_path)
            keypoints2d = np.nan_to_num(keypoints2d, nan=0.0, posinf=0.0, neginf=0.0)
            keypoints2d[..., 2] = np.clip(keypoints2d[..., 2], 0.0, 1.0)

            timestamps = np.asarray(data.get("timestamps", []))
            fps = _infer_fps(timestamps=timestamps)
            style = seq_id.split("_")[0]

            np.savez_compressed(
                out_file,
                keypoints2d=keypoints2d.astype(np.float32),
                fps=np.array(fps, dtype=np.float32),
                style=np.array(style),
            )
            print(f"[DONE] ({idx}/{len(files)}) {seq_id} -> {out_file.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] ({idx}/{len(files)}) {seq_id} 处理失败: {exc}")

    print(f"[DONE] 输出目录: {out_root}")


if __name__ == "__main__":
    main()

