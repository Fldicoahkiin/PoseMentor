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
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="固定机位索引（1-9）；默认 -1 按检测稳定性自动选择",
    )
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


def _resolve_source_video_name(seq_id: str, camera_id: str) -> str:
    if "_cAll_" in seq_id:
        return f"{seq_id.replace('_cAll_', f'_{camera_id}_')}.mp4"
    return f"{seq_id}.mp4"


def _select_camera_index(
    keypoints: np.ndarray,
    det_scores: np.ndarray | None,
    preferred_camera_idx: int | None,
) -> int:
    view_count = int(keypoints.shape[0])
    if preferred_camera_idx is not None and 0 <= preferred_camera_idx < view_count:
        return preferred_camera_idx

    if det_scores is not None and det_scores.shape[0] == view_count:
        per_view = np.nanmean(np.nan_to_num(det_scores, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
        return int(np.argmax(per_view))

    conf = np.nan_to_num(keypoints[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    per_view = np.mean(conf, axis=(1, 2))
    return int(np.argmax(per_view))


def _select_main_person(
    data: dict[str, np.ndarray],
    file_path: Path,
    preferred_camera_idx: int | None,
) -> tuple[np.ndarray, str]:
    keypoints = np.asarray(data.get("keypoints2d"))
    if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
        return keypoints.astype(np.float32), "c01"

    if keypoints.ndim != 4 or keypoints.shape[-1] != 3:
        raise ValueError(f"不支持的 keypoints2d 形状: {keypoints.shape} ({file_path.name})")

    det_scores = np.asarray(data.get("det_scores")) if data.get("det_scores") is not None else None
    selected_camera_idx = _select_camera_index(
        keypoints=keypoints,
        det_scores=det_scores,
        preferred_camera_idx=preferred_camera_idx,
    )
    camera_id = f"c{selected_camera_idx + 1:02d}"
    selected = keypoints[selected_camera_idx]
    return selected.astype(np.float32), camera_id


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    preferred_camera_idx = args.camera_index - 1 if args.camera_index > 0 else None

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
            keypoints2d, camera_id = _select_main_person(
                data=data,
                file_path=file_path,
                preferred_camera_idx=preferred_camera_idx,
            )
            keypoints2d = np.nan_to_num(keypoints2d, nan=0.0, posinf=0.0, neginf=0.0)
            keypoints2d[..., 2] = np.clip(keypoints2d[..., 2], 0.0, 1.0)

            timestamps = np.asarray(data.get("timestamps", []))
            fps = _infer_fps(timestamps=timestamps)
            style = seq_id.split("_")[0]
            source_video_name = _resolve_source_video_name(seq_id=seq_id, camera_id=camera_id)
            camera_index = int(camera_id[1:]) - 1

            np.savez_compressed(
                out_file,
                keypoints2d=keypoints2d.astype(np.float32),
                fps=np.array(fps, dtype=np.float32),
                style=np.array(style),
                camera_id=np.array(camera_id),
                camera_index=np.array(camera_index, dtype=np.int32),
                source_video_name=np.array(source_video_name),
            )
            print(f"[DONE] ({idx}/{len(files)}) {seq_id} -> {out_file.name} ({camera_id})")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] ({idx}/{len(files)}) {seq_id} 处理失败: {exc}")

    print(f"[DONE] 输出目录: {out_root}")


if __name__ == "__main__":
    main()
