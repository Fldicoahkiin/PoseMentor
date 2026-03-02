#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from posementor.settings import get_paths
from posementor.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO11-Pose 批量提取 AIST++ 2D 关键点")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--weights", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-videos", type=int, default=0, help="仅处理前 N 个视频，0 表示全部")
    parser.add_argument(
        "--video-ext",
        type=str,
        default="mp4",
        help="视频后缀，例如 mp4 或 avi",
    )
    return parser.parse_args()


def find_video_files(video_root: Path, ext: str) -> list[Path]:
    pattern = f"*.{ext.lstrip('.')}"
    return sorted(video_root.glob(pattern))


def style_from_filename(file_name: str) -> str:
    return file_name.split("_")[0]


def extract_single_video(model: YOLO, video_path: Path, conf: float) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames: list[np.ndarray] = []
    for result in model.predict(source=str(video_path), stream=True, conf=conf, verbose=False):
        if result.keypoints is None or len(result.keypoints) == 0:
            frames.append(np.zeros((17, 3), dtype=np.float32))
            continue

        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()
        person_idx = int(np.argmax(kp_conf.mean(axis=1)))
        kp = np.concatenate([kp_xy[person_idx], kp_conf[person_idx, :, None]], axis=-1)
        frames.append(kp.astype(np.float32))

    if not frames:
        raise RuntimeError(f"视频无有效帧: {video_path}")

    return np.stack(frames, axis=0), float(fps if fps > 0 else 30.0)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    video_root = Path(cfg["aist_root"]) / cfg.get("videos_subdir", "videos")
    out_root = ensure_dir(Path(cfg["processed_root"]) / "yolo2d")

    videos = find_video_files(video_root, args.video_ext)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    if not videos:
        raise FileNotFoundError(
            f"未在 {video_root} 找到 *.{args.video_ext} 视频。"
            "请先准备 AIST++ 视频，或修改 --video-ext。"
        )

    print(f"[INFO] 加载模型: {args.weights}")
    model = YOLO(args.weights)

    print(f"[INFO] 待处理视频数: {len(videos)}")
    for idx, video_path in enumerate(videos, start=1):
        seq_id = video_path.stem
        out_file = out_root / f"{seq_id}.npz"
        if out_file.exists():
            print(f"[SKIP] {seq_id} 已存在")
            continue

        try:
            keypoints2d, fps = extract_single_video(model=model, video_path=video_path, conf=args.conf)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {seq_id} 提取失败: {exc}")
            continue

        np.savez_compressed(
            out_file,
            keypoints2d=keypoints2d,
            fps=np.array(fps, dtype=np.float32),
            style=np.array(style_from_filename(seq_id)),
        )
        print(f"[DONE] ({idx}/{len(videos)}) {seq_id} -> {out_file}")

    print("[DONE] YOLO11 关键点提取完成。")


if __name__ == "__main__":
    main()
