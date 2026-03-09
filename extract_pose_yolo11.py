#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from posementor.multiview.naming import build_video_rel_path, build_video_seq_id
from posementor.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO11-Pose 批量提取 AIST++ 2D 关键点")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="视频目录，默认使用 data.yaml 中的 aist_root/videos_subdir",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认使用 data.yaml 中的 processed_root/yolo2d",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描视频目录，适用于四机位 session 树形目录",
    )
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


def find_video_files(video_root: Path, ext: str, recursive: bool) -> list[Path]:
    pattern = f"*.{ext.lstrip('.')}"
    if recursive:
        return sorted(video_root.rglob(pattern))
    return sorted(video_root.glob(pattern))


def style_from_filename(file_name: str) -> str:
    token = file_name.split("_")[0]
    return token if token.startswith("g") else "unknown"


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

    video_root = (
        args.video_root
        if args.video_root is not None
        else Path(cfg["aist_root"]) / cfg.get("videos_subdir", "videos")
    )
    out_root = ensure_dir(
        args.out_dir if args.out_dir is not None else Path(cfg["processed_root"]) / "yolo2d"
    )

    videos = find_video_files(video_root, args.video_ext, recursive=args.recursive)
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
        if args.recursive:
            seq_id = build_video_seq_id(video_root=video_root, video_path=video_path)
        else:
            seq_id = video_path.stem
        out_file = out_root / f"{seq_id}.npz"
        if out_file.exists():
            print(f"[SKIP] {seq_id} 已存在")
            continue

        try:
            keypoints2d, fps = extract_single_video(
                model=model,
                video_path=video_path,
                conf=args.conf,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {seq_id} 提取失败: {exc}")
            continue

        source_video_rel = build_video_rel_path(video_root=video_root, video_path=video_path)
        np.savez_compressed(
            out_file,
            keypoints2d=keypoints2d,
            fps=np.array(fps, dtype=np.float32),
            style=np.array(style_from_filename(seq_id)),
            source_video_name=np.array(video_path.name),
            source_video_rel=np.array(source_video_rel),
            camera_id=np.array(video_path.stem.lower()),
        )
        print(f"[DONE] ({idx}/{len(videos)}) {seq_id} -> {out_file}")

    print("[DONE] YOLO11 关键点提取完成。")


if __name__ == "__main__":
    main()
