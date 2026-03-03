#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from posementor.pipeline.realtime_coach import CoachConfig, RealtimeDanceCoach
from posementor.utils.io import ensure_dir, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量模型测试与评估报告")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/aistpp/videos"))
    parser.add_argument("--style", type=str, default="gBR")
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--yolo-weights", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--lift-ckpt", type=str, default="artifacts/lift_demo.ckpt")
    parser.add_argument("--norm", type=str, default="artifacts/lift_demo_norm.npz")
    parser.add_argument("--template-dir", type=str, default="data/processed/aistpp/gt3d")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/eval/summary.csv"))
    return parser.parse_args()


def collect_video_paths(root: Path) -> list[Path]:
    items = sorted(root.glob("*.mp4"))
    if not items:
        items = sorted(root.rglob("*.mp4"))
    return items


def eval_single_video(coach: RealtimeDanceCoach, video_path: Path, style: str) -> dict[str, float | str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    coach.reset()
    scores: list[float] = []
    mpjpe_vals: list[float] = []
    angle_vals: list[float] = []

    frame_count = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = coach.process_frame(frame_rgb=frame_rgb, style=style)
        score = float(result["score"])
        mpjpe = float(result["mpjpe_mm"])
        angle = float(result["angle_error_deg"])
        is_ready = bool(result.get("is_ready", False))

        if is_ready:
            scores.append(score)
            mpjpe_vals.append(mpjpe)
            angle_vals.append(angle)

        frame_count += 1

    cap.release()

    elapsed = max(1e-6, time.time() - start)
    runtime_fps = frame_count / elapsed

    return {
        "video": video_path.name,
        "frames": float(frame_count),
        "fps": float(runtime_fps),
        "avg_score": float(np.mean(scores) if scores else 0.0),
        "avg_mpjpe_mm": float(np.mean(mpjpe_vals) if mpjpe_vals else 0.0),
        "avg_angle_deg": float(np.mean(angle_vals) if angle_vals else 0.0),
    }


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")

    videos = collect_video_paths(args.input_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    if not videos:
        raise RuntimeError(f"未找到视频文件: {args.input_dir}")

    coach = RealtimeDanceCoach(
        CoachConfig(
            yolo_weights=args.yolo_weights,
            lift_checkpoint=args.lift_ckpt,
            norm_file=args.norm,
            template_dir=args.template_dir,
            tts_engine="none",
        )
    )

    rows: list[dict[str, float | str]] = []
    for idx, video_path in enumerate(videos, start=1):
        print(f"[INFO] ({idx}/{len(videos)}) 测试: {video_path.name}")
        row = eval_single_video(coach=coach, video_path=video_path, style=args.style)
        rows.append(row)
        print(
            "[DONE] "
            f"score={row['avg_score']:.2f} mpjpe={row['avg_mpjpe_mm']:.2f} "
            f"angle={row['avg_angle_deg']:.2f} fps={row['fps']:.2f}"
        )

    ensure_dir(args.output_csv.parent)
    write_csv(
        path=args.output_csv,
        rows=[{k: v for k, v in row.items()} for row in rows],
        fieldnames=["video", "frames", "fps", "avg_score", "avg_mpjpe_mm", "avg_angle_deg"],
    )

    global_score = float(np.mean([float(row["avg_score"]) for row in rows]))
    global_mpjpe = float(np.mean([float(row["avg_mpjpe_mm"]) for row in rows]))
    global_angle = float(np.mean([float(row["avg_angle_deg"]) for row in rows]))
    global_fps = float(np.mean([float(row["fps"]) for row in rows]))

    print(f"[DONE] 报告输出: {args.output_csv}")
    print(
        "[SUMMARY] "
        f"videos={len(rows)} score={global_score:.2f} mpjpe={global_mpjpe:.2f} "
        f"angle={global_angle:.2f} fps={global_fps:.2f}"
    )


if __name__ == "__main__":
    main()
