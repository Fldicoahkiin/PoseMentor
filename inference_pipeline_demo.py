#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from posementor.pipeline.realtime_coach import CoachConfig, RealtimeDanceCoach
from posementor.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单摄像头推理 Demo：YOLO11 + 3D lift + DTW 打分")
    parser.add_argument("--style", type=str, default="gBR", help="参考舞种，例如 gBR/gPO/gHO")
    parser.add_argument("--yolo-weights", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--lift-ckpt", type=str, default="artifacts/lift_demo.ckpt")
    parser.add_argument("--norm", type=str, default="artifacts/lift_demo_norm.npz")
    parser.add_argument("--template-dir", type=str, default="data/processed/aistpp/gt3d")
    parser.add_argument("--source", type=str, default="webcam", help="webcam 或视频路径")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--save", type=str, default="outputs/inference_demo.mp4")
    parser.add_argument("--show", action="store_true", help="实时弹窗显示")
    return parser.parse_args()


def process_stream(cap: cv2.VideoCapture, coach: RealtimeDanceCoach, style: str, writer: cv2.VideoWriter | None, show: bool) -> None:
    frame_count = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = coach.process_frame(frame_rgb=frame_rgb, style=style)
        vis_rgb = result["annotated"]
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        if writer is not None:
            writer.write(vis_bgr)

        frame_count += 1
        if show:
            cv2.imshow("PoseMentor Demo", vis_bgr)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    elapsed = max(1e-6, time.time() - t0)
    fps = frame_count / elapsed
    print(f"[DONE] 推理结束，平均 FPS = {fps:.2f}")


def main() -> None:
    args = parse_args()

    coach = RealtimeDanceCoach(
        CoachConfig(
            yolo_weights=args.yolo_weights,
            lift_checkpoint=args.lift_ckpt,
            norm_file=args.norm,
            template_dir=args.template_dir,
        )
    )

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_id)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源: {args.source}")

    writer: cv2.VideoWriter | None = None
    if args.save:
        out_path = Path(args.save)
        ensure_dir(out_path.parent)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[INFO] 输出视频: {out_path}")

    try:
        process_stream(cap=cap, coach=coach, style=args.style, writer=writer, show=args.show)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
