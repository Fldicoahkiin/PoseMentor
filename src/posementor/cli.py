from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_python_script(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, script, *extra_args]
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PoseMentor 一体化 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-aist", help="下载并预处理 AIST++")
    p_prepare.add_argument("--config", default="configs/data.yaml")
    p_prepare.add_argument("--download", action="store_true")
    p_prepare.add_argument("--extract", action="store_true")
    p_prepare.add_argument("--download-videos", action="store_true")
    p_prepare.add_argument("--video-limit", type=int, default=0)
    p_prepare.add_argument("--agree-aist-license", action="store_true")
    p_prepare.add_argument("--skip-preprocess", action="store_true")
    p_prepare.add_argument("--limit", type=int, default=0)

    p_yolo = sub.add_parser("extract-yolo2d", help="使用 YOLO11 提取 2D 关键点")
    p_yolo.add_argument("--config", default="configs/data.yaml")
    p_yolo.add_argument("--video-root", default="")
    p_yolo.add_argument("--out-dir", default="")
    p_yolo.add_argument("--recursive", action="store_true")
    p_yolo.add_argument("--weights", default="yolo11m-pose.pt")
    p_yolo.add_argument("--conf", type=float, default=0.35)
    p_yolo.add_argument("--max-videos", type=int, default=0)

    p_aist2d = sub.add_parser("extract-aist2d", help="从 AIST++ 官方 2D 注释构建训练输入")
    p_aist2d.add_argument("--config", default="configs/data.yaml")
    p_aist2d.add_argument("--max-files", type=int, default=0)
    p_aist2d.add_argument("--overwrite", action="store_true")

    p_train = sub.add_parser("train-lift", help="训练 3D Lift 模型")
    p_train.add_argument("--config", default="configs/train.yaml")
    p_train.add_argument("--epochs", type=int, default=0)
    p_train.add_argument("--max-train-pairs", type=int, default=0)
    p_train.add_argument("--max-val-pairs", type=int, default=0)
    p_train.add_argument("--sample-stride", type=int, default=0)
    p_train.add_argument("--seq-len", type=int, default=0)
    p_train.add_argument("--num-workers", type=int, default=-1)
    p_train.add_argument("--export-onnx", action="store_true")

    p_multi = sub.add_parser("prepare-multiview", help="处理四机位数据")
    p_multi.add_argument("--config", default="configs/multiview.yaml")
    p_multi.add_argument("--limit-sessions", type=int, default=0)

    p_report = sub.add_parser("report-multiview", help="生成四机位处理报告")
    p_report.add_argument("--manifest", default="data/processed/multiview/multiview_manifest.csv")
    p_report.add_argument(
        "--output",
        default="outputs/visualization/multiview/multiview_report.html",
    )

    p_backend = sub.add_parser("serve-backend", help="启动 Backend API")
    p_backend.add_argument("--host", default="0.0.0.0")
    p_backend.add_argument("--port", type=int, default=8787)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command

    if cmd == "prepare-aist":
        extra = ["--config", args.config]
        if args.download:
            extra.append("--download")
        if args.extract:
            extra.append("--extract")
        if args.download_videos:
            extra.append("--download-videos")
            if args.video_limit > 0:
                extra.extend(["--video-limit", str(args.video_limit)])
            if args.agree_aist_license:
                extra.append("--agree-aist-license")
        if args.skip_preprocess:
            extra.append("--skip-preprocess")
        if args.limit > 0:
            extra.extend(["--limit", str(args.limit)])
        code = _run_python_script("download_and_prepare_aist.py", extra)
        raise SystemExit(code)

    if cmd == "extract-yolo2d":
        extra = [
            "--config",
            args.config,
            "--weights",
            args.weights,
            "--conf",
            str(args.conf),
        ]
        if args.video_root:
            extra.extend(["--video-root", args.video_root])
        if args.out_dir:
            extra.extend(["--out-dir", args.out_dir])
        if args.recursive:
            extra.append("--recursive")
        if args.max_videos > 0:
            extra.extend(["--max-videos", str(args.max_videos)])
        code = _run_python_script("extract_pose_yolo11.py", extra)
        raise SystemExit(code)

    if cmd == "extract-aist2d":
        extra = ["--config", args.config]
        if args.max_files > 0:
            extra.extend(["--max-files", str(args.max_files)])
        if args.overwrite:
            extra.append("--overwrite")
        code = _run_python_script("extract_pose_aist2d.py", extra)
        raise SystemExit(code)

    if cmd == "train-lift":
        extra = ["--config", args.config]
        if args.epochs > 0:
            extra.extend(["--epochs", str(args.epochs)])
        if args.max_train_pairs > 0:
            extra.extend(["--max-train-pairs", str(args.max_train_pairs)])
        if args.max_val_pairs > 0:
            extra.extend(["--max-val-pairs", str(args.max_val_pairs)])
        if args.sample_stride > 0:
            extra.extend(["--sample-stride", str(args.sample_stride)])
        if args.seq_len > 0:
            extra.extend(["--seq-len", str(args.seq_len)])
        if args.num_workers >= 0:
            extra.extend(["--num-workers", str(args.num_workers)])
        if args.export_onnx:
            extra.append("--export-onnx")
        code = _run_python_script("train_3d_lift_demo.py", extra)
        raise SystemExit(code)

    if cmd == "prepare-multiview":
        extra = ["--config", args.config]
        if args.limit_sessions > 0:
            extra.extend(["--limit-sessions", str(args.limit_sessions)])
        code = _run_python_script("prepare_multiview_dataset.py", extra)
        raise SystemExit(code)

    if cmd == "report-multiview":
        extra = ["--manifest", args.manifest, "--output", args.output]
        code = _run_python_script("visualize_multiview_report.py", extra)
        raise SystemExit(code)

    if cmd == "serve-backend":
        script = Path("backend_api.py")
        if not script.exists():
            raise FileNotFoundError(f"找不到脚本: {script}")
        cmdline = [sys.executable, "backend_api.py", "--host", args.host, "--port", str(args.port)]
        print(f"[CMD] {' '.join(cmdline)}")
        code = subprocess.run(cmdline, check=False).returncode  # noqa: S603
        raise SystemExit(code)

    raise SystemExit(2)
