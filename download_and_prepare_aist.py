#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from posementor.data.aist_loader import find_gt3d_files, load_gt3d_file
from posementor.utils.io import download_with_progress, ensure_dir, load_yaml, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并预处理 AIST++（快速 Demo 版）")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="数据配置文件",
    )
    parser.add_argument("--download", action="store_true", help="下载 AIST++ 注释压缩包")
    parser.add_argument("--extract", action="store_true", help="下载后自动解压 zip 文件")
    parser.add_argument("--download-videos", action="store_true", help="按官方视频列表下载 10M 视频")
    parser.add_argument(
        "--video-limit",
        type=int,
        default=0,
        help="视频下载数量上限，0 表示全量",
    )
    parser.add_argument(
        "--agree-aist-license",
        action="store_true",
        help="确认你已阅读并同意 AIST++ 数据许可后再下载视频",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="只下载不预处理")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个序列，0 表示全部")
    return parser.parse_args()


def _extract_zip(archive_path: Path, extract_root: Path) -> None:
    ensure_dir(extract_root)
    marker = extract_root / f".{archive_path.stem}.extracted.ok"
    if marker.exists():
        print(f"[SKIP] 已解压: {archive_path.name}")
        return

    print(f"[EXTRACT] {archive_path.name} -> {extract_root}")
    with zipfile.ZipFile(archive_path, mode="r") as zf:
        zf.extractall(extract_root)
    marker.write_text("ok\n", encoding="utf-8")


def do_download(cfg: dict, auto_extract: bool) -> None:
    download_items = cfg.get("download_urls", [])
    if not download_items:
        print("[INFO] data.yaml 未配置 download_urls，跳过下载。")
        return

    raw_root = Path(cfg["aist_root"])
    ensure_dir(raw_root)

    for item in download_items:
        url = item["url"]
        rel_target = Path(item["target"])
        target = raw_root / rel_target

        if target.exists():
            print(f"[SKIP] 已存在: {target}")
        else:
            print(f"[DOWNLOAD] {url}")
            download_with_progress(url=url, target=target)

        if auto_extract and target.suffix.lower() == ".zip":
            extract_to = raw_root / Path(item.get("extract_to", ""))
            _extract_zip(target, extract_to)


def _read_text_lines(url: str) -> list[str]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def do_download_videos(cfg: dict, video_limit: int) -> None:
    raw_root = Path(cfg["aist_root"])
    videos_dir = ensure_dir(raw_root / cfg.get("videos_subdir", "videos"))

    video_cfg = cfg.get("video_download", {})
    list_url = video_cfg.get("list_url")
    source_base_url = video_cfg.get("source_base_url")

    if not list_url or not source_base_url:
        raise ValueError("data.yaml 缺少 video_download.list_url 或 source_base_url")

    video_ids = _read_text_lines(list_url)
    if video_limit > 0:
        video_ids = video_ids[:video_limit]

    print(f"[INFO] 视频下载任务数: {len(video_ids)}")
    for idx, video_id in enumerate(video_ids, start=1):
        stem = Path(video_id).stem
        target = videos_dir / f"{stem}.mp4"
        if target.exists():
            print(f"[SKIP] ({idx}/{len(video_ids)}) {target.name} 已存在")
            continue

        url = f"{source_base_url.rstrip('/')}/{stem}.mp4"
        try:
            print(f"[DOWNLOAD] ({idx}/{len(video_ids)}) {url}")
            download_with_progress(url=url, target=target)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 下载失败 {stem}: {exc}")


def preprocess_aist(cfg: dict, limit: int = 0) -> None:
    raw_root = Path(cfg["aist_root"])
    annotations_root = raw_root / cfg.get("annotations_subdir", "annotations")
    processed_root = Path(cfg["processed_root"])
    gt_out_dir = ensure_dir(processed_root / "gt3d")

    gt_files = find_gt3d_files(annotations_root)
    if not gt_files:
        raise FileNotFoundError(
            f"未找到 AIST++ 3D 注释文件，请检查目录: {annotations_root}\n"
            "你可以先下载 fullset.zip 并解压后再运行本脚本。"
        )

    if limit > 0:
        gt_files = gt_files[:limit]

    print(f"[INFO] 共发现 {len(gt_files)} 个候选序列，开始预处理。")

    metadata_rows: list[dict[str, object]] = []
    style_counter: Counter[str] = Counter()

    for idx, file_path in enumerate(gt_files, start=1):
        try:
            seq = load_gt3d_file(file_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 跳过 {file_path.name}: {exc}")
            continue

        out_file = gt_out_dir / f"{seq.seq_id}.npz"
        np.savez_compressed(
            out_file,
            joints3d=seq.joints3d.astype(np.float32),
            fps=np.array(seq.fps, dtype=np.int32),
            style=np.array(seq.style),
        )

        metadata_rows.append(
            {
                "seq_id": seq.seq_id,
                "style": seq.style,
                "frames": seq.joints3d.shape[0],
                "fps": seq.fps,
                "source_file": str(file_path),
            }
        )
        style_counter[seq.style] += 1

        if idx % 100 == 0:
            print(f"[INFO] 已处理 {idx}/{len(gt_files)}")

    metadata_path = processed_root / "aist_metadata.csv"
    write_csv(
        metadata_path,
        rows=metadata_rows,
        fieldnames=["seq_id", "style", "frames", "fps", "source_file"],
    )

    print(f"[DONE] 3D GT 已输出到: {gt_out_dir}")
    print(f"[DONE] 元数据已输出到: {metadata_path}")
    print("[INFO] 舞种分布:")
    for style, count in style_counter.most_common():
        print(f"  - {style}: {count}")


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"找不到配置文件: {args.config}")

    cfg = load_yaml(args.config)

    if args.download:
        do_download(cfg=cfg, auto_extract=args.extract)

    if args.download_videos:
        if not args.agree_aist_license:
            raise ValueError("下载视频前请显式传入 --agree-aist-license")
        do_download_videos(cfg=cfg, video_limit=args.video_limit)

    if not args.skip_preprocess:
        preprocess_aist(cfg=cfg, limit=args.limit)


if __name__ == "__main__":
    main()
