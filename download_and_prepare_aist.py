#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from posementor.data.aist_loader import find_gt3d_files, load_gt3d_file
from posementor.utils.io import download_with_progress, ensure_dir, load_yaml, write_csv

CAMERA_PATTERN = re.compile(r"_c(\d+)_", re.IGNORECASE)


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
        "--group-limit",
        type=int,
        default=0,
        help="动作组下载数量上限（按同一动作多机位分组），0 表示全量",
    )
    parser.add_argument(
        "--camera-ids",
        type=str,
        default="",
        help="仅下载指定机位，逗号分隔（例如 c01,c02,c03）",
    )
    parser.add_argument(
        "--min-cameras-per-group",
        type=int,
        default=2,
        help="每个动作组最少保留机位数量，默认 2（至少多机位）",
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


def _normalize_camera_id(value: str) -> str:
    raw = value.strip().lower()
    if not raw:
        return ""
    if raw.startswith("c"):
        raw = raw[1:]
    if not raw.isdigit():
        return ""
    return f"c{int(raw):02d}"


def _parse_camera_ids(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []
    cameras: list[str] = []
    for token in raw_text.split(","):
        camera_id = _normalize_camera_id(token)
        if not camera_id:
            continue
        if camera_id not in cameras:
            cameras.append(camera_id)
    return cameras


def _video_group_key(stem: str) -> str:
    return CAMERA_PATTERN.sub("_cAll_", stem)


def _video_camera_id(stem: str) -> str:
    matched = CAMERA_PATTERN.search(stem)
    if not matched:
        return ""
    return f"c{int(matched.group(1)):02d}"


def _select_video_stems(
    video_ids: list[str],
    *,
    group_limit: int,
    video_limit: int,
    camera_ids: list[str],
    min_cameras_per_group: int,
) -> list[str]:
    grouped: dict[str, dict[str, str]] = {}
    for raw_value in video_ids:
        stem = Path(raw_value).stem
        camera_id = _video_camera_id(stem)
        if camera_ids and camera_id not in camera_ids:
            continue
        group_key = _video_group_key(stem)
        camera_map = grouped.setdefault(group_key, {})
        if camera_id and camera_id not in camera_map:
            camera_map[camera_id] = stem

    min_required = max(2, min_cameras_per_group)
    selected_stems: list[str] = []
    selected_groups = 0
    target_cameras = camera_ids.copy()

    for group_key in sorted(grouped.keys()):
        camera_map = grouped[group_key]
        cameras_available = sorted(camera_map.keys())
        if len(cameras_available) < min_required:
            continue
        if target_cameras and not set(target_cameras).issubset(set(cameras_available)):
            continue

        camera_order = target_cameras if target_cameras else cameras_available
        for camera_id in camera_order:
            stem = camera_map.get(camera_id)
            if stem:
                selected_stems.append(stem)

        selected_groups += 1
        if group_limit > 0 and selected_groups >= group_limit:
            break

    if video_limit > 0:
        selected_stems = selected_stems[:video_limit]
    return selected_stems


def do_download_videos(
    cfg: dict,
    video_limit: int,
    group_limit: int,
    camera_ids: list[str],
    min_cameras_per_group: int,
) -> None:
    raw_root = Path(cfg["aist_root"])
    videos_dir = ensure_dir(raw_root / cfg.get("videos_subdir", "videos"))

    video_cfg = cfg.get("video_download", {})
    list_url = video_cfg.get("list_url")
    source_base_url = video_cfg.get("source_base_url")

    if not list_url or not source_base_url:
        raise ValueError("data.yaml 缺少 video_download.list_url 或 source_base_url")

    video_ids = _read_text_lines(list_url)
    selected_stems = _select_video_stems(
        video_ids,
        group_limit=group_limit,
        video_limit=video_limit,
        camera_ids=camera_ids,
        min_cameras_per_group=min_cameras_per_group,
    )

    if not selected_stems:
        print("[WARN] 未匹配到符合条件的视频，请检查机位和分组参数。")
        return

    camera_text = ",".join(camera_ids) if camera_ids else "all"
    print(
        "[INFO] 视频下载任务:"
        f" count={len(selected_stems)} group_limit={group_limit or 'all'}"
        f" cameras={camera_text} min_cameras={max(2, min_cameras_per_group)}"
    )

    for idx, stem in enumerate(selected_stems, start=1):
        target = videos_dir / f"{stem}.mp4"
        if target.exists():
            print(f"[SKIP] ({idx}/{len(selected_stems)}) {target.name} 已存在")
            continue

        url = f"{source_base_url.rstrip('/')}/{stem}.mp4"
        try:
            print(f"[DOWNLOAD] ({idx}/{len(selected_stems)}) {url}")
            download_with_progress(url=url, target=target)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 下载失败 {stem}: {exc}")


def _resolve_annotations_root(raw_root: Path, annotations_subdir: str) -> Path:
    configured_root = raw_root / annotations_subdir
    if find_gt3d_files(configured_root):
        return configured_root

    candidates: list[Path] = []
    for keypoints_dir in raw_root.rglob("keypoints3d"):
        if not keypoints_dir.is_dir():
            continue
        parent = keypoints_dir.parent
        if parent not in candidates:
            candidates.append(parent)
    if not candidates:
        return configured_root

    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    selected = candidates[0]
    if selected != configured_root:
        print(f"[INFO] 自动识别注释目录: {selected}")
    return selected


def preprocess_aist(cfg: dict, limit: int = 0) -> None:
    raw_root = Path(cfg["aist_root"])
    annotations_root = _resolve_annotations_root(
        raw_root=raw_root,
        annotations_subdir=cfg.get("annotations_subdir", "annotations"),
    )
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
        do_download_videos(
            cfg=cfg,
            video_limit=args.video_limit,
            group_limit=args.group_limit,
            camera_ids=_parse_camera_ids(args.camera_ids),
            min_cameras_per_group=args.min_cameras_per_group,
        )

    if not args.skip_preprocess:
        preprocess_aist(cfg=cfg, limit=args.limit)


if __name__ == "__main__":
    main()
