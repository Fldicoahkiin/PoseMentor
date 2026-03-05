#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from posementor.data.aist_loader import find_gt3d_files, load_gt3d_file
from posementor.utils.io import download_with_progress, ensure_dir, load_yaml, write_csv

CAMERA_PATTERN = re.compile(r"_c(\d+)_", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")
BYTES_PER_MIB = 1024 * 1024
DEFAULT_MIN_VALID_VIDEO_SIZE = 1 * BYTES_PER_MIB


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
        "--ranges",
        type=str,
        default="",
        help="仅下载指定区间（1-based，逗号分隔，如 1-300,1000-1200）",
    )
    parser.add_argument(
        "--assume-speed-mbps",
        type=float,
        default=10.0,
        help="估算剩余时间时使用的基准网速（Mbps）",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="单个视频下载失败后的重试次数",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("outputs/runtime/aist_download_state.json"),
        help="下载状态文件路径（用于续传失败列表）",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="仅下载上次失败的视频列表（基于 state-file）",
    )
    parser.add_argument(
        "--agree-aist-license",
        action="store_true",
        help="确认你已阅读并同意 AIST++ 数据许可后再下载视频",
    )
    parser.add_argument("--plan-only", action="store_true", help="仅展示下载计划，不执行下载")
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


def _parse_ranges(raw_text: str) -> list[tuple[int, int]]:
    if not raw_text.strip():
        return []

    ranges: list[tuple[int, int]] = []
    for token in raw_text.split(","):
        value = token.strip()
        if not value:
            continue
        matched = RANGE_PATTERN.match(value)
        if not matched:
            raise ValueError(f"非法 ranges 段: {value}")
        start = int(matched.group(1))
        end = int(matched.group(2)) if matched.group(2) else start
        if start <= 0 or end <= 0:
            raise ValueError(f"ranges 仅支持正整数: {value}")
        if end < start:
            start, end = end, start
        ranges.append((start, end))

    ranges.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return merged


def _format_ranges(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "all"
    segments: list[str] = []
    for start, end in ranges:
        if start == end:
            segments.append(str(start))
        else:
            segments.append(f"{start}-{end}")
    return ",".join(segments)


def _apply_ranges(stems: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    if not ranges:
        return stems
    picked: list[str] = []
    added: set[str] = set()
    total = len(stems)
    for start, end in ranges:
        left = max(1, start)
        right = min(total, end)
        if left > right:
            continue
        for stem in stems[left - 1:right]:
            if stem in added:
                continue
            picked.append(stem)
            added.add(stem)
    return picked


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hh = value // 3600
    mm = (value % 3600) // 60
    ss = value % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _estimate_video_size_bytes(videos_dir: Path, video_cfg: dict) -> int:
    default_mb = float(video_cfg.get("estimated_video_size_mb", 18.0))
    default_bytes = max(1, int(default_mb * BYTES_PER_MIB))

    if not videos_dir.exists():
        return default_bytes

    sample_sizes: list[int] = []
    for idx, file_path in enumerate(sorted(videos_dir.rglob("*.mp4"))):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size <= 0:
            continue
        sample_sizes.append(size)
        if idx >= 799:
            break

    if len(sample_sizes) < 16:
        return default_bytes

    sample_sizes.sort()
    mid = len(sample_sizes) // 2
    if len(sample_sizes) % 2 == 0:
        return int((sample_sizes[mid - 1] + sample_sizes[mid]) / 2)
    return int(sample_sizes[mid])


def _target_is_ready(path: Path, min_valid_size: int) -> bool:
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return size >= min_valid_size


def _download_video_with_retry(url: str, target: Path, retry: int) -> bool:
    temp_path = target.with_suffix(f"{target.suffix}.part")
    attempts = max(1, retry + 1)
    for attempt in range(1, attempts + 1):
        try:
            if temp_path.exists():
                temp_path.unlink()
            download_with_progress(url=url, target=temp_path)
            temp_path.replace(target)
            return True
        except Exception as exc:  # noqa: BLE001
            if temp_path.exists():
                temp_path.unlink()
            if attempt >= attempts:
                print(f"[WARN] 下载失败 {target.stem}: {exc}")
                return False
            print(f"[RETRY] {target.name} 第 {attempt}/{attempts - 1} 次重试前失败: {exc}")
            time.sleep(min(6, attempt * 2))
    return False


def _load_download_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _save_download_state(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_failed_stems_from_state(path: Path) -> list[str]:
    payload = _load_download_state(path)
    rows = payload.get("failed_stems", [])
    if not isinstance(rows, list):
        return []
    stems: list[str] = []
    for item in rows:
        if not isinstance(item, str):
            continue
        stem = Path(item).stem
        if not stem:
            continue
        if stem not in stems:
            stems.append(stem)
    return stems


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
    ranges: list[tuple[int, int]],
    assume_speed_mbps: float,
    retry: int,
    plan_only: bool,
    state_file: Path,
    resume_failed: bool,
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
    selected_stems = _apply_ranges(selected_stems, ranges)

    if resume_failed:
        failed_stems = _read_failed_stems_from_state(state_file)
        if not failed_stems:
            print(f"[WARN] 未在状态文件中发现失败记录: {state_file}")
            return
        failed_set = set(failed_stems)
        selected_stems = [stem for stem in selected_stems if stem in failed_set]
        if not selected_stems:
            print("[WARN] 当前筛选条件下无可续传失败项，请检查 Profile/ranges 配置。")
            return
        print(f"[INFO] 续传模式启用：共匹配到 {len(selected_stems)} 个失败视频。")

    if not selected_stems:
        print("[WARN] 未匹配到符合条件的视频，请检查机位和分组参数。")
        return

    camera_text = ",".join(camera_ids) if camera_ids else "all"
    min_valid_size = int(video_cfg.get("min_valid_size_bytes", DEFAULT_MIN_VALID_VIDEO_SIZE))
    estimated_video_size = _estimate_video_size_bytes(videos_dir, video_cfg)
    assume_speed_bytes = max(1, int(max(0.1, assume_speed_mbps) * 125_000))

    existing_count = 0
    pending_stems: list[str] = []
    existing_bytes = 0
    for stem in selected_stems:
        target = videos_dir / f"{stem}.mp4"
        if _target_is_ready(target, min_valid_size=min_valid_size):
            existing_count += 1
            try:
                existing_bytes += target.stat().st_size
            except OSError:
                existing_bytes += estimated_video_size
            continue
        pending_stems.append(stem)

    pending_count = len(pending_stems)
    total_count = len(selected_stems)
    estimated_total_bytes = existing_bytes + pending_count * estimated_video_size
    estimated_pending_bytes = max(0, estimated_total_bytes - existing_bytes)
    eta_seconds = estimated_pending_bytes / assume_speed_bytes

    print(
        "[INFO] 视频下载任务:"
        f" total={total_count} existed={existing_count} pending={pending_count}"
        f" group_limit={group_limit or 'all'} cameras={camera_text}"
        f" min_cameras={max(2, min_cameras_per_group)} ranges={_format_ranges(ranges)}"
        f" resume_failed={'yes' if resume_failed else 'no'}"
    )
    print(
        "[INFO] 估算:"
        f" avg_video={_format_bytes(estimated_video_size)}"
        f" pending_size={_format_bytes(estimated_pending_bytes)}"
        f" assume_speed={assume_speed_mbps:.1f}Mbps"
        f" eta~{_format_duration(eta_seconds)}"
    )

    if plan_only:
        print("[DONE] 仅展示计划，未执行下载。")
        return

    if pending_count == 0:
        _save_download_state(
            state_file,
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "status": "completed",
                "selected_total": total_count,
                "existing_count": existing_count,
                "pending_count": 0,
                "failed_count": 0,
                "failed_stems": [],
                "state_file": str(state_file),
                "video_root": str(videos_dir),
            },
        )
        print("[DONE] 目标文件已全部存在，无需下载。")
        print(f"[INFO] 状态已更新: {state_file}")
        return

    start_ts = time.time()
    transferred_bytes = 0
    done_count = existing_count
    failed_count = 0
    failed_stems_run: list[str] = []

    for idx, stem in enumerate(selected_stems, start=1):
        target = videos_dir / f"{stem}.mp4"
        if _target_is_ready(target, min_valid_size=min_valid_size):
            print(f"[SKIP] ({idx}/{len(selected_stems)}) {target.name} 已存在")
            continue

        url = f"{source_base_url.rstrip('/')}/{stem}.mp4"
        print(f"[DOWNLOAD] ({idx}/{len(selected_stems)}) {url}")
        before_size = 0
        if target.exists():
            try:
                before_size = target.stat().st_size
            except OSError:
                before_size = 0

        ok = _download_video_with_retry(url=url, target=target, retry=retry)
        if ok:
            done_count += 1
            try:
                after_size = target.stat().st_size
            except OSError:
                after_size = estimated_video_size
            transferred_bytes += max(0, after_size - before_size)
        else:
            failed_count += 1
            failed_stems_run.append(stem)

        elapsed = max(0.1, time.time() - start_ts)
        measured_speed = transferred_bytes / elapsed
        current_speed = max(measured_speed, assume_speed_bytes)
        remaining_count = max(0, total_count - done_count)
        remaining_bytes = max(0, remaining_count * estimated_video_size)
        progress = (done_count / total_count) * 100 if total_count > 0 else 0.0
        eta = remaining_bytes / current_speed if current_speed > 0 else 0
        print(
            "[PROGRESS]"
            f" {done_count}/{total_count} ({progress:.1f}%)"
            f" failed={failed_count}"
            f" speed~{(current_speed / 125_000):.2f}Mbps"
            f" eta~{_format_duration(eta)}"
        )

    print(
        "[DONE] 视频下载结束:"
        f" success={done_count - existing_count}/{pending_count}"
        f" skipped={existing_count}"
        f" failed={failed_count}"
    )
    _save_download_state(
        state_file,
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "completed" if failed_count == 0 else "partial",
            "selected_total": total_count,
            "existing_count": existing_count,
            "pending_count": pending_count,
            "downloaded_now": done_count - existing_count,
            "failed_count": failed_count,
            "failed_stems": failed_stems_run,
            "camera_ids": camera_ids,
            "group_limit": group_limit,
            "ranges": _format_ranges(ranges),
            "resume_failed": resume_failed,
            "state_file": str(state_file),
            "video_root": str(videos_dir),
        },
    )
    print(f"[INFO] 状态已写入: {state_file}")


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
    if args.plan_only:
        args.skip_preprocess = True
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
            ranges=_parse_ranges(args.ranges),
            assume_speed_mbps=args.assume_speed_mbps,
            retry=args.retry,
            plan_only=args.plan_only,
            state_file=args.state_file,
            resume_failed=args.resume_failed,
        )

    if not args.skip_preprocess:
        preprocess_aist(cfg=cfg, limit=args.limit)


if __name__ == "__main__":
    main()
