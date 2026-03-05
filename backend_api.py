#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from posementor.infra.command_runner import JobRunner
from posementor.infra.job_store import JobRecord, JobStore
from posementor.pipeline.preview_renderer import find_sequence_id, render_pose_preview_videos
from posementor.utils.io import ensure_dir, load_yaml, save_yaml

PROJECT_ROOT = Path(__file__).resolve().parent
JOB_ROOT = PROJECT_ROOT / "outputs" / "job_center"
DATASET_REGISTRY_FILE = PROJECT_ROOT / "configs" / "datasets.yaml"
STANDARD_REGISTRY_FILE = PROJECT_ROOT / "configs" / "standards.yaml"
ARTIFACT_ROOT = ensure_dir(PROJECT_ROOT / "artifacts")
DATA_ROOT = ensure_dir(PROJECT_ROOT / "data")
OUTPUT_ROOT = ensure_dir(PROJECT_ROOT / "outputs")

store = JobStore(root=JOB_ROOT)
runner = JobRunner(
    store=store,
    cwd=PROJECT_ROOT,
    max_workers=int(os.environ.get("POSEMENTOR_JOB_WORKERS", "1")),
)

app = FastAPI(title="PoseMentor Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts-files", StaticFiles(directory=ARTIFACT_ROOT), name="artifacts-files")
app.mount("/data-files", StaticFiles(directory=DATA_ROOT), name="data-files")
app.mount("/outputs-files", StaticFiles(directory=OUTPUT_ROOT), name="outputs-files")


class DataPrepareRequest(BaseModel):
    dataset_id: str = "aistpp"
    config: str = "configs/data.yaml"
    download_annotations: bool = True
    extract_annotations: bool = True
    download_videos: bool = False
    video_limit: int = 120
    agree_license: bool = False
    preprocess_limit: int = 0


class ExtractRequest(BaseModel):
    dataset_id: str = "aistpp"
    config: str = "configs/data.yaml"
    input_dir: str | None = None
    out_dir: str | None = None
    recursive: bool = False
    video_ext: str = "mp4"
    weights: str = "yolo11m-pose.pt"
    conf: float = 0.35
    max_videos: int = 0


class TrainRequest(BaseModel):
    dataset_id: str = "aistpp"
    config: str = "configs/train.yaml"
    yolo2d_dir: str | None = None
    gt3d_dir: str | None = None
    artifact_dir: str | None = None
    export_onnx: bool = True


class MultiViewRequest(BaseModel):
    config: str = "configs/multiview.yaml"
    limit_sessions: int = 0


class EvaluateRequest(BaseModel):
    dataset_id: str = "aistpp"
    input_dir: str = "data/raw/aistpp/videos"
    style: str = "gBR"
    max_videos: int = 10
    output_csv: str = "outputs/eval/summary.csv"


class DatasetUpsertRequest(BaseModel):
    id: str
    name: str
    stage: str = "planned"
    mode: str = "singleview"
    data_config: str = ""
    train_config: str = ""
    video_root: str = ""
    notes: str = ""


DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")
CAMERA_TOKEN_PATTERN = re.compile(r"_c\d+_")
PREVIEW_YOLO_WEIGHTS = os.environ.get("POSEMENTOR_PREVIEW_YOLO_WEIGHTS", "yolo11m-pose.pt")
PREVIEW_YOLO_CONF = float(os.environ.get("POSEMENTOR_PREVIEW_YOLO_CONF", "0.35"))

_preview_pose_model = None
_preview_pose_model_lock = threading.Lock()


def _read_dataset_registry() -> dict:
    if not DATASET_REGISTRY_FILE.exists():
        return {"datasets": []}
    data = load_yaml(DATASET_REGISTRY_FILE)
    if not isinstance(data, dict):
        return {"datasets": []}
    datasets = data.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []
    return {"datasets": datasets}


def _read_standard_registry() -> dict:
    if not STANDARD_REGISTRY_FILE.exists():
        return {"standards": []}
    data = load_yaml(STANDARD_REGISTRY_FILE)
    if not isinstance(data, dict):
        return {"standards": []}
    standards = data.get("standards", [])
    if not isinstance(standards, list):
        standards = []
    return {"standards": standards}


def _find_dataset(dataset_id: str) -> dict | None:
    registry = _read_dataset_registry()
    for item in registry["datasets"]:
        if isinstance(item, dict) and str(item.get("id")) == dataset_id:
            return item
    return None


def _normalize_dataset_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(item.get("id", "")).strip(),
        "name": str(item.get("name", "")).strip(),
        "stage": str(item.get("stage", "planned")).strip() or "planned",
        "mode": str(item.get("mode", "singleview")).strip() or "singleview",
        "data_config": str(item.get("data_config", "")).strip(),
        "train_config": str(item.get("train_config", "")).strip(),
        "video_root": str(item.get("video_root", "")).strip(),
        "notes": str(item.get("notes", "")).strip(),
    }


def _guess_dataset_video_root(dataset_id: str, mode: str) -> Path:
    if dataset_id == "aistpp":
        return PROJECT_ROOT / "data" / "raw" / "aistpp" / "videos"
    if mode == "multiview":
        return PROJECT_ROOT / "data" / "raw" / "multiview"
    return PROJECT_ROOT / "data" / "raw" / dataset_id / "videos"


def _resolve_dataset_video_root(dataset: dict[str, str]) -> Path:
    video_root = dataset.get("video_root", "").strip()
    if video_root:
        candidate = Path(video_root)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate

    data_config = dataset.get("data_config", "").strip()
    if data_config:
        from_cfg = _resolve_video_root_from_data_config(PROJECT_ROOT / data_config)
        if from_cfg is not None:
            return from_cfg

    return _guess_dataset_video_root(dataset_id=dataset["id"], mode=dataset["mode"])


def _enrich_dataset_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    item = _normalize_dataset_item(raw_item)
    if not item["id"]:
        return item

    root = _resolve_dataset_video_root(item)
    item["video_root"] = _to_project_relative(root)
    item["video_root_exists"] = root.exists()
    return item


def _dataset_registry_payload() -> dict[str, list[dict[str, Any]]]:
    registry = _read_dataset_registry()
    rows: list[dict[str, Any]] = []
    for item in registry["datasets"]:
        if not isinstance(item, dict):
            continue
        normalized = _enrich_dataset_item(item)
        if normalized.get("id"):
            rows.append(normalized)
    return {"datasets": rows}


def _assert_dataset_exists(dataset_id: str) -> None:
    registry = _dataset_registry_payload()
    ids = {str(item.get("id")) for item in registry["datasets"] if isinstance(item, dict)}
    if dataset_id not in ids:
        raise HTTPException(status_code=400, detail=f"未知 dataset_id: {dataset_id}")


def _assert_aist_dataset(dataset_id: str) -> None:
    if dataset_id != "aistpp":
        raise HTTPException(
            status_code=400,
            detail=(
                f"dataset_id={dataset_id} 目前仅支持通用提取/训练接口，"
                "数据准备任务暂只支持 aistpp"
            ),
        )


def _job_to_dict(job: JobRecord) -> dict[str, object]:
    status = "succeeded" if job.status == "success" else job.status
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": status,
        "command": job.command,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "return_code": job.return_code,
        "log_path": job.log_path,
        "error_message": job.error_message,
    }


def _read_job_log_text(path: Path, max_chars: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars > 0:
        text = text[-max_chars:]
    return text


def _parse_job_progress(job: JobRecord, log_text: str) -> dict[str, object]:
    status = "succeeded" if job.status == "success" else job.status
    phase = "generic"
    if "train_3d_lift" in job.name:
        phase = "train"
    elif "evaluate_model" in job.name:
        phase = "evaluate"
    elif "pose_extract" in job.name:
        phase = "extract"
    elif "data_prepare" in job.name:
        phase = "prepare"
    elif "multiview_prepare" in job.name:
        phase = "multiview"

    progress = 1.0 if status == "succeeded" else 0.0
    current_step = 0
    total_step = 0
    train_step_matches = re.findall(
        r"\[TRAIN_STEP\]\s*epoch=(\d+)(?:\s*/\s*(\d+))?\s*step=(\d+)\s*/\s*(\d+)",
        log_text,
    )
    if train_step_matches:
        epoch_now_raw, epoch_total_raw, step_now_raw, step_total_raw = train_step_matches[-1]
        epoch_now = max(1, int(epoch_now_raw))
        epoch_total = int(epoch_total_raw) if epoch_total_raw else 0
        step_now = max(0, int(step_now_raw))
        step_total = max(1, int(step_total_raw))
        if epoch_total > 0:
            total_step = epoch_total * step_total
            current_step = min(total_step, (epoch_now - 1) * step_total + step_now)
            progress = max(progress, min(1.0, current_step / max(1, total_step)))
        else:
            current_step = min(step_total, step_now)
            total_step = step_total
            progress = max(progress, min(1.0, current_step / max(1, total_step)))

    marker_matches = re.findall(r"\[PROGRESS\]\s*epoch=(\d+)(?:\s*/\s*(\d+))?", log_text)
    if marker_matches:
        current_raw, total_raw = marker_matches[-1]
        current_step = int(current_raw)
        total_step = int(total_raw) if total_raw else 0
        if total_step > 0:
            progress = max(progress, min(1.0, current_step / total_step))
    else:
        pair_matches = re.findall(r"\((\d+)\s*/\s*(\d+)\)", log_text)
        if pair_matches:
            current_step, total_step = map(int, pair_matches[-1])
            if total_step > 0:
                progress = max(progress, min(1.0, current_step / total_step))

    if status == "failed":
        progress = max(0.0, min(1.0, progress))
    elif status == "running" and progress <= 0.0 and log_text.strip():
        progress = 0.01

    event_lines: list[str] = []
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(
            ("[PROGRESS]", "[SUMMARY]", "[DONE]", "[INFO]", "[WARN]", "[ERROR]")
        ):
            event_lines.append(stripped)
        if len(event_lines) >= 6:
            break
    event_lines.reverse()

    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": status,
        "phase": phase,
        "progress": progress,
        "current_step": current_step,
        "total_step": total_step,
        "events": event_lines,
    }


def _resolve_video_root_from_data_config(config_file: Path) -> Path | None:
    if not config_file.exists():
        return None
    data = load_yaml(config_file)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("videos_root"), str):
        return PROJECT_ROOT / str(data["videos_root"])
    if isinstance(data.get("video_root"), str):
        return PROJECT_ROOT / str(data["video_root"])
    if isinstance(data.get("aist_root"), str) and isinstance(data.get("videos_subdir"), str):
        return PROJECT_ROOT / str(data["aist_root"]) / str(data["videos_subdir"])
    return None


def _resolve_pose_dirs_from_dataset(dataset: dict[str, str]) -> tuple[Path, Path]:
    yolo_dir: Path | None = None
    gt_dir: Path | None = None

    data_config = dataset.get("data_config", "").strip()
    if data_config:
        cfg_path = PROJECT_ROOT / data_config
        if cfg_path.exists():
            data_cfg = load_yaml(cfg_path)
            if isinstance(data_cfg, dict) and isinstance(data_cfg.get("processed_root"), str):
                processed_root = PROJECT_ROOT / str(data_cfg["processed_root"])
                yolo_dir = processed_root / "yolo2d"
                gt_dir = processed_root / "gt3d"

    train_config = dataset.get("train_config", "").strip()
    if train_config:
        train_cfg_path = PROJECT_ROOT / train_config
        if train_cfg_path.exists():
            train_cfg = load_yaml(train_cfg_path)
            if isinstance(train_cfg, dict) and isinstance(train_cfg.get("data"), dict):
                data_section = train_cfg["data"]
                if yolo_dir is None and isinstance(data_section.get("yolo2d_dir"), str):
                    yolo_dir = PROJECT_ROOT / str(data_section["yolo2d_dir"])
                if gt_dir is None and isinstance(data_section.get("gt3d_dir"), str):
                    gt_dir = PROJECT_ROOT / str(data_section["gt3d_dir"])

    if yolo_dir is None or gt_dir is None:
        default_root = PROJECT_ROOT / "data" / "processed" / dataset["id"]
        yolo_dir = yolo_dir or (default_root / "yolo2d")
        gt_dir = gt_dir or (default_root / "gt3d")
    return yolo_dir, gt_dir


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".ckpt", ".pth", ".pt", ".onnx", ".npz"}:
        return "model"
    if "visualizations" in path.parts:
        return "visualization"
    if suffix in {".csv", ".json", ".txt", ".yaml", ".yml"}:
        return "report"
    return "other"


def _to_project_relative(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def _video_group_key(path: Path) -> str:
    stem = path.stem
    return CAMERA_TOKEN_PATTERN.sub("_cAll_", stem)


def _resolve_gt_seq_id(gt_dir: Path, source_stem: str, source_name: str) -> str | None:
    direct = gt_dir / f"{source_stem}.npz"
    if direct.exists():
        return source_stem
    call_stem = CAMERA_TOKEN_PATTERN.sub("_cAll_", source_stem)
    call_file = gt_dir / f"{call_stem}.npz"
    if call_file.exists():
        return call_stem
    # 兼容旧目录命名，退回到遍历查找。
    for file_path in sorted(gt_dir.glob("*.npz")):
        if file_path.stem.endswith(source_name.replace(".mp4", "")):
            return file_path.stem
    return None


def _load_keypoints2d(npz_path: Path) -> tuple[np.ndarray, float]:
    with np.load(npz_path) as data:
        if "keypoints2d" not in data.files:
            raise KeyError(f"{npz_path} 缺少 keypoints2d 字段")
        keypoints2d = data["keypoints2d"].astype(np.float32)
        fps_value = 0.0
        if "fps" in data.files:
            try:
                fps_value = float(np.asarray(data["fps"]).reshape(-1)[0])
            except Exception:
                fps_value = 0.0
    return keypoints2d, fps_value


def _get_preview_pose_model():
    global _preview_pose_model
    if _preview_pose_model is not None:
        return _preview_pose_model
    with _preview_pose_model_lock:
        if _preview_pose_model is None:
            from ultralytics import YOLO

            _preview_pose_model = YOLO(PREVIEW_YOLO_WEIGHTS)
    return _preview_pose_model


def _extract_pose2d_from_video(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    model = _get_preview_pose_model()
    frames: list[np.ndarray] = []
    for result in model.predict(
        source=str(video_path),
        stream=True,
        conf=PREVIEW_YOLO_CONF,
        verbose=False,
    ):
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
    return np.stack(frames, axis=0), fps


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "posementor-backend",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def health_compat() -> dict[str, str]:
    # 兼容代理层将后端挂在 /api 前缀时的健康检查路径。
    return {"status": "ok"}


@app.get("/api")
@app.get("/api/")
def api_root() -> dict[str, str]:
    return {
        "service": "posementor-backend",
        "status": "ok",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/jobs")
def list_jobs() -> dict[str, list[dict[str, object]]]:
    return {"jobs": [_job_to_dict(job) for job in store.list_jobs()]}


@app.get("/datasets")
@app.get("/api/datasets")
def list_datasets() -> dict:
    return _dataset_registry_payload()


@app.get("/standards")
@app.get("/api/standards")
def list_standards() -> dict:
    return _read_standard_registry()


@app.get("/workspace/source-preview")
@app.get("/api/workspace/source-preview")
def source_preview(dataset_id: str = "aistpp", limit: int = 3) -> dict[str, object]:
    _assert_dataset_exists(dataset_id)
    bounded_limit = max(1, min(limit, 500))

    dataset = _find_dataset(dataset_id)
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=404, detail=f"dataset_id={dataset_id} 未注册")
    video_root = _resolve_dataset_video_root(_normalize_dataset_item(dataset))
    preview_cache_dir = OUTPUT_ROOT / "preview_cache" / dataset_id

    samples: list[dict[str, object]] = []
    if video_root.exists():
        candidates = sorted(video_root.rglob("*.mp4"))
        grouped: dict[str, list[Path]] = {}
        for path in candidates:
            key = _video_group_key(path)
            grouped.setdefault(key, []).append(path)

        selected_keys = sorted(grouped.keys())[:bounded_limit]
        for group_key in selected_keys:
            for path in sorted(grouped[group_key]):
                camera_match = re.search(r"_c(\d+)_", path.stem)
                camera_id = f"c{camera_match.group(1)}" if camera_match else ""
                rel_project = _to_project_relative(path)
                stat = path.stat()
                url = ""
                if path.is_relative_to(DATA_ROOT):
                    rel_data = path.relative_to(DATA_ROOT).as_posix()
                    url = f"/data-files/{rel_data}"
                pose2d_file = preview_cache_dir / f"{path.stem}_pose2d.mp4"
                pose3d_file = preview_cache_dir / f"{path.stem}_pose3d.mp4"
                samples.append(
                    {
                        "name": path.name,
                        "path": rel_project,
                        "url": url,
                        "size_bytes": stat.st_size,
                        "group_key": group_key,
                        "camera_id": camera_id,
                        "pose2d_exists": pose2d_file.exists(),
                        "pose3d_exists": pose3d_file.exists(),
                    }
                )

    return {
        "dataset_id": dataset_id,
        "video_root": _to_project_relative(video_root),
        "samples": samples,
    }


@app.get("/workspace/pose-preview")
@app.get("/api/workspace/pose-preview")
def workspace_pose_preview(dataset_id: str, video_path: str, refresh: bool = False) -> dict[str, object]:
    _assert_dataset_exists(dataset_id)
    dataset = _find_dataset(dataset_id)
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=404, detail=f"dataset_id={dataset_id} 未注册")

    source_video = (PROJECT_ROOT / video_path).resolve()
    if not source_video.exists() or not source_video.is_file():
        raise HTTPException(status_code=404, detail=f"视频不存在: {video_path}")
    try:
        source_video.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="video_path 必须位于项目目录内") from exc
    try:
        rel_data_video = source_video.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="video_path 必须位于 data 目录内") from exc

    yolo_dir, gt_dir = _resolve_pose_dirs_from_dataset(_normalize_dataset_item(dataset))
    if not yolo_dir.exists() or not gt_dir.exists():
        detail = (
            "2D/3D 数据目录不存在: "
            f"yolo2d={_to_project_relative(yolo_dir)} "
            f"gt3d={_to_project_relative(gt_dir)}"
        )
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    source_name = source_video.name
    source_stem = source_video.stem

    gt_seq_id = _resolve_gt_seq_id(gt_dir=gt_dir, source_stem=source_stem, source_name=source_name)
    if not gt_seq_id:
        raise HTTPException(status_code=404, detail=f"未找到视频对应 3D 序列: {source_name}")
    gt_file = gt_dir / f"{gt_seq_id}.npz"
    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"未找到 3D 文件: seq_id={gt_seq_id}")
    with np.load(gt_file) as gt_data:
        joints3d = gt_data["joints3d"].astype(np.float32)

    preview_pose_cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id / "pose2d_npz")
    source_pose2d_cache = preview_pose_cache_dir / f"{source_stem}.npz"
    source_pose2d_file = yolo_dir / f"{source_stem}.npz"

    keypoints2d: np.ndarray
    fps_value: float
    pose2d_dep_file: Path
    if source_pose2d_file.exists():
        keypoints2d, fps_value = _load_keypoints2d(source_pose2d_file)
        pose2d_dep_file = source_pose2d_file
    elif source_pose2d_cache.exists():
        keypoints2d, fps_value = _load_keypoints2d(source_pose2d_cache)
        pose2d_dep_file = source_pose2d_cache
    else:
        try:
            keypoints2d, fps_value = _extract_pose2d_from_video(source_video)
            np.savez_compressed(
                source_pose2d_cache,
                keypoints2d=keypoints2d,
                fps=np.array(fps_value, dtype=np.float32),
                source=np.array(source_name),
            )
            pose2d_dep_file = source_pose2d_cache
        except Exception:
            fallback_seq_id = find_sequence_id(
                yolo2d_dir=yolo_dir,
                video_stem=source_stem,
                source_video_name=source_name,
            )
            if not fallback_seq_id:
                raise HTTPException(status_code=404, detail=f"未找到视频对应 2D 关键点: {source_name}")
            fallback_yolo_file = yolo_dir / f"{fallback_seq_id}.npz"
            if not fallback_yolo_file.exists():
                raise HTTPException(status_code=404, detail=f"未找到 2D 文件: seq_id={fallback_seq_id}")
            keypoints2d, fps_value = _load_keypoints2d(fallback_yolo_file)
            pose2d_dep_file = fallback_yolo_file

    cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id)
    output_2d = cache_dir / f"{source_stem}_pose2d.mp4"
    output_3d = cache_dir / f"{source_stem}_pose3d.mp4"
    source_mtime = source_video.stat().st_mtime
    dep_mtime = max(pose2d_dep_file.stat().st_mtime, gt_file.stat().st_mtime, source_mtime)
    need_render = bool(refresh)
    if output_2d.exists() and output_3d.exists():
        output_mtime = min(output_2d.stat().st_mtime, output_3d.stat().st_mtime)
        if not need_render:
            need_render = output_mtime < dep_mtime

    stats: dict[str, float] = {
        "fps": max(0.0, fps_value),
        "frames": float(min(len(keypoints2d), len(joints3d))),
    }
    if need_render:
        stats = render_pose_preview_videos(
            source_video=source_video,
            keypoints2d=keypoints2d,
            joints3d=joints3d,
            output_2d=output_2d,
            output_3d=output_3d,
        )

    return {
        "dataset_id": dataset_id,
        "seq_id": gt_seq_id,
        "source_video_url": f"/data-files/{rel_data_video.as_posix()}",
        "pose2d_video_url": f"/outputs-files/preview_cache/{dataset_id}/{output_2d.name}",
        "pose3d_video_url": f"/outputs-files/preview_cache/{dataset_id}/{output_3d.name}",
        "fps": stats["fps"],
        "frames": stats["frames"],
    }


@app.post("/datasets/upsert")
@app.post("/api/datasets/upsert")
def upsert_dataset(req: DatasetUpsertRequest) -> dict[str, object]:
    normalized = _normalize_dataset_item(req.model_dump())
    if not DATASET_ID_PATTERN.fullmatch(normalized["id"]):
        raise HTTPException(
            status_code=400,
            detail="dataset_id 仅允许字母、数字、下划线和短横线，长度 2-64",
        )
    if normalized["mode"] not in {"singleview", "multiview"}:
        raise HTTPException(status_code=400, detail="mode 仅支持 singleview 或 multiview")
    if not normalized["name"]:
        normalized["name"] = normalized["id"]

    registry = _read_dataset_registry()
    rows = [item for item in registry.get("datasets", []) if isinstance(item, dict)]

    found = False
    for idx, item in enumerate(rows):
        if str(item.get("id", "")).strip() == normalized["id"]:
            rows[idx] = normalized
            found = True
            break
    if not found:
        rows.append(normalized)

    save_yaml(DATASET_REGISTRY_FILE, {"datasets": rows})
    return {"ok": True, "dataset": _enrich_dataset_item(normalized)}


@app.get("/artifacts/status")
@app.get("/api/artifacts/status")
def artifact_status() -> dict[str, object]:
    curves_file = ARTIFACT_ROOT / "visualizations" / "training_curves.html"
    sample2d_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_2d_latest.png"
    sample3d_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_3d_latest.html"
    sample_video_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_video_latest.mp4"
    sample2d_video_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_2d_latest.mp4"
    sample3d_video_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_3d_latest.mp4"
    sample_sync_meta_file = (
        ARTIFACT_ROOT / "visualizations" / "samples" / "sample_sync_meta_latest.json"
    )
    summary_file = ARTIFACT_ROOT / "visualizations" / "samples" / "sample_summary_latest.txt"

    return {
        "curves_exists": curves_file.exists(),
        "curves_url": "/artifacts-files/visualizations/training_curves.html",
        "sample_video_exists": sample_video_file.exists(),
        "sample_video_url": "/artifacts-files/visualizations/samples/sample_video_latest.mp4",
        "sample_2d_exists": sample2d_file.exists(),
        "sample_2d_url": "/artifacts-files/visualizations/samples/sample_2d_latest.png",
        "sample_2d_video_exists": sample2d_video_file.exists(),
        "sample_2d_video_url": "/artifacts-files/visualizations/samples/sample_2d_latest.mp4",
        "sample_3d_exists": sample3d_file.exists(),
        "sample_3d_url": "/artifacts-files/visualizations/samples/sample_3d_latest.html",
        "sample_3d_video_exists": sample3d_video_file.exists(),
        "sample_3d_video_url": "/artifacts-files/visualizations/samples/sample_3d_latest.mp4",
        "sample_sync_meta_exists": sample_sync_meta_file.exists(),
        "sample_sync_meta_url": (
            "/artifacts-files/visualizations/samples/sample_sync_meta_latest.json"
        ),
        "summary_exists": summary_file.exists(),
        "summary_url": "/artifacts-files/visualizations/samples/sample_summary_latest.txt",
    }


@app.get("/artifacts/manifest")
@app.get("/api/artifacts/manifest")
def artifact_manifest(limit: int = 200) -> dict[str, object]:
    bounded_limit = max(1, min(limit, 1000))
    files = [path for path in ARTIFACT_ROOT.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    files = files[:bounded_limit]

    rows: list[dict[str, object]] = []
    by_kind: dict[str, int] = {}
    for path in files:
        stat = path.stat()
        rel = path.relative_to(ARTIFACT_ROOT).as_posix()
        kind = _artifact_kind(path)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        rows.append(
            {
                "name": path.name,
                "path": rel,
                "url": f"/artifacts-files/{rel}",
                "kind": kind,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )

    return {
        "count": len(rows),
        "by_kind": by_kind,
        "files": rows,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return _job_to_dict(job)


@app.get("/jobs/{job_id}/log")
def get_job_log(job_id: str, max_chars: int = 8000) -> dict[str, str]:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    path = Path(job.log_path)
    if not path.exists():
        return {"log": ""}

    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars > 0:
        text = text[-max_chars:]
    return {"log": text}


@app.get("/jobs/{job_id}/progress")
@app.get("/api/jobs/{job_id}/progress")
def get_job_progress(job_id: str) -> dict[str, object]:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    log_path = Path(job.log_path)
    log_text = _read_job_log_text(log_path, max_chars=200_000)
    return _parse_job_progress(job=job, log_text=log_text)


@app.post("/jobs/data/prepare")
def start_data_prepare(req: DataPrepareRequest) -> dict[str, str]:
    _assert_dataset_exists(req.dataset_id)
    _assert_aist_dataset(req.dataset_id)

    if req.download_videos and not req.agree_license:
        raise HTTPException(status_code=400, detail="download_videos=true 时必须勾选 agree_license")

    command = [
        "uv",
        "run",
        "python",
        "download_and_prepare_aist.py",
        "--config",
        req.config,
    ]

    if req.download_annotations:
        command.append("--download")
    if req.extract_annotations:
        command.append("--extract")
    if req.download_videos:
        command.append("--download-videos")
        command.extend(["--video-limit", str(req.video_limit)])
        command.append("--agree-aist-license")
    if req.preprocess_limit > 0:
        command.extend(["--limit", str(req.preprocess_limit)])

    job_id = runner.submit(name="data_prepare", command=command)
    return {"job_id": job_id}


@app.post("/jobs/pose/extract")
def start_pose_extract(req: ExtractRequest) -> dict[str, str]:
    _assert_dataset_exists(req.dataset_id)

    command = [
        "uv",
        "run",
        "python",
        "extract_pose_yolo11.py",
        "--config",
        req.config,
        "--weights",
        req.weights,
        "--conf",
        str(req.conf),
        "--video-ext",
        req.video_ext,
    ]
    if req.input_dir:
        command.extend(["--video-root", req.input_dir])
    if req.out_dir:
        command.extend(["--out-dir", req.out_dir])
    if req.recursive:
        command.append("--recursive")
    if req.max_videos > 0:
        command.extend(["--max-videos", str(req.max_videos)])

    job_id = runner.submit(name=f"pose_extract_{req.dataset_id}", command=command)
    return {"job_id": job_id}


@app.post("/jobs/train")
def start_train(req: TrainRequest) -> dict[str, str]:
    _assert_dataset_exists(req.dataset_id)

    command = [
        "uv",
        "run",
        "python",
        "train_3d_lift_demo.py",
        "--config",
        req.config,
    ]
    if req.yolo2d_dir:
        command.extend(["--yolo2d-dir", req.yolo2d_dir])
    if req.gt3d_dir:
        command.extend(["--gt3d-dir", req.gt3d_dir])
    if req.artifact_dir:
        command.extend(["--artifact-dir", req.artifact_dir])
    if req.export_onnx:
        command.append("--export-onnx")

    job_id = runner.submit(name=f"train_3d_lift_{req.dataset_id}", command=command)
    return {"job_id": job_id}


@app.post("/jobs/multiview/prepare")
def start_multiview_prepare(req: MultiViewRequest) -> dict[str, str]:
    command = [
        "uv",
        "run",
        "python",
        "prepare_multiview_dataset.py",
        "--config",
        req.config,
    ]
    if req.limit_sessions > 0:
        command.extend(["--limit-sessions", str(req.limit_sessions)])

    job_id = runner.submit(name="multiview_prepare", command=command)
    return {"job_id": job_id}


@app.post("/jobs/evaluate")
def start_evaluate(req: EvaluateRequest) -> dict[str, str]:
    _assert_dataset_exists(req.dataset_id)

    command = [
        "uv",
        "run",
        "python",
        "evaluate_model_suite.py",
        "--input-dir",
        req.input_dir,
        "--style",
        req.style,
        "--max-videos",
        str(req.max_videos),
        "--output-csv",
        req.output_csv,
    ]

    job_id = runner.submit(name=f"evaluate_model_{req.dataset_id}", command=command)
    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="PoseMentor Backend API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    cli_args = parser.parse_args()
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
