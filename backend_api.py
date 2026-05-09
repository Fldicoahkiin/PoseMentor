#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from posementor.data.aist_alignment import (
    collect_group_video_paths,
    extract_camera_id,
    load_aist_alignment_meta,
    load_group_keypoints2d,
    plan_group_preview,
    read_video_stats,
    resolve_group_seq_id,
)
from posementor.infra.command_runner import JobRunner
from posementor.infra.job_store import JobRecord, JobStore
from posementor.multiview.naming import build_video_rel_path, build_video_seq_id
from posementor.pipeline.preview_renderer import (
    POSE3D_PREVIEW_VERSION,
    build_pose2d_preview_data,
    build_pose3d_preview_data,
    find_sequence_id,
    render_pose_preview_videos,
    render_source_preview_video,
)
from posementor.utils.io import ensure_dir, load_yaml, save_yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
JOB_ROOT = PROJECT_ROOT / "outputs" / "job_center"
DATASET_REGISTRY_FILE = PROJECT_ROOT / "configs" / "datasets.yaml"
STANDARD_REGISTRY_FILE = PROJECT_ROOT / "configs" / "standards.yaml"
ARTIFACT_ROOT = ensure_dir(PROJECT_ROOT / "artifacts")
DATA_ROOT = ensure_dir(PROJECT_ROOT / "data")
OUTPUT_ROOT = ensure_dir(PROJECT_ROOT / "outputs")
AIST_ANNOTATIONS_ROOT = DATA_ROOT / "raw" / "aistpp" / "annotations" / "aist_plusplus_final"

store = JobStore(root=JOB_ROOT)
runner = JobRunner(
    store=store,
    cwd=PROJECT_ROOT,
    max_workers=int(os.environ.get("POSEMENTOR_JOB_WORKERS", "1")),
)

app = FastAPI(title="PoseMentor Backend", version="0.1.0")

# CORS: 从环境变量读取允许的 origins，避免通配符 + credentials 的危险组合
_cors_origins_raw = os.environ.get(
    "POSEMENTOR_CORS_ORIGINS",
    "http://localhost:7860,http://localhost:5173,http://127.0.0.1:7860,http://127.0.0.1:5173",
)
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注意: StaticFiles 会暴露整个目录，生产环境应限制访问范围或添加鉴权
app.mount("/artifacts-files", StaticFiles(directory=ARTIFACT_ROOT), name="artifacts-files")
app.mount("/data-files", StaticFiles(directory=DATA_ROOT), name="data-files")
app.mount("/outputs-files", StaticFiles(directory=OUTPUT_ROOT), name="outputs-files")

# 业务路由统一定义在 router 上，挂载到 / 和 /api 两个前缀，消除重复注册
router = APIRouter()

DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")

_SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_/.\-]+$")
_SAFE_EXT_PATTERN = re.compile(r"^[a-z0-9]{1,6}$")


def _validate_safe_path(value: str, field_name: str) -> str:
    """校验路径字段：不以 '-' 开头、不含 '..'、仅允许安全字符。"""
    if value.startswith("-"):
        raise ValueError(f"{field_name} 不得以 '-' 开头")
    if ".." in value:
        raise ValueError(f"{field_name} 不得包含 '..'")
    if not _SAFE_PATH_PATTERN.match(value):
        raise ValueError(f"{field_name} 包含非法字符")
    return value


def _validate_optional_safe_path(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_safe_path(value, field_name)


def _check_dataset_id(v: str) -> str:
    if not DATASET_ID_PATTERN.match(v):
        raise ValueError("dataset_id 格式非法")
    return v


def _check_config(v: str) -> str:
    return _validate_safe_path(v, "config")


class DataPrepareRequest(BaseModel):
    dataset_id: str = "aistpp"
    config: str = "configs/data.yaml"
    download_annotations: bool = True
    extract_annotations: bool = True
    download_videos: bool = False
    video_limit: int = 120
    agree_license: bool = False
    preprocess_limit: int = 0

    check_dataset_id = field_validator("dataset_id")(_check_dataset_id)
    check_config = field_validator("config")(_check_config)


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

    check_dataset_id = field_validator("dataset_id")(_check_dataset_id)
    check_config = field_validator("config")(_check_config)

    @field_validator("input_dir")
    @classmethod
    def check_input_dir(cls, v: str | None) -> str | None:
        return _validate_optional_safe_path(v, "input_dir")

    @field_validator("out_dir")
    @classmethod
    def check_out_dir(cls, v: str | None) -> str | None:
        return _validate_optional_safe_path(v, "out_dir")

    @field_validator("video_ext")
    @classmethod
    def check_video_ext(cls, v: str) -> str:
        if not _SAFE_EXT_PATTERN.match(v):
            raise ValueError("video_ext 格式非法，仅允许 1-6 位小写字母数字")
        return v

    @field_validator("weights")
    @classmethod
    def check_weights(cls, v: str) -> str:
        return _validate_safe_path(v, "weights")


class TrainRequest(BaseModel):
    dataset_id: str = "aistpp"
    config: str = "configs/train.yaml"
    yolo2d_dir: str | None = None
    gt3d_dir: str | None = None
    artifact_dir: str | None = None
    export_onnx: bool = False

    check_dataset_id = field_validator("dataset_id")(_check_dataset_id)
    check_config = field_validator("config")(_check_config)

    @field_validator("yolo2d_dir", "gt3d_dir", "artifact_dir")
    @classmethod
    def check_optional_dirs(cls, v: str | None) -> str | None:
        return _validate_optional_safe_path(v, "dir")


class MultiViewRequest(BaseModel):
    config: str = "configs/multiview.yaml"
    limit_sessions: int = 0

    check_config = field_validator("config")(_check_config)


class MultiViewTriangulateRequest(BaseModel):
    config: str = "configs/multiview.yaml"
    calibration: str | None = None
    limit_sessions: int = 0

    check_config = field_validator("config")(_check_config)

    @field_validator("calibration")
    @classmethod
    def check_calibration(cls, v: str | None) -> str | None:
        return _validate_optional_safe_path(v, "calibration")


class EvaluateRequest(BaseModel):
    dataset_id: str = "aistpp"
    input_dir: str = "data/raw/aistpp/videos"
    style: str = "gBR"
    max_videos: int = 10
    output_csv: str = "outputs/eval/summary.csv"

    check_dataset_id = field_validator("dataset_id")(_check_dataset_id)

    @field_validator("input_dir", "output_csv")
    @classmethod
    def check_paths(cls, v: str) -> str:
        return _validate_safe_path(v, "path")

    @field_validator("style")
    @classmethod
    def check_style(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]{1,32}$", v):
            raise ValueError("style 格式非法")
        return v


class DatasetUpsertRequest(BaseModel):
    id: str
    name: str
    stage: str = "planned"
    mode: str = "singleview"
    data_config: str = ""
    train_config: str = ""
    video_root: str = ""
    notes: str = ""


CAMERA_TOKEN_PATTERN = re.compile(r"_c\d+_")
PREVIEW_YOLO_WEIGHTS = os.environ.get("POSEMENTOR_PREVIEW_YOLO_WEIGHTS", "yolo11m-pose.pt")
PREVIEW_YOLO_CONF = float(os.environ.get("POSEMENTOR_PREVIEW_YOLO_CONF", "0.35"))

_preview_pose_model = None
_preview_pose_model_lock = threading.Lock()
_preview_group_locks: dict[str, threading.Lock] = {}
_preview_group_locks_guard = threading.Lock()


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


def _standard_priority(item: object) -> tuple[int, str]:
    if not isinstance(item, dict):
        return (9, "")
    standard_id = str(item.get("id", "")).strip().lower()
    name = str(item.get("name", "")).strip().lower()
    source = str(item.get("source", "")).strip().lower()
    stage = str(item.get("stage", "")).strip().lower()
    is_aist = "aist" in standard_id or "aist" in name
    if is_aist and stage == "active":
        return (0, standard_id)
    if is_aist:
        return (1, standard_id)
    if source == "private" and stage == "active":
        return (2, standard_id)
    if source == "private":
        return (3, standard_id)
    return (4, standard_id)


def _read_standard_registry() -> dict:
    if not STANDARD_REGISTRY_FILE.exists():
        return {"standards": []}
    data = load_yaml(STANDARD_REGISTRY_FILE)
    if not isinstance(data, dict):
        return {"standards": []}
    standards = data.get("standards", [])
    if not isinstance(standards, list):
        standards = []
    ordered = sorted(standards, key=_standard_priority)
    return {"standards": ordered}


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
    if _find_dataset(dataset_id) is None:
        raise HTTPException(status_code=400, detail=f"未知 dataset_id: {dataset_id}")


def _get_dataset_or_404(dataset_id: str) -> dict:
    """查找 dataset 并返回，不存在则抛出 400。合并了 _assert + _find 两步操作。"""
    dataset = _find_dataset(dataset_id)
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=400, detail=f"未知 dataset_id: {dataset_id}")
    return dataset


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
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
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
    status = job.status
    phase = "generic"
    if "train_3d_lift" in job.name:
        phase = "train"
    elif "evaluate_model" in job.name:
        phase = "evaluate"
    elif "pose_extract" in job.name:
        phase = "extract"
    elif "data_prepare" in job.name:
        phase = "prepare"
    elif "multiview_prepare" in job.name or "multiview_triangulate" in job.name:
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


def _resolve_existing_gt_seq_id(gt_dir: Path, candidates: list[str]) -> str | None:
    for seq_id in candidates:
        if not seq_id:
            continue
        seq_file = gt_dir / f"{seq_id}.npz"
        if seq_file.exists():
            return seq_id
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


def _preview_video_cache_valid(video_path: Path) -> bool:
    if not video_path.exists():
        return False
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        cap.release()


def _get_preview_pose_model():
    global _preview_pose_model  # noqa: PLW0603
    with _preview_pose_model_lock:
        if _preview_pose_model is None:
            from ultralytics import YOLO

            _preview_pose_model = YOLO(PREVIEW_YOLO_WEIGHTS)
        return _preview_pose_model


def _try_infer_3d_from_model(
    keypoints2d: np.ndarray,
    fps: float,
    checkpoint: str = "artifacts/lift_demo.ckpt",
    norm_file: str = "artifacts/lift_demo_norm.npz",
) -> np.ndarray | None:
    """用已训练的 PoseLiftTransformer 从 2D 关键点推理 3D 骨架。"""
    ckpt_path = PROJECT_ROOT / checkpoint
    norm_path = PROJECT_ROOT / norm_file
    if not ckpt_path.exists() or not norm_path.exists():
        logger.debug("推理模型不存在: %s / %s", ckpt_path, norm_path)
        return None

    try:
        import torch

        from posementor.models.lift_net import PoseLiftTransformer
        from posementor.utils.math3d import center_pose

        # 加载归一化参数
        with np.load(norm_path) as nf:
            norm_mean = nf["mean_2d"].astype(np.float32)
            norm_std = np.clip(nf["std_2d"].astype(np.float32), 1e-6, None)

        # 加载模型
        device = torch.device("cpu")
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state_dict = state.get("state_dict", state)
        cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}
        # 从 checkpoint 推断模型参数
        proj_key = next((k for k in cleaned if "input_proj.weight" in k), None)
        hidden_dim = int(cleaned[proj_key].shape[0]) if proj_key else 256
        pos_key = next((k for k in cleaned if "time_pos_embed" in k), None)
        max_seq_len = int(cleaned[pos_key].shape[1]) if pos_key else 81
        model = PoseLiftTransformer(hidden_dim=hidden_dim, max_seq_len=max_seq_len)
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        model.to(device)

        # 处理多人数据：取第一人
        kp = keypoints2d[:, 0, :, :2] if keypoints2d.ndim == 4 else keypoints2d[:, :, :2]
        kp = kp.astype(np.float32)
        kp_norm = (kp - norm_mean) / norm_std

        # 滑窗推理
        seq_len = 81
        frames_3d: list[np.ndarray] = []
        total = len(kp_norm)
        with torch.no_grad():
            for start in range(0, total, seq_len // 2):
                end = min(start + seq_len, total)
                window = kp_norm[start:end]
                if len(window) < seq_len:
                    pad = np.zeros((seq_len - len(window), *window.shape[1:]), dtype=np.float32)
                    window = np.concatenate([window, pad], axis=0)
                x = torch.from_numpy(window[None]).float().to(device)
                pred = model(x).cpu().numpy()[0]
                valid_len = min(seq_len, end - start)
                frames_3d.append(pred[:valid_len])

        joints3d = np.concatenate(frames_3d, axis=0)[:total]
        joints3d = center_pose(joints3d).astype(np.float32)
        logger.info("3D 推理完成: %d 帧, checkpoint=%s", len(joints3d), checkpoint)
        return joints3d
    except Exception:
        logger.warning("3D 推理失败", exc_info=True)
        return None


def _extract_pose2d_from_video(
    video_path: Path, max_persons: int = 1,
) -> tuple[np.ndarray, float]:
    """提取视频 2D 关键点。max_persons=1 时返回 [T,17,3]，>1 时返回 [T,P,17,3]。"""
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
            if max_persons <= 1:
                frames.append(np.zeros((17, 3), dtype=np.float32))
            else:
                frames.append(np.zeros((max_persons, 17, 3), dtype=np.float32))
            continue
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()
        if kp_conf.shape[0] == 0:
            if max_persons <= 1:
                frames.append(np.zeros((17, 3), dtype=np.float32))
            else:
                frames.append(np.zeros((max_persons, 17, 3), dtype=np.float32))
            continue

        if max_persons <= 1:
            person_idx = int(np.argmax(kp_conf.mean(axis=1)))
            kp = np.concatenate([kp_xy[person_idx], kp_conf[person_idx, :, None]], axis=-1)
            frames.append(kp.astype(np.float32))
        else:
            # 按置信度降序排列，保留 max_persons 个人
            scores = kp_conf.mean(axis=1)
            order = np.argsort(-scores)[:max_persons]
            person_kps = np.zeros((max_persons, 17, 3), dtype=np.float32)
            for slot, pidx in enumerate(order):
                person_kps[slot] = np.concatenate(
                    [kp_xy[pidx], kp_conf[pidx, :, None]], axis=-1,
                ).astype(np.float32)
            frames.append(person_kps)
    if not frames:
        raise RuntimeError(f"视频无有效帧: {video_path}")
    return np.stack(frames, axis=0), fps


def _get_preview_group_lock(group_key: str) -> threading.Lock:
    with _preview_group_locks_guard:
        lock = _preview_group_locks.get(group_key)
        if lock is None:
            lock = threading.Lock()
            _preview_group_locks[group_key] = lock
        return lock



def _build_preview_cache_key(paths: list[Path]) -> str:
    existing_times = [path.stat().st_mtime_ns for path in paths if path.exists()]
    return str(int(max(existing_times, default=0)))



def _preview_pipeline_mtime_ns() -> int:
    files = {
        Path(__file__),
        Path(render_pose_preview_videos.__code__.co_filename),
        Path(build_pose2d_preview_data.__code__.co_filename),
    }
    return max(path.stat().st_mtime_ns for path in files if path.exists())



def _workspace_pose_preview_aist(
    dataset_id: str,
    source_video: Path,
    rel_data_video: Path,
    dataset_video_root: Path,
    gt_dir: Path,
    source_seq_id: str,
    source_stem: str,
    refresh: bool,
) -> dict[str, object] | None:
    if dataset_id != "aistpp":
        return None
    if not AIST_ANNOTATIONS_ROOT.exists():
        return None

    camera_id = extract_camera_id(source_stem)
    if not camera_id:
        return None
    group_seq_id = resolve_group_seq_id(source_stem)
    if group_seq_id == source_stem:
        return None

    gt_seq_id = _resolve_existing_gt_seq_id(
        gt_dir=gt_dir,
        candidates=[group_seq_id, source_seq_id, source_stem],
    )
    if not gt_seq_id:
        return None
    gt_file = gt_dir / f"{gt_seq_id}.npz"
    if not gt_file.exists():
        return None

    group_video_paths = collect_group_video_paths(dataset_video_root, group_seq_id)
    if camera_id not in group_video_paths:
        return None

    align_cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id / "alignment")
    alignment_file = align_cache_dir / f"{group_seq_id}_alignment.json"
    keypoints2d_file = AIST_ANNOTATIONS_ROOT / "keypoints2d" / f"{group_seq_id}.pkl"
    if not keypoints2d_file.exists():
        return None

    group_lock = _get_preview_group_lock(f"{dataset_id}:{group_seq_id}")
    with group_lock:
        alignment_meta = load_aist_alignment_meta(
            group_seq_id=group_seq_id,
            annotations_root=str(AIST_ANNOTATIONS_ROOT),
            cache_dir=str(align_cache_dir),
            refresh=refresh,
        )
        with np.load(gt_file) as gt_data:
            joints3d = gt_data["joints3d"].astype(np.float32)

        video_stats = {
            item_camera_id: read_video_stats(item_path)
            for item_camera_id, item_path in group_video_paths.items()
        }
        preview_plan = plan_group_preview(
            alignment=alignment_meta,
            video_stats=video_stats,
            joints3d_frame_total=int(joints3d.shape[0]),
        )
        frame_total = int(preview_plan["frame_total"])
        if frame_total <= 0:
            raise HTTPException(status_code=400, detail=f"AIST 对齐后无可用帧: {group_seq_id}")
        timeline_start_frame = int(preview_plan["timeline_start_frame"])
        camera_trim_start = int(preview_plan["camera_trim_start"][camera_id])

        keypoints2d_all, _timeline_fps = load_group_keypoints2d(
            group_seq_id,
            str(AIST_ANNOTATIONS_ROOT),
        )
        camera_index = int(camera_id[1:]) - 1
        keypoints2d = keypoints2d_all[
            camera_index,
            camera_trim_start : camera_trim_start + frame_total,
        ].astype(np.float32)
        joints3d_view = joints3d[
            timeline_start_frame : timeline_start_frame + frame_total,
        ].astype(np.float32)
        if keypoints2d.shape[0] != frame_total or joints3d_view.shape[0] != frame_total:
            raise HTTPException(status_code=400, detail=f"AIST 预览裁切失败: {group_seq_id}")

        cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id)
        output_source = cache_dir / f"{source_seq_id}_source.mp4"
        output_2d_data = cache_dir / f"{source_seq_id}_pose2d.json"
        output_3d_data = cache_dir / f"{group_seq_id}_pose3d.json"

        dep_mtime = max(
            source_video.stat().st_mtime_ns,
            gt_file.stat().st_mtime_ns,
            keypoints2d_file.stat().st_mtime_ns,
            alignment_file.stat().st_mtime_ns if alignment_file.exists() else 0,
            _preview_pipeline_mtime_ns(),
        )
        need_render = bool(refresh) or not output_source.exists()
        if output_source.exists() and not need_render:
            need_render = output_source.stat().st_mtime_ns < dep_mtime
        if output_source.exists() and not need_render:
            need_render = not _preview_video_cache_valid(output_source)

        stats: dict[str, float] = {
            "fps": float(video_stats[camera_id]["fps"]),
            "frames": float(frame_total),
        }
        if need_render:
            stats = render_source_preview_video(
                source_video=source_video,
                output_source=output_source,
                source_frame_offset=camera_trim_start,
                frame_total=frame_total,
            )

        source_width = int(video_stats[camera_id]["width"])
        source_height = int(video_stats[camera_id]["height"])
        expected_frame_total = int(stats["frames"])

        need_export_pose2d_data = bool(refresh) or not output_2d_data.exists()
        if output_2d_data.exists() and not need_export_pose2d_data:
            need_export_pose2d_data = output_2d_data.stat().st_mtime_ns < dep_mtime
        if output_2d_data.exists() and not need_export_pose2d_data:
            try:
                pose2d_meta = json.loads(output_2d_data.read_text(encoding="utf-8"))
                need_export_pose2d_data = any(
                    [
                        int(pose2d_meta.get("frame_count", -1)) != expected_frame_total,
                        int(pose2d_meta.get("joint_count", -1)) != int(keypoints2d.shape[1]),
                        int(pose2d_meta.get("frame_width", -1)) != source_width,
                        int(pose2d_meta.get("frame_height", -1)) != source_height,
                    ]
                )
            except Exception:  # noqa: BLE001
                logger.debug("pose2d 缓存元数据校验失败，将重新导出", exc_info=True)
                need_export_pose2d_data = True
        if need_export_pose2d_data or need_render:
            pose2d_data = build_pose2d_preview_data(
                keypoints2d=keypoints2d,
                fps=float(stats["fps"]),
                frame_width=source_width,
                frame_height=source_height,
                frame_total=expected_frame_total,
            )
            output_2d_data.write_text(
                json.dumps(pose2d_data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

        need_export_pose3d_data = bool(refresh) or not output_3d_data.exists()
        if output_3d_data.exists() and not need_export_pose3d_data:
            need_export_pose3d_data = output_3d_data.stat().st_mtime_ns < dep_mtime
        if output_3d_data.exists() and not need_export_pose3d_data:
            try:
                pose3d_meta = json.loads(output_3d_data.read_text(encoding="utf-8"))
                need_export_pose3d_data = any(
                    [
                        int(pose3d_meta.get("preview_version", 0)) != POSE3D_PREVIEW_VERSION,
                        int(pose3d_meta.get("frame_count", -1)) != expected_frame_total,
                        int(pose3d_meta.get("joint_count", -1)) != int(joints3d_view.shape[1]),
                    ]
                )
            except Exception:  # noqa: BLE001
                logger.debug("pose3d 缓存元数据校验失败，将重新导出", exc_info=True)
                need_export_pose3d_data = True
        if need_export_pose3d_data or need_render:
            pose3d_data = build_pose3d_preview_data(
                joints3d=joints3d_view,
                fps=float(stats["fps"]),
                frame_total=expected_frame_total,
            )
            output_3d_data.write_text(
                json.dumps(pose3d_data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

        alignment_payload = {
            **preview_plan,
            "camera_id": camera_id,
            "alignment_data_url": (
                f"/outputs-files/preview_cache/{dataset_id}/alignment/{alignment_file.name}"
            ),
            "current_camera": {
                "camera_id": camera_id,
                "offset_frames": int(preview_plan["camera_offsets"][camera_id]),
                "trim_start": int(preview_plan["camera_trim_start"][camera_id]),
                "frame_count": int(preview_plan["camera_frame_count"][camera_id]),
                "sync_error_px": float(preview_plan["camera_sync_error_px"].get(camera_id, 0.0)),
                "geometry": preview_plan["camera_geometry"].get(camera_id),
            },
        }
        cache_key = _build_preview_cache_key(
            [output_source, output_2d_data, output_3d_data, alignment_file]
        )
        return {
            "dataset_id": dataset_id,
            "seq_id": group_seq_id,
            "camera_id": camera_id,
            "source_video_url": f"/outputs-files/preview_cache/{dataset_id}/{output_source.name}",
            "pose2d_video_url": "",
            "pose2d_data_url": f"/outputs-files/preview_cache/{dataset_id}/{output_2d_data.name}",
            "pose3d_video_url": "",
            "pose3d_data_url": f"/outputs-files/preview_cache/{dataset_id}/{output_3d_data.name}",
            "cache_key": cache_key,
            "fps": stats["fps"],
            "frames": stats["frames"],
            "alignment": alignment_payload,
            "source_video_fallback_url": f"/data-files/{rel_data_video.as_posix()}",
        }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "posementor-backend",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@router.get("/health")
def health() -> dict[str, str]:
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


@router.get("/jobs")
def list_jobs() -> dict[str, list[dict[str, object]]]:
    return {"jobs": [_job_to_dict(job) for job in store.list_jobs()]}


@router.get("/datasets")
def list_datasets() -> dict:
    return _dataset_registry_payload()


@router.get("/standards")
def list_standards() -> dict:
    return _read_standard_registry()


@router.get("/workspace/source-preview")
def source_preview(dataset_id: str = "aistpp", limit: int = 3) -> dict[str, object]:
    dataset = _get_dataset_or_404(dataset_id)
    bounded_limit = max(1, min(limit, 500))
    normalized = _normalize_dataset_item(dataset)
    video_root = _resolve_dataset_video_root(normalized)
    preview_cache_dir = OUTPUT_ROOT / "preview_cache" / dataset_id
    yolo_dir, gt_dir = _resolve_pose_dirs_from_dataset(normalized)

    samples: list[dict[str, object]] = []
    if video_root.exists():
        candidates = sorted(video_root.rglob("*.mp4"))
        grouped: dict[str, list[Path]] = {}
        is_multiview = str(dataset.get("mode", "")).strip() == "multiview"
        for path in candidates:
            if is_multiview:
                try:
                    key = path.parent.relative_to(video_root).as_posix() or path.parent.name
                except ValueError:
                    key = path.parent.name
            else:
                key = _video_group_key(path)
            grouped.setdefault(key, []).append(path)

        selected_keys = sorted(grouped.keys())[:bounded_limit]
        for group_key in selected_keys:
            group_rows: list[dict[str, object]] = []
            for path in sorted(grouped[group_key]):
                camera_match = re.search(r"_c(\d+)_", path.stem)
                camera_id = f"c{camera_match.group(1)}" if camera_match else path.stem.lower()
                rel_project = _to_project_relative(path)
                stat = path.stat()
                url = ""
                if path.is_relative_to(DATA_ROOT):
                    rel_data = path.relative_to(DATA_ROOT).as_posix()
                    url = f"/data-files/{rel_data}"
                seq_id = build_video_seq_id(video_root=video_root, video_path=path)
                group_seq_id = CAMERA_TOKEN_PATTERN.sub("_cAll_", seq_id)
                pose2d_file = preview_cache_dir / f"{seq_id}_pose2d.mp4"
                pose3d_file = preview_cache_dir / f"{group_seq_id}_pose3d.mp4"
                official_pose2d_file = AIST_ANNOTATIONS_ROOT / "keypoints2d" / f"{group_seq_id}.pkl"
                group_rows.append(
                    {
                        "name": path.name,
                        "path": rel_project,
                        "url": url,
                        "size_bytes": stat.st_size,
                        "group_key": group_key,
                        "camera_id": camera_id,
                        "pose2d_exists": pose2d_file.exists()
                        or (yolo_dir / f"{seq_id}.npz").exists()
                        or (dataset_id == "aistpp" and official_pose2d_file.exists()),
                        "pose3d_exists": pose3d_file.exists()
                        or (gt_dir / f"{seq_id}.npz").exists()
                        or (gt_dir / f"{group_seq_id}.npz").exists(),
                    }
                )
            group_ready = all(
                bool(row.get("pose2d_exists")) and bool(row.get("pose3d_exists"))
                for row in group_rows
            )
            if not group_ready:
                for row in group_rows:
                    row["pose2d_exists"] = False
                    row["pose3d_exists"] = False
            samples.extend(group_rows)

    return {
        "dataset_id": dataset_id,
        "video_root": _to_project_relative(video_root),
        "samples": samples,
    }


@router.get("/workspace/pose-preview")
def workspace_pose_preview(
    dataset_id: str,
    video_path: str,
    refresh: bool = False,
    model: str = "artifacts/lift_demo.ckpt",
) -> dict[str, object]:
    dataset = _get_dataset_or_404(dataset_id)
    normalized = _normalize_dataset_item(dataset)

    # 路径安全校验: 先检查路径组件再 resolve，防止 path traversal 和 symlink 逃逸
    video_path_obj = Path(video_path)
    if ".." in video_path_obj.parts or any(p.startswith("-") for p in video_path_obj.parts):
        raise HTTPException(status_code=400, detail="无效的 video_path")
    source_video = (PROJECT_ROOT / video_path).resolve()
    # 先做安全边界检查再暴露文件是否存在，避免 oracle
    try:
        rel_data_video = source_video.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="无效的 video_path") from exc
    if not source_video.exists() or not source_video.is_file():
        raise HTTPException(status_code=404, detail="无效的 video_path")

    source_name = source_video.name
    source_stem = source_video.stem
    dataset_video_root = _resolve_dataset_video_root(normalized)
    source_video_rel = build_video_rel_path(video_root=dataset_video_root, video_path=source_video)
    source_seq_id = build_video_seq_id(video_root=dataset_video_root, video_path=source_video)

    yolo_dir, gt_dir = _resolve_pose_dirs_from_dataset(normalized)
    aist_preview_payload = _workspace_pose_preview_aist(
        dataset_id=dataset_id,
        source_video=source_video,
        rel_data_video=rel_data_video,
        dataset_video_root=dataset_video_root,
        gt_dir=gt_dir,
        source_seq_id=source_seq_id,
        source_stem=source_stem,
        refresh=refresh,
    )
    if aist_preview_payload is not None:
        return aist_preview_payload
    # 尝试查找 3D GT 数据；自定义视频可能没有，此时仅返回 2D 预览
    has_gt3d = False
    joints3d: np.ndarray | None = None
    gt_file: Path | None = None
    gt_seq_id: str = ""
    fallback_seq_id = ""
    if yolo_dir.exists() and gt_dir.exists():
        fallback_seq_id = find_sequence_id(
            yolo2d_dir=yolo_dir,
            video_stem=source_stem,
            source_video_name=source_name,
            source_video_rel=source_video_rel,
        )
        gt_seq_id = _resolve_existing_gt_seq_id(
            gt_dir=gt_dir,
            candidates=[
                source_seq_id,
                source_stem,
                CAMERA_TOKEN_PATTERN.sub("_cAll_", source_stem),
                fallback_seq_id,
                CAMERA_TOKEN_PATTERN.sub("_cAll_", fallback_seq_id) if fallback_seq_id else "",
            ],
        )
        if gt_seq_id:
            gt_file = gt_dir / f"{gt_seq_id}.npz"
            if gt_file.exists():
                with np.load(gt_file) as gt_data:
                    joints3d = gt_data["joints3d"].astype(np.float32)
                has_gt3d = True

    preview_pose_cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id / "pose2d_npz")
    source_pose2d_cache = preview_pose_cache_dir / f"{source_seq_id}.npz"
    source_pose2d_file = yolo_dir / f"{source_seq_id}.npz"
    direct_pose2d_file = yolo_dir / f"{source_stem}.npz"
    fallback_pose2d_file = yolo_dir / f"{fallback_seq_id}.npz" if fallback_seq_id else None

    keypoints2d: np.ndarray
    fps_value: float
    pose2d_dep_file: Path
    if source_pose2d_file.exists():
        keypoints2d, fps_value = _load_keypoints2d(source_pose2d_file)
        pose2d_dep_file = source_pose2d_file
    elif direct_pose2d_file.exists():
        keypoints2d, fps_value = _load_keypoints2d(direct_pose2d_file)
        pose2d_dep_file = direct_pose2d_file
    elif source_pose2d_cache.exists():
        keypoints2d, fps_value = _load_keypoints2d(source_pose2d_cache)
        pose2d_dep_file = source_pose2d_cache
    else:
        try:
            is_inference = str(dataset.get("stage", "")).strip() == "inference"
            max_persons = 6 if is_inference else 1
            keypoints2d, fps_value = _extract_pose2d_from_video(
                source_video, max_persons=max_persons,
            )
            np.savez_compressed(
                source_pose2d_cache,
                keypoints2d=keypoints2d,
                fps=np.array(fps_value, dtype=np.float32),
                source=np.array(source_name),
            )
            pose2d_dep_file = source_pose2d_cache
        except Exception as exc:
            if fallback_pose2d_file is None or not fallback_pose2d_file.exists():
                if not fallback_seq_id:
                    raise HTTPException(
                        status_code=404,
                        detail=f"未找到视频对应 2D 关键点: {source_name}",
                    ) from exc
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到 2D 文件: seq_id={fallback_seq_id}",
                ) from exc
            keypoints2d, fps_value = _load_keypoints2d(fallback_pose2d_file)
            pose2d_dep_file = fallback_pose2d_file

    cache_dir = ensure_dir(OUTPUT_ROOT / "preview_cache" / dataset_id)
    cache_stem = source_seq_id
    output_source = cache_dir / f"{cache_stem}_source.mp4"
    output_2d = cache_dir / f"{cache_stem}_pose2d.mp4"
    output_2d_data = cache_dir / f"{cache_stem}_pose2d.json"
    output_3d = cache_dir / f"{cache_stem}_pose3d.mp4"
    output_3d_data = cache_dir / f"{cache_stem}_pose3d.json"
    source_cap = cv2.VideoCapture(str(source_video))
    source_width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
    source_height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
    source_cap.release()
    source_mtime = source_video.stat().st_mtime_ns
    dep_mtime_parts = [pose2d_dep_file.stat().st_mtime_ns, source_mtime, _preview_pipeline_mtime_ns()]
    if gt_file is not None and gt_file.exists():
        dep_mtime_parts.append(gt_file.stat().st_mtime_ns)
    dep_mtime = max(dep_mtime_parts)

    # 检查需要渲染的输出文件列表（无 3D GT 时跳过 3D 视频）
    required_outputs = [output_source, output_2d]
    if has_gt3d:
        required_outputs.append(output_3d)

    need_render = bool(refresh)
    if not all(p.exists() for p in required_outputs):
        need_render = True
    elif not need_render:
        output_mtime = min(p.stat().st_mtime_ns for p in required_outputs)
        if output_mtime < dep_mtime:
            need_render = True
        elif not all(_preview_video_cache_valid(p) for p in required_outputs):
            need_render = True

    # 无 GT3D 时尝试用已训练模型推理 3D
    if not has_gt3d:
        # model 参数对应 ckpt 文件路径，norm 文件取同名 _norm.npz
        norm_file = model.replace(".ckpt", "_norm.npz") if model.endswith(".ckpt") else "artifacts/lift_demo_norm.npz"
        inferred_3d = _try_infer_3d_from_model(keypoints2d, fps_value, checkpoint=model, norm_file=norm_file)
        if inferred_3d is not None:
            joints3d = inferred_3d
            has_gt3d = True  # 标记为有 3D 数据（推理得到的）

    frame_count = len(keypoints2d) if joints3d is None else min(len(keypoints2d), len(joints3d))
    stats: dict[str, float] = {
        "fps": max(0.0, fps_value),
        "frames": float(frame_count),
    }

    # 多人数据需要取第一人用于 render（render 只支持单人 [T,J,C]）
    render_kp2d = keypoints2d[:, 0] if keypoints2d.ndim == 4 else keypoints2d

    if need_render:
        if has_gt3d and joints3d is not None:
            stats = render_pose_preview_videos(
                source_video=source_video,
                keypoints2d=render_kp2d,
                joints3d=joints3d,
                output_source=output_source,
                output_2d=output_2d,
                output_3d=output_3d,
            )
        else:
            dummy_3d = np.zeros((len(render_kp2d), render_kp2d.shape[1], 3), dtype=np.float32)
            render_pose_preview_videos(
                source_video=source_video,
                keypoints2d=render_kp2d,
                joints3d=dummy_3d,
                output_source=output_source,
                output_2d=output_2d,
                output_3d=output_3d,
            )
            stats = {"fps": max(0.0, fps_value), "frames": float(len(render_kp2d))}

    expected_frame_total = int(stats["frames"])
    need_export_pose2d_data = bool(refresh) or not output_2d_data.exists()
    if output_2d_data.exists() and not need_export_pose2d_data:
        need_export_pose2d_data = output_2d_data.stat().st_mtime_ns < dep_mtime
    if output_2d_data.exists() and not need_export_pose2d_data:
        try:
            pose2d_meta = json.loads(output_2d_data.read_text(encoding="utf-8"))
            need_export_pose2d_data = any(
                [
                    int(pose2d_meta.get("frame_count", -1)) != expected_frame_total,
                    int(pose2d_meta.get("joint_count", -1)) != int(keypoints2d.shape[1]),
                    int(pose2d_meta.get("frame_width", -1)) != source_width,
                    int(pose2d_meta.get("frame_height", -1)) != source_height,
                ]
            )
        except Exception:  # noqa: BLE001
            logger.debug("pose2d 缓存元数据校验失败，将重新导出", exc_info=True)
            need_export_pose2d_data = True
    if need_export_pose2d_data or need_render:
        pose2d_data = build_pose2d_preview_data(
            keypoints2d=keypoints2d,
            fps=float(stats["fps"]),
            frame_width=source_width,
            frame_height=source_height,
            frame_total=expected_frame_total,
        )
        output_2d_data.write_text(
            json.dumps(pose2d_data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    if has_gt3d and joints3d is not None:
        need_export_pose3d_data = bool(refresh) or not output_3d_data.exists()
        if output_3d_data.exists() and not need_export_pose3d_data:
            need_export_pose3d_data = output_3d_data.stat().st_mtime_ns < dep_mtime
        if output_3d_data.exists() and not need_export_pose3d_data:
            try:
                pose3d_meta = json.loads(output_3d_data.read_text(encoding="utf-8"))
                need_export_pose3d_data = any(
                    [
                        int(pose3d_meta.get("preview_version", 0)) != POSE3D_PREVIEW_VERSION,
                        int(pose3d_meta.get("frame_count", -1)) != expected_frame_total,
                        int(pose3d_meta.get("joint_count", -1)) != int(joints3d.shape[1]),
                    ]
                )
            except Exception:  # noqa: BLE001
                logger.debug("pose3d 缓存元数据校验失败，将重新导出", exc_info=True)
                need_export_pose3d_data = True
        if need_export_pose3d_data or need_render:
            pose3d_data = build_pose3d_preview_data(
                joints3d=joints3d,
                fps=float(stats["fps"]),
                frame_total=expected_frame_total,
            )
            output_3d_data.write_text(
                json.dumps(pose3d_data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

    source_video_url = f"/data-files/{rel_data_video.as_posix()}"
    if output_source.exists():
        source_video_url = f"/outputs-files/preview_cache/{dataset_id}/{output_source.name}"

    cache_mtime_parts = [
        output_source.stat().st_mtime_ns if output_source.exists() else source_video.stat().st_mtime_ns,
        output_2d.stat().st_mtime_ns if output_2d.exists() else 0,
        output_2d_data.stat().st_mtime_ns if output_2d_data.exists() else 0,
    ]
    if output_3d.exists():
        cache_mtime_parts.append(output_3d.stat().st_mtime_ns)
    if output_3d_data.exists():
        cache_mtime_parts.append(output_3d_data.stat().st_mtime_ns)
    cache_key = str(int(max(cache_mtime_parts)))

    result: dict[str, object] = {
        "dataset_id": dataset_id,
        "seq_id": gt_seq_id if has_gt3d else source_seq_id,
        "source_video_url": source_video_url,
        "pose2d_video_url": f"/outputs-files/preview_cache/{dataset_id}/{output_2d.name}",
        "pose2d_data_url": f"/outputs-files/preview_cache/{dataset_id}/{output_2d_data.name}",
        "pose3d_video_url": f"/outputs-files/preview_cache/{dataset_id}/{output_3d.name}" if has_gt3d else "",
        "pose3d_data_url": f"/outputs-files/preview_cache/{dataset_id}/{output_3d_data.name}" if has_gt3d else "",
        "cache_key": cache_key,
        "fps": stats["fps"],
        "frames": stats["frames"],
    }
    return result


@router.post("/datasets/upsert")
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


@router.get("/artifacts/models")
def list_models() -> dict[str, list[dict[str, object]]]:
    """列出可用于 3D 推理的模型文件。"""
    models: list[dict[str, object]] = []
    for ckpt in sorted(ARTIFACT_ROOT.glob("*.ckpt")):
        norm = ckpt.with_name(ckpt.stem.replace(".ckpt", "") + "_norm.npz")
        # 只保留 lift_demo 主模型，跳过 epoch 版本
        if "epoch=" in ckpt.name:
            continue
        models.append({
            "name": ckpt.stem,
            "path": f"artifacts/{ckpt.name}",
            "norm_exists": norm.exists(),
            "size_bytes": ckpt.stat().st_size,
        })
    return {"models": models}


@router.get("/artifacts/status")
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


@router.get("/artifacts/manifest")
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


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return _job_to_dict(job)


@router.get("/jobs/{job_id}/log")
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


@router.get("/jobs/{job_id}/progress")
def get_job_progress(job_id: str) -> dict[str, object]:
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc

    log_path = Path(job.log_path)
    log_text = _read_job_log_text(log_path, max_chars=200_000)
    return _parse_job_progress(job=job, log_text=log_text)


@router.post("/jobs/data/prepare")
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


@router.post("/jobs/pose/extract")
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


@router.post("/jobs/train")
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


@router.post("/jobs/multiview/prepare")
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


@router.post("/jobs/multiview/triangulate")
def start_multiview_triangulate(req: MultiViewTriangulateRequest) -> dict[str, str]:
    command = [
        "uv",
        "run",
        "python",
        "triangulate_multiview_dataset.py",
        "--config",
        req.config,
    ]
    if req.calibration:
        command.extend(["--calibration", req.calibration])
    if req.limit_sessions > 0:
        command.extend(["--limit-sessions", str(req.limit_sessions)])

    job_id = runner.submit(name="multiview_triangulate", command=command)
    return {"job_id": job_id}


@router.post("/jobs/evaluate")
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


# 将业务路由挂载到 / 和 /api 两个前缀，兼容前端直连和代理层转发
app.include_router(router)
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="PoseMentor Backend API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    cli_args = parser.parse_args()
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
