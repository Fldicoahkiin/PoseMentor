#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from posementor.infra.command_runner import JobRunner
from posementor.infra.job_store import JobRecord, JobStore
from posementor.utils.io import ensure_dir, load_yaml, save_yaml

PROJECT_ROOT = Path(__file__).resolve().parent
JOB_ROOT = PROJECT_ROOT / "outputs" / "job_center"
DATASET_REGISTRY_FILE = PROJECT_ROOT / "configs" / "datasets.yaml"
STANDARD_REGISTRY_FILE = PROJECT_ROOT / "configs" / "standards.yaml"
ARTIFACT_ROOT = ensure_dir(PROJECT_ROOT / "artifacts")
DATA_ROOT = ensure_dir(PROJECT_ROOT / "data")

store = JobStore(root=JOB_ROOT)
runner = JobRunner(store=store, cwd=PROJECT_ROOT)

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
    bounded_limit = max(1, min(limit, 20))

    dataset = _find_dataset(dataset_id)
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=404, detail=f"dataset_id={dataset_id} 未注册")
    video_root = _resolve_dataset_video_root(_normalize_dataset_item(dataset))

    samples: list[dict[str, object]] = []
    if video_root.exists():
        candidates = sorted(video_root.rglob("*.mp4"))
        for path in candidates[:bounded_limit]:
            rel_project = _to_project_relative(path)
            stat = path.stat()
            url = ""
            if path.is_relative_to(DATA_ROOT):
                rel_data = path.relative_to(DATA_ROOT).as_posix()
                url = f"/data-files/{rel_data}"
            samples.append(
                {
                    "name": path.name,
                    "path": rel_project,
                    "url": url,
                    "size_bytes": stat.st_size,
                }
            )

    return {
        "dataset_id": dataset_id,
        "video_root": _to_project_relative(video_root),
        "samples": samples,
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
