#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from posementor.infra.command_runner import JobRunner
from posementor.infra.job_store import JobRecord, JobStore

PROJECT_ROOT = Path(__file__).resolve().parent
JOB_ROOT = PROJECT_ROOT / "outputs" / "job_center"

store = JobStore(root=JOB_ROOT)
runner = JobRunner(store=store, cwd=PROJECT_ROOT)

app = FastAPI(title="PoseMentor Backend", version="0.1.0")


class DataPrepareRequest(BaseModel):
    config: str = "configs/data.yaml"
    download_annotations: bool = True
    extract_annotations: bool = True
    download_videos: bool = False
    video_limit: int = 120
    agree_license: bool = False
    preprocess_limit: int = 0


class ExtractRequest(BaseModel):
    config: str = "configs/data.yaml"
    weights: str = "yolo11m-pose.pt"
    conf: float = 0.35
    max_videos: int = 0


class TrainRequest(BaseModel):
    config: str = "configs/train.yaml"
    export_onnx: bool = True


class MultiViewRequest(BaseModel):
    config: str = "configs/multiview.yaml"
    limit_sessions: int = 0


class EvaluateRequest(BaseModel):
    input_dir: str = "data/raw/aistpp/videos"
    style: str = "gBR"
    max_videos: int = 10
    output_csv: str = "outputs/eval/summary.csv"


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


@app.post("/jobs/data/prepare")
def start_data_prepare(req: DataPrepareRequest) -> dict[str, str]:
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
    ]
    if req.max_videos > 0:
        command.extend(["--max-videos", str(req.max_videos)])

    job_id = runner.submit(name="pose_extract", command=command)
    return {"job_id": job_id}


@app.post("/jobs/train")
def start_train(req: TrainRequest) -> dict[str, str]:
    command = [
        "uv",
        "run",
        "python",
        "train_3d_lift_demo.py",
        "--config",
        req.config,
    ]
    if req.export_onnx:
        command.append("--export-onnx")

    job_id = runner.submit(name="train_3d_lift", command=command)
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

    job_id = runner.submit(name="evaluate_model", command=command)
    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="PoseMentor Backend API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    cli_args = parser.parse_args()
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
