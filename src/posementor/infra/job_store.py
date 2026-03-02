from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from posementor.utils.io import ensure_dir

JobStatus = Literal["queued", "running", "success", "failed"]


@dataclass(slots=True)
class JobRecord:
    job_id: str
    name: str
    command: list[str]
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    log_path: str = ""
    error_message: str = ""


class JobStore:
    """将作业状态持久化到本地 JSON，方便后台与前端共享。"""

    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)
        self.state_file = self.root / "jobs.json"
        self.log_dir = ensure_dir(self.root / "logs")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return

        for item in data:
            try:
                record = JobRecord(**item)
                self._jobs[record.job_id] = record
            except Exception:
                continue

    def _save(self) -> None:
        payload = [asdict(job) for job in sorted(self._jobs.values(), key=lambda x: x.created_at)]
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_job(self, name: str, command: list[str]) -> JobRecord:
        with self._lock:
            job_id = str(uuid.uuid4())
            log_path = str(self.log_dir / f"{job_id}.log")
            record = JobRecord(
                job_id=job_id,
                name=name,
                command=command,
                log_path=log_path,
            )
            self._jobs[job_id] = record
            self._save()
            return record

    def update(self, job_id: str, **kwargs: object) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(record, key, value)
            self._save()
            return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            return self._jobs[job_id]

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)
