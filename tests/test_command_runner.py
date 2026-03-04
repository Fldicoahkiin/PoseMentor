from __future__ import annotations

import sys
import time
from pathlib import Path

from posementor.infra.command_runner import JobRunner
from posementor.infra.job_store import JobStore


def _wait_job_done(store: JobStore, job_id: str, timeout: float = 8.0) -> str:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        status = store.get(job_id).status
        if status in {"success", "failed"}:
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job timeout: {job_id}")


def test_job_runner_default_single_worker_queue(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "job_center")
    runner = JobRunner(store=store, cwd=tmp_path, max_workers=1)

    command = [sys.executable, "-c", "import time; time.sleep(0.18); print('ok')"]

    start = time.perf_counter()
    job_ids = [
        runner.submit(name="job_a", command=command),
        runner.submit(name="job_b", command=command),
        runner.submit(name="job_c", command=command),
    ]

    for job_id in job_ids:
        assert _wait_job_done(store, job_id) == "success"

    elapsed = time.perf_counter() - start
    assert elapsed >= 0.45
