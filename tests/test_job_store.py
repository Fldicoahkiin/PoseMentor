from __future__ import annotations

from pathlib import Path

from posementor.infra.job_store import JobStore


def test_job_store_create_and_query(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path)
    job = store.create_job(name="unit", command=["echo", "ok"])

    fetched = store.get(job.job_id)
    assert fetched.job_id == job.job_id
    assert fetched.status == "queued"

    jobs = store.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].name == "unit"
