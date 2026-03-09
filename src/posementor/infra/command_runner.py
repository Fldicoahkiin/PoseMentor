from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from posementor.infra.job_store import JobStore


class JobRunner:
    """后台任务执行器：串行触发、异步运行、日志落盘。"""

    def __init__(self, store: JobStore, cwd: Path, max_workers: int = 1) -> None:
        self.store = store
        self.cwd = cwd
        self.max_workers = max(1, int(max_workers))
        interrupted = self.store.mark_interrupted_jobs()
        if interrupted > 0:
            print(f"[INFO] 已标记 {interrupted} 个历史运行中任务为中断状态")
        self._queue: queue.Queue[tuple[str, list[str], dict[str, str]]] = queue.Queue()
        self._workers: list[threading.Thread] = []
        for idx in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"job-runner-{idx}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def submit(self, name: str, command: list[str], env: dict[str, str] | None = None) -> str:
        record = self.store.create_job(name=name, command=command)
        self._queue.put((record.job_id, command, env or {}))
        return record.job_id

    def _worker_loop(self) -> None:
        while True:
            job_id, command, env = self._queue.get()
            try:
                self._run(job_id, command, env)
            finally:
                self._queue.task_done()

    def _run(self, job_id: str, command: list[str], env: dict[str, str]) -> None:
        log_path = Path(self.store.get(job_id).log_path)
        self.store.update(job_id, status="running", started_at=time.time())

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(command)}\n")
            log_file.flush()

            try:
                process_env = dict(os.environ)
                process_env.setdefault("PYTHONUNBUFFERED", "1")
                if env:
                    process_env.update(env)
                process = subprocess.Popen(
                    command,
                    cwd=str(self.cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=process_env,
                )
            except Exception as exc:  # noqa: BLE001
                self.store.update(
                    job_id,
                    status="failed",
                    finished_at=time.time(),
                    return_code=-1,
                    error_message=str(exc),
                )
                log_file.write(f"启动失败: {exc}\n")
                return

            if process.stdout is not None:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()

            process.wait()
            if process.returncode == 0:
                self.store.update(
                    job_id,
                    status="success",
                    finished_at=time.time(),
                    return_code=int(process.returncode),
                )
            else:
                self.store.update(
                    job_id,
                    status="failed",
                    finished_at=time.time(),
                    return_code=int(process.returncode),
                    error_message=f"命令退出码: {process.returncode}",
                )
