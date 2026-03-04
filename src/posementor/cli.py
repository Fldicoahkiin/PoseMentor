from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from posementor.local_config import init_local_config, load_local_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_FILE = PROJECT_ROOT / "configs" / "local.yaml"
BACKEND_SERVICE = "backend_api"
FRONTEND_SERVICE = "frontend"
LEGACY_FRONTEND_PATTERNS = ("app_demo.py", "admin_console.py")
VITE_PORT_PATTERN = re.compile(r"--port\s+(\d+)")


def _run_python_script(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, script, *extra_args]
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def _launcher_shell_text() -> str:
    return """#!/usr/bin/env sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ -x "$SCRIPT_DIR/.venv/bin/posementor" ]; then
  exec "$SCRIPT_DIR/.venv/bin/posementor" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$SCRIPT_DIR/posementor_cli.py" "$@"
fi

exec python "$SCRIPT_DIR/posementor_cli.py" "$@"
"""


def _launcher_cmd_text() -> str:
    return """@echo off
set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%.venv\\Scripts\\posementor.exe" (
  "%SCRIPT_DIR%.venv\\Scripts\\posementor.exe" %*
  exit /b %errorlevel%
)

if exist "%SCRIPT_DIR%.venv\\Scripts\\python.exe" (
  "%SCRIPT_DIR%.venv\\Scripts\\python.exe" "%SCRIPT_DIR%posementor_cli.py" %*
  exit /b %errorlevel%
)

python "%SCRIPT_DIR%posementor_cli.py" %*
"""


def _run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print(f"[CMD] {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, env=env, check=False)  # noqa: S603
    return result.returncode


def _platform_group() -> str:
    system_name = platform.system()
    if system_name == "Darwin":
        return "mac"
    if system_name == "Windows":
        return "windows"
    return "linux"


def _runtime_dirs(local_cfg: dict[str, Any]) -> tuple[Path, Path]:
    runtime_cfg = local_cfg.get("runtime", {})
    logs_dir = PROJECT_ROOT / str(runtime_cfg.get("logs_dir", "outputs/runtime/logs"))
    pids_dir = PROJECT_ROOT / str(runtime_cfg.get("pids_dir", "outputs/runtime/pids"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    pids_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, pids_dir


def _ensure_project_dirs(local_cfg: dict[str, Any]) -> None:
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    _runtime_dirs(local_cfg)


def _install_local_launchers() -> None:
    shell_launcher = PROJECT_ROOT / "posementor"
    shell_launcher.write_text(_launcher_shell_text(), encoding="utf-8")
    shell_launcher.chmod(shell_launcher.stat().st_mode | 0o111)

    cmd_launcher = PROJECT_ROOT / "posementor.cmd"
    cmd_launcher.write_text(_launcher_cmd_text(), encoding="utf-8")

    if platform.system() != "Windows":
        print(f"[OK] 已生成入口: {shell_launcher}")
        return

    exe_src = PROJECT_ROOT / ".venv" / "Scripts" / "posementor.exe"
    script_src = PROJECT_ROOT / ".venv" / "Scripts" / "posementor-script.py"
    exe_dst = PROJECT_ROOT / "posementor.exe"
    script_dst = PROJECT_ROOT / "posementor-script.py"

    if exe_src.exists():
        shutil.copy2(exe_src, exe_dst)
        if script_src.exists():
            shutil.copy2(script_src, script_dst)
        print(f"[OK] 已生成入口: {exe_dst}")
    else:
        print("[WARN] 未找到 .venv/Scripts/posementor.exe，已生成 posementor.cmd 兼容入口")


def _pid_file(pids_dir: Path, service_name: str) -> Path:
    return pids_dir / f"{service_name}.pid"


def _log_file(logs_dir: Path, service_name: str) -> Path:
    return logs_dir / f"{service_name}.log"


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    content = pid_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    try:
        return int(content)
    except ValueError:
        return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_process(pid: int) -> None:
    if platform.system() == "Windows":
        subprocess.run(  # noqa: S603
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _service_commands(local_cfg: dict[str, Any]) -> dict[str, list[str]]:
    network_cfg = local_cfg.get("network", {})
    backend_host = str(network_cfg.get("backend_host", "127.0.0.1"))
    backend_port = int(network_cfg.get("backend_port", 8787))
    frontend_host = str(network_cfg.get("frontend_host", "127.0.0.1"))
    frontend_port = int(network_cfg.get("frontend_port", 7860))

    return {
        BACKEND_SERVICE: [
            sys.executable,
            "backend_api.py",
            "--host",
            backend_host,
            "--port",
            str(backend_port),
        ],
        FRONTEND_SERVICE: [
            "pnpm",
            "--dir",
            str(PROJECT_ROOT / "frontend"),
            "dev",
            "--host",
            frontend_host,
            "--port",
            str(frontend_port),
        ],
    }


def _start_service(local_cfg: dict[str, Any], service_name: str) -> int:
    logs_dir, pids_dir = _runtime_dirs(local_cfg)
    pid_path = _pid_file(pids_dir, service_name)
    existing_pid = _read_pid(pid_path)
    if existing_pid is not None and _is_pid_running(existing_pid):
        print(f"[SKIP] {service_name} 已在运行 (pid={existing_pid})")
        return 0
    if existing_pid is not None and pid_path.exists():
        pid_path.unlink()

    command = _service_commands(local_cfg).get(service_name)
    if command is None:
        raise ValueError(f"未知服务: {service_name}")

    if shutil.which(command[0]) is None:
        print(f"[ERROR] 未找到命令: {command[0]}")
        return 2

    host, port = _service_host_port(local_cfg, service_name)
    if _port_is_occupied(host, port):
        if _service_health_ok(service_name, host, port):
            print(f"[SKIP] {service_name} 已在线 ({host}:{port})")
            return 0
        print(f"[ERROR] {service_name} 端口被占用但健康检查失败: {host}:{port}")
        return 4

    log_path = _log_file(logs_dir, service_name)
    with log_path.open("ab") as logf:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=PROJECT_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(str(process.pid), encoding="utf-8")
    time.sleep(0.6)
    if process.poll() is not None:
        print(f"[ERROR] {service_name} 启动失败，请查看日志: {log_path}")
        return 3

    print(f"[OK] {service_name} 已启动 (pid={process.pid})")
    return 0


def _stop_service(local_cfg: dict[str, Any], service_name: str) -> int:
    _, pids_dir = _runtime_dirs(local_cfg)
    pid_path = _pid_file(pids_dir, service_name)
    pid = _read_pid(pid_path)
    if pid is None:
        host, port = _service_host_port(local_cfg, service_name)
        if _port_is_occupied(host, port) and _service_health_ok(service_name, host, port):
            print(f"[SKIP] {service_name} 正在运行但不由 CLI 托管 ({host}:{port})")
            return 0
        print(f"[SKIP] {service_name} 未运行")
        return 0

    if _is_pid_running(pid):
        _stop_process(pid)
        time.sleep(0.5)
        if _is_pid_running(pid):
            print(f"[WARN] {service_name} 进程仍在运行 (pid={pid})")
            return 1
        print(f"[OK] 已停止 {service_name} (pid={pid})")
    else:
        print(f"[SKIP] {service_name} 进程不存在 (pid={pid})")

    if pid_path.exists():
        pid_path.unlink()
    return 0


def _tail_lines(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = text.splitlines()
    return "\n".join(rows[-lines:])


def _port_is_occupied(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _service_host_port(local_cfg: dict[str, Any], service_name: str) -> tuple[str, int]:
    network_cfg = local_cfg.get("network", {})
    if service_name == BACKEND_SERVICE:
        return str(network_cfg.get("backend_host", "127.0.0.1")), int(network_cfg.get("backend_port", 8787))
    if service_name == FRONTEND_SERVICE:
        return str(network_cfg.get("frontend_host", "127.0.0.1")), int(network_cfg.get("frontend_port", 7860))
    raise ValueError(f"未知服务: {service_name}")


def _service_health_ok(service_name: str, host: str, port: int) -> bool:
    path = "/health" if service_name == BACKEND_SERVICE else "/"
    try:
        connection = HTTPConnection(host=host, port=port, timeout=1.0)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(200).decode("utf-8", errors="ignore")
        connection.close()
    except OSError:
        return False
    if service_name == BACKEND_SERVICE:
        return response.status == 200 and "ok" in body.lower()
    return response.status in {200, 304}


def _list_local_processes() -> list[tuple[int, str]]:
    if platform.system() == "Windows":
        return []
    try:
        result = subprocess.run(  # noqa: S603
            ["ps", "-ax", "-o", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []

    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        rows.append((pid, parts[1]))
    return rows


def _legacy_pids(local_cfg: dict[str, Any]) -> list[tuple[int, str]]:
    rows = _list_local_processes()
    network_cfg = local_cfg.get("network", {})
    configured_frontend_port = int(network_cfg.get("frontend_port", 7860))
    project_text = str(PROJECT_ROOT)
    pids: list[tuple[int, str]] = []

    for pid, command in rows:
        if project_text not in command:
            continue

        if any(pattern in command for pattern in LEGACY_FRONTEND_PATTERNS):
            pids.append((pid, command))
            continue

        if "vite" in command and "frontend" in command:
            match = VITE_PORT_PATTERN.search(command)
            if match:
                try:
                    port = int(match.group(1))
                except ValueError:
                    port = configured_frontend_port
            else:
                port = configured_frontend_port
            if port != configured_frontend_port:
                pids.append((pid, command))
    return pids


def _cleanup(local_cfg: dict[str, Any]) -> int:
    failed = False
    legacy = _legacy_pids(local_cfg)
    if not legacy:
        print("[OK] 未发现历史前端进程")
    else:
        print("=== 清理历史前端进程 ===")
        for pid, command in legacy:
            _stop_process(pid)
            time.sleep(0.2)
            if _is_pid_running(pid):
                failed = True
                print(f"[WARN] 结束失败 pid={pid} cmd={command}")
            else:
                print(f"[OK] 已结束 pid={pid} cmd={command}")

    _, pids_dir = _runtime_dirs(local_cfg)
    for service_name in [BACKEND_SERVICE, FRONTEND_SERVICE]:
        pid_path = _pid_file(pids_dir, service_name)
        pid = _read_pid(pid_path)
        if pid is None:
            continue
        if _is_pid_running(pid):
            continue
        pid_path.unlink(missing_ok=True)
        print(f"[OK] 已清理僵尸 PID 文件: {pid_path}")

    if failed:
        print("[WARN] 部分进程未结束，请手动检查 `lsof -i :7861` / `lsof -i :7863`")
        return 1
    return 0


def _doctor(local_cfg: dict[str, Any]) -> int:
    checks: list[tuple[str, bool, str]] = []

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        (
            "Python",
            python_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    for cmd in ["uv", "node", "pnpm"]:
        exists = shutil.which(cmd) is not None
        checks.append((f"命令 {cmd}", exists, "已安装" if exists else "缺失"))

    checks.append(("本地配置", LOCAL_CONFIG_FILE.exists(), str(LOCAL_CONFIG_FILE)))
    checks.append(("项目入口", (PROJECT_ROOT / "posementor").exists(), str(PROJECT_ROOT / "posementor")))

    network_cfg = local_cfg.get("network", {})
    backend_host = str(network_cfg.get("backend_host", "127.0.0.1"))
    backend_port = int(network_cfg.get("backend_port", 8787))
    frontend_host = str(network_cfg.get("frontend_host", "127.0.0.1"))
    frontend_port = int(network_cfg.get("frontend_port", 7860))
    checks.append(
        (
            "Backend 端口",
            True,
            f"{backend_host}:{backend_port} {'占用中' if _port_is_occupied(backend_host, backend_port) else '可用'}",
        )
    )
    checks.append(
        (
            "Frontend 端口",
            True,
            f"{frontend_host}:{frontend_port} {'占用中' if _port_is_occupied(frontend_host, frontend_port) else '可用'}",
        )
    )

    annotations = PROJECT_ROOT / "data" / "raw" / "aistpp" / "annotations"
    checks.append(("AIST 注释目录", annotations.exists(), str(annotations)))
    yolo_weights = PROJECT_ROOT / "yolo11m-pose.pt"
    checks.append(("YOLO 权重", yolo_weights.exists(), str(yolo_weights)))

    print("=== PoseMentor Doctor ===")
    all_ok = True
    failed_items: list[str] = []
    for name, ok, detail in checks:
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] {name:<14} {detail}")
        all_ok = all_ok and ok
        if not ok:
            failed_items.append(name)

    if all_ok:
        print("[DONE] 环境检查通过")
        return 0

    print("\n修复建议：")
    if "命令 uv" in failed_items:
        print("- 先安装 uv：https://docs.astral.sh/uv/")
    if "命令 node" in failed_items or "命令 pnpm" in failed_items:
        print("- 先安装 Node.js 20+ 和 pnpm，然后重跑 `uv run posementor init`")
    if "本地配置" in failed_items:
        print("- 先执行 `uv run posementor config` 生成本地配置")
    if "项目入口" in failed_items:
        print("- 先执行 `uv run posementor install-launchers` 生成 ./posementor 入口")
    if "AIST 注释目录" in failed_items:
        print("- 执行 `uv run posementor quickstart --skip-train` 先准备数据")
    if "YOLO 权重" in failed_items:
        print("- 将 yolo11m-pose.pt 放在项目根目录，或先用官方2D流程训练")

    print("[WARN] 存在未通过项，请按上方提示处理")
    return 1


def _run_quickstart(local_cfg: dict[str, Any], args: argparse.Namespace) -> int:
    defaults = local_cfg.get("defaults", {})
    data_cfg = args.data_config or str(defaults.get("data_config", "configs/data.yaml"))
    train_cfg = args.train_config or str(defaults.get("train_config", "configs/train.yaml"))

    if not args.skip_data:
        code = _run_python_script(
            "download_and_prepare_aist.py",
            ["--config", data_cfg, "--download", "--extract"],
        )
        if code != 0:
            return code

    if not args.skip_extract:
        code = _run_python_script("extract_pose_aist2d.py", ["--config", data_cfg])
        if code != 0:
            return code

    if not args.skip_train:
        train_args = ["--config", train_cfg, "--epochs", str(args.epochs)]
        if args.export_onnx:
            train_args.append("--export-onnx")
        code = _run_python_script("train_3d_lift_demo.py", train_args)
        if code != 0:
            return code

    if args.up:
        for service_name in [BACKEND_SERVICE, FRONTEND_SERVICE]:
            code = _start_service(local_cfg, service_name)
            if code != 0:
                return code
        network_cfg = local_cfg.get("network", {})
        print(
            "[DONE] 服务地址："
            f" backend=http://{network_cfg.get('backend_host', '127.0.0.1')}:"
            f"{network_cfg.get('backend_port', 8787)} "
            f"frontend=http://{network_cfg.get('frontend_host', '127.0.0.1')}:"
            f"{network_cfg.get('frontend_port', 7860)}"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PoseMentor 一体化 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="生成或更新本地配置")
    p_config.add_argument("--force", action="store_true")
    p_config.add_argument("--profile", default="quick")
    p_config.add_argument("--backend-host", default="127.0.0.1")
    p_config.add_argument("--backend-port", type=int, default=8787)
    p_config.add_argument("--frontend-host", default="127.0.0.1")
    p_config.add_argument("--frontend-port", type=int, default=7860)
    p_config.add_argument("--dataset-id", default="aistpp")
    p_config.add_argument("--standard-id", default="private_action_core")
    p_config.add_argument("--config-path", default=str(LOCAL_CONFIG_FILE))

    p_config_init = sub.add_parser("config-init", help="初始化本地配置")
    p_config_init.add_argument("--force", action="store_true")
    p_config_init.add_argument("--profile", default="quick")
    p_config_init.add_argument("--backend-host", default="127.0.0.1")
    p_config_init.add_argument("--backend-port", type=int, default=8787)
    p_config_init.add_argument("--frontend-host", default="127.0.0.1")
    p_config_init.add_argument("--frontend-port", type=int, default=7860)
    p_config_init.add_argument("--dataset-id", default="aistpp")
    p_config_init.add_argument("--standard-id", default="private_action_core")
    p_config_init.add_argument("--config-path", default=str(LOCAL_CONFIG_FILE))

    p_config_show = sub.add_parser("config-show", help="显示当前本地配置")
    p_config_show.add_argument("--config-path", default=str(LOCAL_CONFIG_FILE))

    sub.add_parser("install-launchers", help="生成 ./posementor 与 Windows 入口")

    sub.add_parser("doctor", help="检查运行环境和关键依赖")

    p_init = sub.add_parser("init", help="安装依赖并初始化本地配置")
    p_init.add_argument("--config-path", default=str(LOCAL_CONFIG_FILE))
    p_init.add_argument("--force-config", action="store_true")

    sub.add_parser("up", help="启动前后端服务")
    sub.add_parser("down", help="停止前后端服务")
    sub.add_parser("start", help="启动前后端服务（up 别名）")
    sub.add_parser("stop", help="停止前后端服务（down 别名）")
    sub.add_parser("restart", help="重启前后端服务")
    sub.add_parser("status", help="查看服务状态")
    sub.add_parser("cleanup", help="清理历史前端进程和僵尸 PID 记录")

    p_logs = sub.add_parser("logs", help="查看服务日志")
    p_logs.add_argument("--service", choices=[BACKEND_SERVICE, FRONTEND_SERVICE, "all"], default="all")
    p_logs.add_argument("--lines", type=int, default=120)

    p_quickstart = sub.add_parser("quickstart", help="执行最小可用链路")
    p_quickstart.add_argument("--data-config", default="")
    p_quickstart.add_argument("--train-config", default="")
    p_quickstart.add_argument("--epochs", type=int, default=1)
    p_quickstart.add_argument("--skip-data", action="store_true")
    p_quickstart.add_argument("--skip-extract", action="store_true")
    p_quickstart.add_argument("--skip-train", action="store_true")
    p_quickstart.add_argument("--export-onnx", action="store_true")
    p_quickstart.add_argument("--up", action="store_true", help="训练结束后自动启动前后端")

    p_prepare = sub.add_parser("prepare-aist", help="下载并预处理 AIST++")
    p_prepare.add_argument("--config", default="configs/data.yaml")
    p_prepare.add_argument("--download", action="store_true")
    p_prepare.add_argument("--extract", action="store_true")
    p_prepare.add_argument("--download-videos", action="store_true")
    p_prepare.add_argument("--video-limit", type=int, default=0)
    p_prepare.add_argument("--agree-aist-license", action="store_true")
    p_prepare.add_argument("--skip-preprocess", action="store_true")
    p_prepare.add_argument("--limit", type=int, default=0)

    p_yolo = sub.add_parser("extract-yolo2d", help="使用 YOLO11 提取 2D 关键点")
    p_yolo.add_argument("--config", default="configs/data.yaml")
    p_yolo.add_argument("--video-root", default="")
    p_yolo.add_argument("--out-dir", default="")
    p_yolo.add_argument("--recursive", action="store_true")
    p_yolo.add_argument("--weights", default="yolo11m-pose.pt")
    p_yolo.add_argument("--conf", type=float, default=0.35)
    p_yolo.add_argument("--max-videos", type=int, default=0)

    p_aist2d = sub.add_parser("extract-aist2d", help="从 AIST++ 官方 2D 注释构建训练输入")
    p_aist2d.add_argument("--config", default="configs/data.yaml")
    p_aist2d.add_argument("--max-files", type=int, default=0)
    p_aist2d.add_argument("--overwrite", action="store_true")

    p_train = sub.add_parser("train-lift", help="训练 3D Lift 模型")
    p_train.add_argument("--config", default="configs/train.yaml")
    p_train.add_argument("--epochs", type=int, default=0)
    p_train.add_argument("--max-train-pairs", type=int, default=0)
    p_train.add_argument("--max-val-pairs", type=int, default=0)
    p_train.add_argument("--sample-stride", type=int, default=0)
    p_train.add_argument("--seq-len", type=int, default=0)
    p_train.add_argument("--num-workers", type=int, default=-1)
    p_train.add_argument("--export-onnx", action="store_true")

    p_multi = sub.add_parser("prepare-multiview", help="处理四机位数据")
    p_multi.add_argument("--config", default="configs/multiview.yaml")
    p_multi.add_argument("--limit-sessions", type=int, default=0)

    p_report = sub.add_parser("report-multiview", help="生成四机位处理报告")
    p_report.add_argument("--manifest", default="data/processed/multiview/multiview_manifest.csv")
    p_report.add_argument(
        "--output",
        default="outputs/visualization/multiview/multiview_report.html",
    )

    p_backend = sub.add_parser("serve-backend", help="启动 Backend API")
    p_backend.add_argument("--host", default="0.0.0.0")
    p_backend.add_argument("--port", type=int, default=8787)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command
    local_cfg = load_local_config(Path(getattr(args, "config_path", LOCAL_CONFIG_FILE)))

    if cmd in {"config", "config-init"}:
        overrides = {
            "profile": args.profile,
            "network.backend_host": args.backend_host,
            "network.backend_port": args.backend_port,
            "network.frontend_host": args.frontend_host,
            "network.frontend_port": args.frontend_port,
            "defaults.dataset_id": args.dataset_id,
            "defaults.standard_id": args.standard_id,
        }
        config_path, created = init_local_config(
            config_path=Path(args.config_path),
            force=args.force,
            overrides=overrides,
        )
        state = "已创建" if created else "已存在"
        print(f"[DONE] 本地配置{state}: {config_path}")
        raise SystemExit(0)

    if cmd == "config-show":
        cfg = load_local_config(Path(args.config_path))
        print(cfg)
        raise SystemExit(0)

    if cmd == "install-launchers":
        _install_local_launchers()
        raise SystemExit(0)

    if cmd == "doctor":
        raise SystemExit(_doctor(local_cfg))

    if cmd == "init":
        init_local_config(Path(args.config_path), force=args.force_config)
        local_cfg = load_local_config(Path(args.config_path))
        _ensure_project_dirs(local_cfg)
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(PROJECT_ROOT / ".uv_cache")

        code = _run_command(
            ["uv", "sync", "--group", "dev", "--group", _platform_group()],
            cwd=PROJECT_ROOT,
            env=env,
        )
        if code != 0:
            raise SystemExit(code)

        code = _run_command(
            ["pnpm", "--dir", str(PROJECT_ROOT / "frontend"), "install"],
            cwd=PROJECT_ROOT,
            env=env,
        )
        if code == 0:
            _install_local_launchers()
        raise SystemExit(code)

    if cmd in {"up", "start"}:
        for service_name in [BACKEND_SERVICE, FRONTEND_SERVICE]:
            code = _start_service(local_cfg, service_name)
            if code != 0:
                raise SystemExit(code)
        network_cfg = local_cfg.get("network", {})
        print(
            "[DONE] 服务地址："
            f" backend=http://{network_cfg.get('backend_host', '127.0.0.1')}:"
            f"{network_cfg.get('backend_port', 8787)} "
            f"frontend=http://{network_cfg.get('frontend_host', '127.0.0.1')}:"
            f"{network_cfg.get('frontend_port', 7860)}"
        )
        raise SystemExit(0)

    if cmd in {"down", "stop"}:
        codes = [_stop_service(local_cfg, BACKEND_SERVICE), _stop_service(local_cfg, FRONTEND_SERVICE)]
        raise SystemExit(max(codes))

    if cmd == "restart":
        stop_codes = [_stop_service(local_cfg, BACKEND_SERVICE), _stop_service(local_cfg, FRONTEND_SERVICE)]
        if max(stop_codes) not in {0, 1}:
            raise SystemExit(max(stop_codes))
        for service_name in [BACKEND_SERVICE, FRONTEND_SERVICE]:
            code = _start_service(local_cfg, service_name)
            if code != 0:
                raise SystemExit(code)
        network_cfg = local_cfg.get("network", {})
        print(
            "[DONE] 服务地址："
            f" backend=http://{network_cfg.get('backend_host', '127.0.0.1')}:"
            f"{network_cfg.get('backend_port', 8787)} "
            f"frontend=http://{network_cfg.get('frontend_host', '127.0.0.1')}:"
            f"{network_cfg.get('frontend_port', 7860)}"
        )
        raise SystemExit(0)

    if cmd == "status":
        logs_dir, pids_dir = _runtime_dirs(local_cfg)
        for service_name in [BACKEND_SERVICE, FRONTEND_SERVICE]:
            pid_path = _pid_file(pids_dir, service_name)
            pid = _read_pid(pid_path)
            status = "stopped"
            if pid is not None and _is_pid_running(pid):
                status = "running"
            elif pid is not None and pid_path.exists():
                pid_path.unlink()
                pid = None
            host, port = _service_host_port(local_cfg, service_name)
            if status == "stopped" and _port_is_occupied(host, port) and _service_health_ok(service_name, host, port):
                status = "running(external)"
            print(
                f"{service_name}: {status}"
                + (f" (pid={pid})" if pid is not None else "")
                + f" addr={host}:{port}"
                + f" log={_log_file(logs_dir, service_name)}"
            )
        raise SystemExit(0)

    if cmd == "cleanup":
        raise SystemExit(_cleanup(local_cfg))

    if cmd == "logs":
        logs_dir, _ = _runtime_dirs(local_cfg)
        targets = [BACKEND_SERVICE, FRONTEND_SERVICE] if args.service == "all" else [args.service]
        for idx, service_name in enumerate(targets):
            if idx > 0:
                print()
            path = _log_file(logs_dir, service_name)
            print(f"===== {service_name} ({path}) =====")
            text = _tail_lines(path, max(1, args.lines))
            print(text if text else "(暂无日志)")
        raise SystemExit(0)

    if cmd == "quickstart":
        raise SystemExit(_run_quickstart(local_cfg, args))

    if cmd == "prepare-aist":
        extra = ["--config", args.config]
        if args.download:
            extra.append("--download")
        if args.extract:
            extra.append("--extract")
        if args.download_videos:
            extra.append("--download-videos")
            if args.video_limit > 0:
                extra.extend(["--video-limit", str(args.video_limit)])
            if args.agree_aist_license:
                extra.append("--agree-aist-license")
        if args.skip_preprocess:
            extra.append("--skip-preprocess")
        if args.limit > 0:
            extra.extend(["--limit", str(args.limit)])
        code = _run_python_script("download_and_prepare_aist.py", extra)
        raise SystemExit(code)

    if cmd == "extract-yolo2d":
        extra = [
            "--config",
            args.config,
            "--weights",
            args.weights,
            "--conf",
            str(args.conf),
        ]
        if args.video_root:
            extra.extend(["--video-root", args.video_root])
        if args.out_dir:
            extra.extend(["--out-dir", args.out_dir])
        if args.recursive:
            extra.append("--recursive")
        if args.max_videos > 0:
            extra.extend(["--max-videos", str(args.max_videos)])
        code = _run_python_script("extract_pose_yolo11.py", extra)
        raise SystemExit(code)

    if cmd == "extract-aist2d":
        extra = ["--config", args.config]
        if args.max_files > 0:
            extra.extend(["--max-files", str(args.max_files)])
        if args.overwrite:
            extra.append("--overwrite")
        code = _run_python_script("extract_pose_aist2d.py", extra)
        raise SystemExit(code)

    if cmd == "train-lift":
        extra = ["--config", args.config]
        if args.epochs > 0:
            extra.extend(["--epochs", str(args.epochs)])
        if args.max_train_pairs > 0:
            extra.extend(["--max-train-pairs", str(args.max_train_pairs)])
        if args.max_val_pairs > 0:
            extra.extend(["--max-val-pairs", str(args.max_val_pairs)])
        if args.sample_stride > 0:
            extra.extend(["--sample-stride", str(args.sample_stride)])
        if args.seq_len > 0:
            extra.extend(["--seq-len", str(args.seq_len)])
        if args.num_workers >= 0:
            extra.extend(["--num-workers", str(args.num_workers)])
        if args.export_onnx:
            extra.append("--export-onnx")
        code = _run_python_script("train_3d_lift_demo.py", extra)
        raise SystemExit(code)

    if cmd == "prepare-multiview":
        extra = ["--config", args.config]
        if args.limit_sessions > 0:
            extra.extend(["--limit-sessions", str(args.limit_sessions)])
        code = _run_python_script("prepare_multiview_dataset.py", extra)
        raise SystemExit(code)

    if cmd == "report-multiview":
        extra = ["--manifest", args.manifest, "--output", args.output]
        code = _run_python_script("visualize_multiview_report.py", extra)
        raise SystemExit(code)

    if cmd == "serve-backend":
        script = Path("backend_api.py")
        if not script.exists():
            raise FileNotFoundError(f"找不到脚本: {script}")
        cmdline = [sys.executable, "backend_api.py", "--host", args.host, "--port", str(args.port)]
        print(f"[CMD] {' '.join(cmdline)}")
        code = subprocess.run(cmdline, check=False).returncode  # noqa: S603
        raise SystemExit(code)

    raise SystemExit(2)
