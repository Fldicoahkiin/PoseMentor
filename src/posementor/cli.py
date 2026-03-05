from __future__ import annotations

import argparse
import importlib.util
import json
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

from posementor.local_config import init_local_config, load_local_config, upsert_local_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_FILE = PROJECT_ROOT / "configs" / "local.yaml"
BACKEND_SERVICE = "backend_api"
FRONTEND_SERVICE = "frontend"
LEGACY_FRONTEND_PATTERNS = ("app_demo.py", "admin_console.py")
VITE_PORT_PATTERN = re.compile(r"--port\s+(\d+)")
VIDEO_CAMERA_PATTERN = re.compile(r"_c(\d+)_", re.IGNORECASE)
AIST_VIDEO_PROFILES: dict[str, dict[str, Any]] = {
    "mv3_quick": {
        "name": "三机位快速包",
        "camera_ids": ["c01", "c02", "c03"],
        "group_limit": 40,
        "min_cameras_per_group": 3,
        "description": "3 机位 × 40 动作组，适合快速联调。",
    },
    "mv5_standard": {
        "name": "五机位标准包",
        "camera_ids": ["c01", "c02", "c03", "c04", "c05"],
        "group_limit": 80,
        "min_cameras_per_group": 5,
        "description": "5 机位 × 80 动作组，平衡效果与体量。",
    },
    "mv9_core": {
        "name": "九机位核心包",
        "camera_ids": ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09"],
        "group_limit": 120,
        "min_cameras_per_group": 9,
        "description": "9 机位 × 120 动作组，完整多机位训练基线。",
    },
    "mv9_full": {
        "name": "九机位全量包",
        "camera_ids": ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09"],
        "group_limit": 0,
        "min_cameras_per_group": 9,
        "description": "9 机位全量下载，耗时长，适合长期训练。",
    },
}


def _run_python_script(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, script, *extra_args]
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def _run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print(f"[CMD] {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, env=env, check=False)  # noqa: S603
    return result.returncode


def _video_group_key(stem: str) -> str:
    return VIDEO_CAMERA_PATTERN.sub("_cAll_", stem)


def _video_camera_id(stem: str) -> str:
    matched = VIDEO_CAMERA_PATTERN.search(stem)
    if not matched:
        return ""
    return f"c{int(matched.group(1)):02d}"


def _inspect_aist_videos(video_root: Path) -> dict[str, Any]:
    files = sorted(video_root.rglob("*.mp4")) if video_root.exists() else []
    camera_set: set[str] = set()
    group_set: set[str] = set()
    for path in files:
        stem = path.stem
        camera_id = _video_camera_id(stem)
        if camera_id:
            camera_set.add(camera_id)
        group_set.add(_video_group_key(stem))
    return {
        "video_root": str(video_root),
        "file_count": len(files),
        "group_count": len(group_set),
        "camera_ids": sorted(camera_set),
    }


def _resolve_aist_state_file(local_cfg: dict[str, Any]) -> Path:
    defaults = local_cfg.get("defaults", {})
    state_text = str(defaults.get("aist_download_state_file", "outputs/runtime/aist_download_state.json"))
    state_path = Path(state_text)
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    return state_path


def _read_aist_state(local_cfg: dict[str, Any]) -> dict[str, Any]:
    state_path = _resolve_aist_state_file(local_cfg)
    if not state_path.exists():
        return {"path": str(state_path), "exists": False}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(state_path), "exists": True, "parse_ok": False}
    if not isinstance(payload, dict):
        return {"path": str(state_path), "exists": True, "parse_ok": False}
    payload["path"] = str(state_path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _profile_target_count(profile_id: str) -> str:
    profile = AIST_VIDEO_PROFILES[profile_id]
    group_limit = int(profile["group_limit"])
    camera_count = len(profile["camera_ids"])
    if group_limit <= 0:
        return f"all x {camera_count} 机位"
    return str(group_limit * camera_count)


def _profile_satisfied(stats: dict[str, Any], profile_id: str) -> bool:
    profile = AIST_VIDEO_PROFILES[profile_id]
    cameras = set(stats.get("camera_ids", []))
    required = set(profile["camera_ids"])
    if not required.issubset(cameras):
        return False
    group_limit = int(profile["group_limit"])
    if group_limit <= 0:
        return stats.get("group_count", 0) > 0
    return int(stats.get("group_count", 0)) >= group_limit


def _print_profile_table(stats: dict[str, Any], current_profile_id: str) -> None:
    print("\n=== AIST 多机位下载配置 ===")
    print("编号  Profile         机位              目标视频量     当前状态")
    for idx, profile_id in enumerate(AIST_VIDEO_PROFILES.keys(), start=1):
        profile = AIST_VIDEO_PROFILES[profile_id]
        camera_text = ",".join(profile["camera_ids"])
        target = _profile_target_count(profile_id)
        ok_text = "已满足" if _profile_satisfied(stats, profile_id) else "待下载"
        mark = "*" if profile_id == current_profile_id else " "
        print(
            f"{mark}{idx:<2}  {profile_id:<13} {camera_text:<17} {target:<12} {ok_text}"
        )
        print(f"     {profile['description']}")


def _prompt_yes_no(title: str, *, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{title} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("[WARN] 请输入 y 或 n")


def _prompt_choice(title: str, options: list[str], *, default_index: int = 0) -> str:
    print(f"\n{title}")
    for idx, option in enumerate(options, start=1):
        mark = "*" if idx - 1 == default_index else " "
        print(f" {mark} {idx}. {option}")
    while True:
        raw = input(f"输入编号 [默认 {default_index + 1}]: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print("[WARN] 编号无效，请重新输入")


def _prompt_int(title: str, *, default: int) -> int:
    while True:
        raw = input(f"{title} [默认 {default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            return int(raw)
        print("[WARN] 请输入数字")


def _prompt_float(title: str, *, default: float) -> float:
    while True:
        raw = input(f"{title} [默认 {default:.1f}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("[WARN] 请输入数字")
            continue
        if value <= 0:
            print("[WARN] 数值必须大于 0")
            continue
        return value


def _prompt_text(title: str, *, default: str = "") -> str:
    suffix = f" [默认 {default}]" if default else ""
    raw = input(f"{title}{suffix}: ").strip()
    if raw:
        return raw
    return default


def _bootstrap_tools(auto_install: bool) -> tuple[bool, list[str]]:
    rows: list[str] = []
    ok = True

    has_uv = shutil.which("uv") is not None
    if not has_uv and auto_install:
        code = _run_command([sys.executable, "-m", "pip", "install", "--upgrade", "uv"], cwd=PROJECT_ROOT)
        has_uv = code == 0 and shutil.which("uv") is not None
    rows.append(f"uv: {'OK' if has_uv else 'FAIL'}")
    ok = ok and has_uv

    has_node = shutil.which("node") is not None
    rows.append(f"node: {'OK' if has_node else 'FAIL'}")
    ok = ok and has_node

    has_pnpm = shutil.which("pnpm") is not None
    if not has_pnpm and auto_install:
        if shutil.which("corepack") is not None:
            _run_command(["corepack", "enable"], cwd=PROJECT_ROOT)
            _run_command(["corepack", "prepare", "pnpm@latest", "--activate"], cwd=PROJECT_ROOT)
        elif shutil.which("npm") is not None:
            _run_command(["npm", "install", "-g", "pnpm"], cwd=PROJECT_ROOT)
        has_pnpm = shutil.which("pnpm") is not None
    rows.append(f"pnpm: {'OK' if has_pnpm else 'FAIL'}")
    ok = ok and has_pnpm
    return ok, rows


def _run_aist_download_profile(
    local_cfg: dict[str, Any],
    profile_id: str,
    *,
    data_config: str | None = None,
    ranges: str | None = None,
    assume_speed_mbps: float | None = None,
    retry: int | None = None,
    resume_failed: bool | None = None,
) -> int:
    profile = AIST_VIDEO_PROFILES[profile_id]
    defaults = local_cfg.get("defaults", {})
    data_cfg = data_config or str(defaults.get("data_config", "configs/data.yaml"))
    ranges_text = ranges if ranges is not None else str(defaults.get("aist_video_ranges", "")).strip()
    speed_value = (
        float(assume_speed_mbps)
        if assume_speed_mbps is not None
        else float(defaults.get("aist_assume_speed_mbps", 10.0))
    )
    retry_value = int(retry) if retry is not None else int(defaults.get("aist_download_retry", 2))
    state_file = str(defaults.get("aist_download_state_file", "outputs/runtime/aist_download_state.json"))
    use_resume_failed = (
        bool(resume_failed) if resume_failed is not None else bool(defaults.get("aist_resume_failed", False))
    )
    script_args = [
        "--config",
        data_cfg,
        "--download-videos",
        "--agree-aist-license",
        "--skip-preprocess",
        "--camera-ids",
        ",".join(profile["camera_ids"]),
        "--min-cameras-per-group",
        str(profile["min_cameras_per_group"]),
        "--assume-speed-mbps",
        f"{max(0.1, speed_value):.1f}",
        "--retry",
        str(max(0, retry_value)),
        "--state-file",
        state_file,
    ]
    if int(profile["group_limit"]) > 0:
        script_args.extend(["--group-limit", str(profile["group_limit"])])
    if ranges_text:
        script_args.extend(["--ranges", ranges_text])
    if use_resume_failed:
        script_args.append("--resume-failed")
    return _run_python_script("download_and_prepare_aist.py", script_args)


def _run_config_wizard(local_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    print("=== PoseMentor Config UI ===")
    tool_ok, tool_rows = _bootstrap_tools(auto_install=False)
    print("环境检查：")
    for row in tool_rows:
        print(f"- {row}")
    if not tool_ok and _prompt_yes_no("检测到缺失工具，是否尝试自动安装？", default=True):
        tool_ok, tool_rows = _bootstrap_tools(auto_install=True)
        print("安装后检查：")
        for row in tool_rows:
            print(f"- {row}")
    if not tool_ok:
        print("[WARN] 仍有工具缺失，建议先修复后继续。")

    current_profile = str(local_cfg.get("defaults", {}).get("aist_video_profile", "mv3_quick"))
    if current_profile not in AIST_VIDEO_PROFILES:
        current_profile = "mv3_quick"
    aist_video_root = PROJECT_ROOT / "data" / "raw" / "aistpp" / "videos"
    stats = _inspect_aist_videos(aist_video_root)
    print(
        "\n当前 AIST 本地状态:"
        f" videos={stats['file_count']} groups={stats['group_count']} cameras={','.join(stats['camera_ids']) or '-'}"
    )
    _print_profile_table(stats, current_profile)

    profile_options = list(AIST_VIDEO_PROFILES.keys())
    selected_profile = _prompt_choice(
        "选择 AIST 多机位下载 Profile",
        profile_options,
        default_index=profile_options.index(current_profile),
    )
    current_ranges = str(local_cfg.get("defaults", {}).get("aist_video_ranges", "")).strip()
    print("\n可选下载区间示例：1-300,1200-1500（留空表示按 Profile 全量范围）")
    selected_ranges = _prompt_text("输入下载区间", default=current_ranges)
    current_speed = float(local_cfg.get("defaults", {}).get("aist_assume_speed_mbps", 10.0))
    assume_speed = _prompt_float("估算网络速度 Mbps", default=current_speed)
    current_retry = int(local_cfg.get("defaults", {}).get("aist_download_retry", 2))
    download_retry = _prompt_int("下载失败重试次数", default=current_retry)
    current_state_file = str(
        local_cfg.get("defaults", {}).get(
            "aist_download_state_file",
            "outputs/runtime/aist_download_state.json",
        )
    )
    state_file = _prompt_text("下载状态文件路径", default=current_state_file)
    current_resume_failed = bool(local_cfg.get("defaults", {}).get("aist_resume_failed", False))
    resume_failed = _prompt_yes_no("下载时默认启用失败续传", default=current_resume_failed)
    backend_port = _prompt_int("后端端口", default=int(args.backend_port))
    frontend_port = _prompt_int("前端端口", default=int(args.frontend_port))

    overrides = {
        "profile": args.profile,
        "network.backend_host": args.backend_host,
        "network.backend_port": backend_port,
        "network.frontend_host": args.frontend_host,
        "network.frontend_port": frontend_port,
        "defaults.dataset_id": args.dataset_id,
        "defaults.standard_id": args.standard_id,
        "defaults.aist_video_profile": selected_profile,
        "defaults.aist_video_ranges": selected_ranges,
        "defaults.aist_assume_speed_mbps": assume_speed,
        "defaults.aist_download_retry": download_retry,
        "defaults.aist_download_state_file": state_file,
        "defaults.aist_resume_failed": resume_failed,
    }
    download_now = _prompt_yes_no("是否立即按所选 Profile 下载 AIST 多机位素材？", default=False)
    return overrides, download_now


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
        service_pid = _read_pid(pid_path)
        if service_pid is None:
            continue
        if _is_pid_running(service_pid):
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

    for cmd in ["uv", "node", "pnpm", "ffmpeg"]:
        exists = shutil.which(cmd) is not None
        checks.append((f"命令 {cmd}", exists, "已安装" if exists else "缺失"))

    modules = [
        "numpy",
        "torch",
        "lightning",
        "ultralytics",
        "cv2",
        "fastapi",
        "gradio",
    ]
    for module_name in modules:
        installed = _module_exists(module_name)
        checks.append((f"Python 包 {module_name}", installed, "可导入" if installed else "缺失"))

    checks.append(("本地配置", LOCAL_CONFIG_FILE.exists(), str(LOCAL_CONFIG_FILE)))
    checks.append(
        ("统一脚本入口", (PROJECT_ROOT / "scripts" / "posementor.py").exists(), str(PROJECT_ROOT / "scripts" / "posementor.py"))
    )

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
    video_stats = _inspect_aist_videos(PROJECT_ROOT / "data" / "raw" / "aistpp" / "videos")
    checks.append(
        (
            "AIST 视频库",
            video_stats["file_count"] > 0,
            f"videos={video_stats['file_count']} groups={video_stats['group_count']}",
        )
    )

    state_info = _read_aist_state(local_cfg)
    state_exists = bool(state_info.get("exists"))
    state_parse_ok = bool(state_info.get("parse_ok", state_exists))
    state_path = str(state_info.get("path", _resolve_aist_state_file(local_cfg)))
    state_detail = state_path if state_exists else f"{state_path} (未生成)"
    checks.append(("下载状态文件", (not state_exists) or state_parse_ok, state_detail))
    failed_count = int(state_info.get("failed_count", 0)) if state_parse_ok else 0
    checks.append(
        (
            "下载失败待续传",
            failed_count == 0,
            f"failed={failed_count}",
        )
    )

    disk_usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = disk_usage.free / (1024**3)
    checks.append(("磁盘可用空间", free_gb >= 30.0, f"{free_gb:.1f} GB"))

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
    if "命令 ffmpeg" in failed_items:
        print("- 安装 ffmpeg（视频解码与导出必须）")
    if any(item.startswith("Python 包 ") for item in failed_items):
        print("- 先执行 `uv run posementor init` 或 `uv sync --group dev --group mac/windows/linux`")
    if "本地配置" in failed_items:
        print("- 先执行 `uv run posementor config` 生成本地配置")
    if "统一脚本入口" in failed_items:
        print("- 检查 `scripts/posementor.py` 是否存在，推荐使用 `python scripts/posementor.py` 执行 CLI")
    if "AIST 注释目录" in failed_items:
        print("- 执行 `uv run posementor quickstart --skip-train` 先准备数据")
    if "AIST 视频库" in failed_items:
        print("- 执行 `./pm config --download-now` 或 `./pm quickstart --download-videos` 下载视频")
    if "YOLO 权重" in failed_items:
        print("- 将 yolo11m-pose.pt 放在项目根目录，或先用官方2D流程训练")
    if "下载状态文件" in failed_items:
        print("- 状态文件不存在或损坏，建议重新执行一次视频下载生成状态")
    if "下载失败待续传" in failed_items:
        print("- 执行 `./pm resume-download` 仅续传失败视频")
    if "磁盘可用空间" in failed_items:
        print("- 建议至少预留 30GB 空间后再继续下载和训练")

    print("[WARN] 存在未通过项，请按上方提示处理")
    return 1


def _run_quickstart(local_cfg: dict[str, Any], args: argparse.Namespace) -> int:
    defaults = local_cfg.get("defaults", {})
    data_cfg = args.data_config or str(defaults.get("data_config", "configs/data.yaml"))
    train_cfg = args.train_config or str(defaults.get("train_config", "configs/train.yaml"))
    selected_profile = args.video_profile or str(defaults.get("aist_video_profile", "mv3_quick"))

    if not args.skip_data:
        code = _run_python_script(
            "download_and_prepare_aist.py",
            ["--config", data_cfg, "--download", "--extract"],
        )
        if code != 0:
            return code

    if args.download_videos:
        if selected_profile not in AIST_VIDEO_PROFILES:
            print(f"[ERROR] 未知视频 Profile: {selected_profile}")
            return 2
        code = _run_aist_download_profile(
            local_cfg,
            selected_profile,
            data_config=data_cfg,
            resume_failed=args.resume_failed,
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
    p_config.add_argument("--aist-video-profile", choices=list(AIST_VIDEO_PROFILES.keys()), default="mv3_quick")
    p_config.add_argument("--aist-video-ranges", default="", help="AIST 视频下载区间，如 1-300,600-900")
    p_config.add_argument("--aist-assume-speed-mbps", type=float, default=10.0, help="下载估算速度（Mbps）")
    p_config.add_argument("--aist-download-retry", type=int, default=2, help="单文件下载失败后的重试次数")
    p_config.add_argument(
        "--aist-download-state-file",
        default="outputs/runtime/aist_download_state.json",
        help="AIST 下载状态文件路径",
    )
    p_config.add_argument(
        "--aist-resume-failed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="配置默认是否启用失败续传",
    )
    p_config.add_argument("--config-path", default=str(LOCAL_CONFIG_FILE))
    p_config.add_argument("--wizard", action="store_true", help="使用交互式 Config UI")
    p_config.add_argument("--plain", action="store_true", help="使用非交互模式写入配置")
    p_config.add_argument("--download-now", action="store_true", help="配置完成后立即执行 AIST 多机位下载")

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
    p_quickstart.add_argument("--download-videos", action="store_true", help="按本地配置下载 AIST 多机位视频")
    p_quickstart.add_argument("--video-profile", default="", help="覆盖本地配置中的 AIST 视频下载 Profile")
    p_quickstart.add_argument(
        "--resume-failed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="下载视频时是否仅续传上次失败项",
    )
    p_quickstart.add_argument("--skip-extract", action="store_true")
    p_quickstart.add_argument("--skip-train", action="store_true")
    p_quickstart.add_argument("--export-onnx", action="store_true")
    p_quickstart.add_argument("--up", action="store_true", help="训练结束后自动启动前后端")

    p_resume = sub.add_parser("resume-download", help="按本地状态文件续传 AIST 下载失败项")
    p_resume.add_argument("--data-config", default="", help="覆盖数据配置文件路径")
    p_resume.add_argument("--video-profile", default="", help="覆盖本地配置中的 AIST 视频下载 Profile")

    p_quality = sub.add_parser("quality", help="运行全局代码质量与训练链路检查")
    p_quality.add_argument("--full", action="store_true", help="启用完整质量检查（含 mypy）")
    p_quality.add_argument("--strict", action="store_true", help="视觉/训练链路指标也作为失败条件")
    p_quality.add_argument("--skip-tests", action="store_true", help="跳过 pytest 回归测试")
    p_quality.add_argument("--skip-mypy", action="store_true", help="跳过 mypy 类型检查")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command
    local_cfg = load_local_config(Path(getattr(args, "config_path", LOCAL_CONFIG_FILE)))

    if cmd == "config":
        use_wizard = (args.wizard or not args.plain) and sys.stdin.isatty()
        download_now = bool(args.download_now)
        if use_wizard:
            overrides, wizard_download_now = _run_config_wizard(local_cfg, args)
            download_now = download_now or wizard_download_now
        else:
            profile_id = args.aist_video_profile.strip() or "mv3_quick"
            if profile_id not in AIST_VIDEO_PROFILES:
                print(f"[ERROR] 未知 aist_video_profile: {profile_id}")
                raise SystemExit(2)
            overrides = {
                "profile": args.profile,
                "network.backend_host": args.backend_host,
                "network.backend_port": args.backend_port,
                "network.frontend_host": args.frontend_host,
                "network.frontend_port": args.frontend_port,
                "defaults.dataset_id": args.dataset_id,
                "defaults.standard_id": args.standard_id,
                "defaults.aist_video_profile": profile_id,
                "defaults.aist_video_ranges": args.aist_video_ranges.strip(),
                "defaults.aist_assume_speed_mbps": max(0.1, float(args.aist_assume_speed_mbps)),
                "defaults.aist_download_retry": max(0, int(args.aist_download_retry)),
                "defaults.aist_download_state_file": args.aist_download_state_file.strip(),
                "defaults.aist_resume_failed": bool(args.aist_resume_failed),
            }
        if args.force:
            config_path, created = init_local_config(
                config_path=Path(args.config_path),
                force=True,
                overrides=overrides,
            )
        else:
            config_path, created = upsert_local_config(
                config_path=Path(args.config_path),
                overrides=overrides,
            )
        state = "已创建" if created else "已存在"
        print(f"[DONE] 本地配置{state}: {config_path}")
        local_cfg = load_local_config(Path(args.config_path))
        selected_profile = str(local_cfg.get("defaults", {}).get("aist_video_profile", "mv3_quick"))
        if download_now:
            if selected_profile not in AIST_VIDEO_PROFILES:
                print(f"[WARN] 未知 Profile: {selected_profile}，跳过下载")
            else:
                code = _run_aist_download_profile(local_cfg, selected_profile)
                if code != 0:
                    raise SystemExit(code)
        stats = _inspect_aist_videos(PROJECT_ROOT / "data" / "raw" / "aistpp" / "videos")
        print(
            "[INFO] 当前 AIST 本地状态:"
            f" videos={stats['file_count']} groups={stats['group_count']} cameras={','.join(stats['camera_ids']) or '-'}"
        )
        raise SystemExit(0)

    if cmd == "resume-download":
        defaults = local_cfg.get("defaults", {})
        data_cfg = args.data_config or str(defaults.get("data_config", "configs/data.yaml"))
        selected_profile = args.video_profile or str(defaults.get("aist_video_profile", "mv3_quick"))
        if selected_profile not in AIST_VIDEO_PROFILES:
            print(f"[ERROR] 未知视频 Profile: {selected_profile}")
            raise SystemExit(2)
        code = _run_aist_download_profile(
            local_cfg,
            selected_profile,
            data_config=data_cfg,
            resume_failed=True,
        )
        raise SystemExit(code)

    if cmd == "doctor":
        raise SystemExit(_doctor(local_cfg))

    if cmd == "quality":
        from posementor.quality import run_quality_suite

        raise SystemExit(
            run_quality_suite(
                project_root=PROJECT_ROOT,
                full=bool(args.full),
                strict=bool(args.strict),
                skip_tests=bool(args.skip_tests),
                skip_mypy=bool(args.skip_mypy),
            )
        )

    if cmd == "init":
        init_local_config(Path(args.config_path), force=args.force_config)
        local_cfg = load_local_config(Path(args.config_path))
        _ensure_project_dirs(local_cfg)
        tool_ok, tool_rows = _bootstrap_tools(auto_install=True)
        print("=== Toolchain 检查 ===")
        for row in tool_rows:
            print(f"- {row}")
        if not tool_ok:
            print("[ERROR] 关键工具缺失，无法继续 init")
            raise SystemExit(2)
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

    raise SystemExit(2)
