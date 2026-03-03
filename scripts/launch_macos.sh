#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/outputs/runtime"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"

ACTION="${1:-all}"

ensure_dirs() {
  mkdir -p "$PID_DIR" "$LOG_DIR"
}

install_deps() {
  cd "$PROJECT_ROOT"
  export UV_CACHE_DIR="$PROJECT_ROOT/.uv_cache"
  uv sync --group dev --group mac
  pnpm --dir "$PROJECT_ROOT/frontend" install
}

start_services() {
  ensure_dirs
  stop_services
  cd "$PROJECT_ROOT"
  export UV_CACHE_DIR="$PROJECT_ROOT/.uv_cache"

  nohup uv run python "$PROJECT_ROOT/backend_api.py" >"$LOG_DIR/backend_api.log" 2>&1 &
  echo $! >"$PID_DIR/backend_api.pid"

  nohup pnpm --dir "$PROJECT_ROOT/frontend" dev --host 127.0.0.1 --port 7860 >"$LOG_DIR/frontend.log" 2>&1 &
  echo $! >"$PID_DIR/frontend.pid"

  echo "服务已启动"
  echo "- Backend API:  http://127.0.0.1:8787"
  echo "- 前端系统:      http://127.0.0.1:7860"
}

stop_services() {
  ensure_dirs
  for service in backend_api frontend; do
    pid_file="$PID_DIR/$service.pid"
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" >/dev/null 2>&1; then
        kill "$pid"
        echo "已停止 $service ($pid)"
      fi
    fi
  done
}

case "$ACTION" in
install)
  install_deps
  ;;
start)
  start_services
  ;;
stop)
  stop_services
  ;;
all)
  install_deps
  start_services
  ;;
*)
  echo "用法: ./scripts/launch_macos.sh [install|start|stop|all]"
  exit 1
  ;;
esac

