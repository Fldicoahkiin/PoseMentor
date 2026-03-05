#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/scripts/posementor.py" "$@"
fi

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/scripts/posementor.py" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$SCRIPT_DIR/scripts/posementor.py" "$@"
fi

exec python "$SCRIPT_DIR/scripts/posementor.py" "$@"
