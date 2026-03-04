from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from posementor.utils.io import ensure_dir, load_yaml, save_yaml

DEFAULT_LOCAL_CONFIG: dict[str, Any] = {
    "profile": "quick",
    "network": {
        "backend_host": "127.0.0.1",
        "backend_port": 8787,
        "frontend_host": "127.0.0.1",
        "frontend_port": 7860,
    },
    "defaults": {
        "dataset_id": "aistpp",
        "standard_id": "private_action_core",
        "train_config": "configs/train.yaml",
        "data_config": "configs/data.yaml",
        "aist_video_profile": "mv3_quick",
    },
    "runtime": {
        "logs_dir": "outputs/runtime/logs",
        "pids_dir": "outputs/runtime/pids",
        "tail_lines": 120,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_by_dotted_key(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def load_local_config(config_path: Path) -> dict[str, Any]:
    payload = deepcopy(DEFAULT_LOCAL_CONFIG)
    if config_path.exists():
        raw = load_yaml(config_path)
        if isinstance(raw, dict):
            _deep_update(payload, raw)
    return payload


def init_local_config(
    config_path: Path,
    force: bool = False,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    if config_path.exists() and not force:
        return config_path, False

    payload = deepcopy(DEFAULT_LOCAL_CONFIG)
    if overrides:
        for key, value in overrides.items():
            _set_by_dotted_key(payload, key, value)

    ensure_dir(config_path.parent)
    save_yaml(config_path, payload)
    return config_path, True


def upsert_local_config(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    created = not config_path.exists()
    payload = load_local_config(config_path)
    if overrides:
        for key, value in overrides.items():
            _set_by_dotted_key(payload, key, value)
    ensure_dir(config_path.parent)
    save_yaml(config_path, payload)
    return config_path, created
