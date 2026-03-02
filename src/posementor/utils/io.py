from __future__ import annotations

import csv
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_pickle_or_npz(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        return {"data": obj}

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    raise ValueError(f"不支持的文件类型: {path}")


def try_get_array(data: dict[str, Any], candidates: list[str]) -> np.ndarray | None:
    for key in candidates:
        value = data.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        return arr
    return None


def download_with_progress(url: str, target: Path) -> None:
    ensure_dir(target.parent)

    def _hook(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = blocks * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(f"\r下载中 {target.name}: {pct:5.1f}%", end="")

    urllib.request.urlretrieve(url, target, _hook)
    print()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
