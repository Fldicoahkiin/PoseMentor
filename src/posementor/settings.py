from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    """统一管理项目路径，避免硬编码路径导致跨平台问题。"""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths() -> ProjectPaths:
    return ProjectPaths(root=get_project_root())
