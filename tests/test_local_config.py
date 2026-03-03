from __future__ import annotations

from pathlib import Path

from posementor.local_config import init_local_config, load_local_config


def test_init_local_config_creates_file(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "local.yaml"
    path, created = init_local_config(config_path=config_path, force=False, overrides=None)
    assert created is True
    assert path.exists()

    cfg = load_local_config(config_path)
    assert cfg["network"]["backend_port"] == 8787
    assert cfg["defaults"]["dataset_id"] == "aistpp"


def test_init_local_config_applies_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "local.yaml"
    _, created = init_local_config(
        config_path=config_path,
        force=False,
        overrides={
            "network.backend_port": 9000,
            "network.frontend_port": 7900,
            "defaults.dataset_id": "custom_singleview",
        },
    )
    assert created is True

    cfg = load_local_config(config_path)
    assert cfg["network"]["backend_port"] == 9000
    assert cfg["network"]["frontend_port"] == 7900
    assert cfg["defaults"]["dataset_id"] == "custom_singleview"


def test_init_local_config_without_force_keeps_existing(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "local.yaml"
    init_local_config(config_path=config_path, force=False, overrides={"network.backend_port": 9001})
    _, created = init_local_config(config_path=config_path, force=False, overrides={"network.backend_port": 9002})
    assert created is False

    cfg = load_local_config(config_path)
    assert cfg["network"]["backend_port"] == 9001
