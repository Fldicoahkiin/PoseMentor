from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import backend_api

client = TestClient(backend_api.app)


def test_root_route_returns_service_info() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "posementor-backend"
    assert payload["status"] == "ok"


def test_compat_health_route() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_root_route() -> None:
    response = client.get("/api")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "posementor-backend"
    assert payload["health"] == "/api/health"


def test_datasets_route_returns_registry() -> None:
    response = client.get("/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("datasets"), list)
    assert any(item.get("id") == "aistpp" for item in payload["datasets"])
    assert all("video_root" in item for item in payload["datasets"])
    assert all("video_root_exists" in item for item in payload["datasets"])


def test_standards_route_returns_registry() -> None:
    response = client.get("/standards")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("standards"), list)
    assert any(item.get("source") == "private" for item in payload["standards"])


def test_invalid_dataset_is_rejected() -> None:
    response = client.post("/jobs/train", json={"dataset_id": "unknown_dataset"})
    assert response.status_code == 400
    assert "未知 dataset_id" in response.json()["detail"]


def test_artifact_status_route() -> None:
    response = client.get("/artifacts/status")
    assert response.status_code == 200
    payload = response.json()
    assert "curves_exists" in payload
    assert "sample_2d_url" in payload
    assert "sample_video_url" in payload
    assert "sample_2d_video_url" in payload
    assert "sample_3d_video_url" in payload


def test_artifact_manifest_route() -> None:
    response = client.get("/artifacts/manifest")
    assert response.status_code == 200
    payload = response.json()
    assert "count" in payload
    assert "files" in payload
    assert isinstance(payload["files"], list)


def test_workspace_source_preview_route() -> None:
    response = client.get("/workspace/source-preview", params={"dataset_id": "aistpp", "limit": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "aistpp"
    assert "samples" in payload


def test_job_progress_route_not_found() -> None:
    response = client.get("/jobs/not-exists/progress")
    assert response.status_code == 404


def test_upsert_dataset_updates_registry(monkeypatch, tmp_path: Path) -> None:
    registry = tmp_path / "datasets.yaml"
    registry.write_text(
        "datasets:\n"
        "  - id: aistpp\n"
        "    name: AIST++\n"
        "    stage: production\n"
        "    mode: singleview\n"
        "    data_config: configs/data.yaml\n"
        "    train_config: configs/train.yaml\n"
        "    video_root: data/raw/aistpp/videos\n"
        "    notes: default\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(backend_api, "DATASET_REGISTRY_FILE", registry)

    payload = {
        "id": "custom_mv_team",
        "name": "Custom MV Team",
        "stage": "planned",
        "mode": "multiview",
        "data_config": "configs/multiview.yaml",
        "train_config": "",
        "video_root": "data/raw/multiview",
        "notes": "team dataset",
    }
    response = client.post("/datasets/upsert", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["dataset"]["id"] == "custom_mv_team"

    response = client.get("/datasets")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert any(item["id"] == "custom_mv_team" for item in datasets)
