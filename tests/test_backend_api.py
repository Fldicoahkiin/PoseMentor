from __future__ import annotations

from fastapi.testclient import TestClient

from backend_api import app


client = TestClient(app)


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
