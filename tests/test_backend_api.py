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
