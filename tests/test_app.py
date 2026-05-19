import pytest
from fastapi.testclient import TestClient

from whisper_proxy.app import create_app


def test_healthz() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"] == "application/json"


def test_healthz_with_custom_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENARC_MODEL", "custom-model")
    monkeypatch.setenv("LOG_FORMAT", "text")
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
