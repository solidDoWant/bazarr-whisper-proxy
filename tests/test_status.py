"""GET /status and GET /healthz route tests — spec task 11."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from whisper_proxy.app import create_app
from whisper_proxy.openarc import OpenArcClient, OpenArcUnavailable

OPENARC_BASE = "http://localhost:8000"
STATUS_URL = f"{OPENARC_BASE}/openarc/status"
MODEL = "qwen3-asr-0_6b-int8-asym"  # default OPENARC_MODEL


def _make_client() -> httpx.Client:
    return TestClient(create_app())


def _mock_model_state(state: str) -> AbstractContextManager[AsyncMock]:
    return patch.object(OpenArcClient, "model_state", new=AsyncMock(return_value=state))


# ---------------------------------------------------------------------------
# /healthz — criteria 1-3
# ---------------------------------------------------------------------------


def test_healthz_returns_200_ok() -> None:
    """AC1: 200 with {"status":"ok"} and application/json."""
    with respx.mock:
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert "application/json" in resp.headers["content-type"]


def test_healthz_no_network_request() -> None:
    """AC2/AC3: healthz must not call OpenArc — succeeds even when unreachable."""
    with respx.mock:
        # Do NOT register any routes — any network call would raise an error
        with _make_client() as client:
            resp = client.get("/healthz")

    assert resp.status_code == 200


def test_healthz_under_50ms() -> None:
    """AC2: p95 under 50 ms on warm process."""
    with respx.mock:
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            latencies: list[float] = []
            for _ in range(20):
                t0 = time.perf_counter()
                client.get("/healthz")
                latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < 50, f"p95 latency {p95:.1f} ms exceeds 50 ms"


# ---------------------------------------------------------------------------
# /status — criterion 4: loaded → 200
# ---------------------------------------------------------------------------


def test_status_loaded_returns_200() -> None:
    """AC4: loaded → 200 with correct body."""
    with _mock_model_state("loaded"):
        with _make_client() as client:
            resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": MODEL, "model_state": "loaded"}
    assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /status — criterion 5: loading/unloaded → 503 + Retry-After
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["loading", "unloaded"])
def test_status_not_loaded_returns_503(state: str) -> None:
    """AC5: loading or unloaded → 503 with detail body and Retry-After: 30."""
    with _mock_model_state(state):
        with _make_client() as client:
            resp = client.get("/status")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "model loading", "model_state": state}
    assert resp.headers.get("retry-after") == "30"
    assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /status — criterion 6: OpenArc unreachable → 503 + unknown
# ---------------------------------------------------------------------------


def test_status_openarc_unreachable_returns_503_unknown() -> None:
    """AC6: unreachable → 503 with model_state unknown and Retry-After: 30."""
    with patch.object(
        OpenArcClient,
        "model_state",
        new=AsyncMock(side_effect=OpenArcUnavailable("connection refused")),
    ):
        with _make_client() as client:
            resp = client.get("/status")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "model loading", "model_state": "unknown"}
    assert resp.headers.get("retry-after") == "30"


# ---------------------------------------------------------------------------
# /status — criterion 7: model not in array → 503 + unknown
# ---------------------------------------------------------------------------


def test_status_model_not_in_array_returns_503_unknown() -> None:
    """AC7: model absent from OpenArc response → 503 with model_state unknown."""
    with respx.mock:
        # Return a model list that does not contain OPENARC_MODEL
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "other-model", "status": "loaded"}]
            )
        )
        with _make_client() as client:
            resp = client.get("/status")

    assert resp.status_code == 503
    body = resp.json()
    assert body["model_state"] == "unknown"
    assert resp.headers.get("retry-after") == "30"
