"""Tests for specs/03-logging.md acceptance criteria."""

import json
import logging
import uuid
from io import StringIO

import pytest
from fastapi.testclient import TestClient

from whisper_proxy.app import create_app
from whisper_proxy.logging_setup import (
    _RequestIdFilter,
    _RFC3339JsonFormatter,
    _TextFormatter,
    record_stage,
    request_id_var,
    stage_timings_var,
)

# ---------------------------------------------------------------------------
# Formatter unit tests
# ---------------------------------------------------------------------------


def _json_handler() -> tuple[logging.StreamHandler, StringIO]:  # type: ignore[type-arg]
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(
        _RFC3339JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
    )
    h.addFilter(_RequestIdFilter())
    return h, buf


def _isolated_logger(name: str, handler: logging.Handler) -> logging.Logger:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    return log


def test_json_format_required_fields() -> None:
    """AC1: JSON output contains timestamp, level, logger, message."""
    h, buf = _json_handler()
    log = _isolated_logger("test.ac1.fields", h)
    log.info("hello world")

    data = json.loads(buf.getvalue().strip())
    assert {"timestamp", "level", "logger", "message"} <= data.keys()
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "test.ac1.fields"
    log.removeHandler(h)


def test_json_timestamp_is_rfc3339() -> None:
    """AC1: timestamp is RFC 3339 (ISO 8601 with timezone offset)."""
    import re

    h, buf = _json_handler()
    log = _isolated_logger("test.ac1.ts", h)
    log.info("ts check")

    data = json.loads(buf.getvalue().strip())
    ts = data["timestamp"]
    # RFC 3339 pattern: YYYY-MM-DDTHH:MM:SS[.ffffff]+HH:MM or Z
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), f"Not RFC3339: {ts!r}"
    assert "+" in ts or ts.endswith("Z"), f"No timezone offset: {ts!r}"
    log.removeHandler(h)


def test_json_in_request_includes_request_id() -> None:
    """AC1: log lines during a request include request_id."""
    h, buf = _json_handler()
    log = _isolated_logger("test.ac1.rid", h)

    tok = request_id_var.set("test-request-id")
    try:
        log.info("inside request")
    finally:
        request_id_var.reset(tok)

    data = json.loads(buf.getvalue().strip())
    assert data.get("request_id") == "test-request-id"
    log.removeHandler(h)


def test_json_outside_request_no_request_id() -> None:
    """AC1: log lines outside a request do NOT include request_id."""
    h, buf = _json_handler()
    log = _isolated_logger("test.ac1.norid", h)
    log.info("outside request")

    data = json.loads(buf.getvalue().strip())
    assert "request_id" not in data
    log.removeHandler(h)


def test_text_format_human_readable() -> None:
    """AC2: text format produces single-line, human-readable output."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(
        _TextFormatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    h.addFilter(_RequestIdFilter())
    log = _isolated_logger("test.ac2.text", h)
    log.info("human readable")

    line = buf.getvalue().strip()
    assert "\n" not in line
    assert "human readable" in line
    assert line.startswith("20")  # date prefix
    log.removeHandler(h)


def test_text_format_includes_request_id_in_request() -> None:
    """AC2: text format includes request_id when in-request."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_TextFormatter(fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s"))
    h.addFilter(_RequestIdFilter())
    log = _isolated_logger("test.ac2.rid", h)

    tok = request_id_var.set("abc-123")
    try:
        log.info("in request")
    finally:
        request_id_var.reset(tok)

    line = buf.getvalue().strip()
    assert "request_id=abc-123" in line
    log.removeHandler(h)


def test_text_format_no_request_id_outside_request() -> None:
    """AC2: text format omits request_id outside a request."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_TextFormatter(fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s"))
    h.addFilter(_RequestIdFilter())
    log = _isolated_logger("test.ac2.norid", h)
    log.info("outside")

    line = buf.getvalue().strip()
    assert "request_id" not in line
    log.removeHandler(h)


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=True)


def _ping_client() -> TestClient:
    """A client over a tiny app exposing GET /ping that returns 200.

    The real /healthz endpoint is suppressed from summary logging in the
    middleware (to reduce noise from kubelet probes), so tests that need to
    observe the summary log line must hit a different path.
    """
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from whisper_proxy.logging_setup import configure_logging
    from whisper_proxy.middleware import CorrelationMiddleware

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncGenerator[None]:
        configure_logging("INFO", "json")
        yield

    app = FastAPI(lifespan=_lifespan)
    app.add_middleware(CorrelationMiddleware)

    @app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"ok": "1"}

    return TestClient(app, raise_server_exceptions=True)


def test_response_has_x_request_id_header() -> None:
    """AC3: every HTTP response carries X-Request-Id."""
    with _client() as c:
        resp = c.get("/healthz")
    assert "x-request-id" in resp.headers
    rid = resp.headers["x-request-id"]
    parsed = uuid.UUID(rid)
    assert parsed.version == 4


def test_echoes_valid_uuid_header() -> None:
    """AC4: valid incoming X-Request-Id is echoed back."""
    incoming = str(uuid.uuid4())
    with _client() as c:
        resp = c.get("/healthz", headers={"x-request-id": incoming})
    assert resp.headers["x-request-id"] == incoming


def test_generates_uuid_for_missing_header() -> None:
    """AC4: missing X-Request-Id triggers generation of a new UUIDv4."""
    with _client() as c:
        resp = c.get("/healthz")
    rid = resp.headers["x-request-id"]
    assert uuid.UUID(rid).version == 4


def test_generates_uuid_for_malformed_header() -> None:
    """AC4: malformed X-Request-Id triggers generation of a new UUIDv4."""
    with _client() as c:
        resp = c.get("/healthz", headers={"x-request-id": "not-a-uuid"})
    rid = resp.headers["x-request-id"]
    assert rid != "not-a-uuid"
    assert uuid.UUID(rid).version == 4


def test_summary_log_emitted_after_request(caplog: pytest.LogCaptureFixture) -> None:
    """AC5: exactly one INFO summary log line after request with required fields."""
    with caplog.at_level(logging.INFO):
        with _ping_client() as c:
            c.get("/ping")

    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert len(summaries) == 1
    rec = summaries[0]
    assert rec.levelno == logging.INFO
    assert hasattr(rec, "method")
    assert hasattr(rec, "path")
    assert hasattr(rec, "status")
    assert hasattr(rec, "total_ms")
    assert hasattr(rec, "request_id")
    assert rec.method == "GET"  # type: ignore[attr-defined]
    assert rec.path == "/ping"  # type: ignore[attr-defined]
    assert rec.status == 200  # type: ignore[attr-defined]
    assert isinstance(rec.total_ms, float)  # type: ignore[attr-defined]


def test_summary_log_request_id_matches_response(caplog: pytest.LogCaptureFixture) -> None:
    """AC5: summary log request_id matches the X-Request-Id response header."""
    with caplog.at_level(logging.INFO):
        with _ping_client() as c:
            resp = c.get("/ping")

    response_rid = resp.headers["x-request-id"]
    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert summaries[0].request_id == response_rid  # type: ignore[attr-defined]


async def test_timing_helper_records_stage(caplog: pytest.LogCaptureFixture) -> None:
    """AC6: record_stage contributes a per-stage timing to the summary."""
    import anyio

    tim_tok = stage_timings_var.set({})
    try:
        async with record_stage("ingest"):
            await anyio.sleep(0)  # non-blocking yield — confirms no event-loop blocking
    finally:
        timings = stage_timings_var.get({}) or {}
        assert "ingest" in timings
        assert timings["ingest"] >= 0.0
        stage_timings_var.reset(tim_tok)


def test_timing_helper_absent_stages_not_in_summary(caplog: pytest.LogCaptureFixture) -> None:
    """AC6: stages not recorded do not appear in the summary log."""
    with caplog.at_level(logging.INFO):
        with _ping_client() as c:
            c.get("/ping")

    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    rec = summaries[0]
    # /ping doesn't call record_stage, so no stage keys like ingest_ms or openarc_ms
    assert not hasattr(rec, "ingest_ms")
    assert not hasattr(rec, "openarc_ms")


def test_timing_helper_stage_in_summary_when_used(caplog: pytest.LogCaptureFixture) -> None:
    """AC6: recorded stages appear in the summary log."""
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from whisper_proxy.logging_setup import configure_logging
    from whisper_proxy.middleware import CorrelationMiddleware

    @asynccontextmanager
    async def _lifespan(a: FastAPI) -> AsyncGenerator[None]:
        configure_logging("INFO", "json")
        yield

    timed_app = FastAPI(lifespan=_lifespan)
    timed_app.add_middleware(CorrelationMiddleware)

    @timed_app.get("/timed")
    async def timed_route() -> dict[str, str]:
        async with record_stage("myop"):
            pass
        return {"ok": "1"}

    with caplog.at_level(logging.INFO):
        with TestClient(timed_app) as c:
            c.get("/timed")

    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert len(summaries) == 1
    assert hasattr(summaries[0], "myop")
    assert isinstance(summaries[0].myop, float)  # type: ignore[attr-defined]


def test_warning_level_suppresses_info_summary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC7: LOG_LEVEL=WARNING suppresses INFO summary logs."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    with caplog.at_level(logging.DEBUG):
        with TestClient(create_app()) as c:
            c.get("/healthz")

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 0


def test_warning_level_still_emits_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC7: LOG_LEVEL=WARNING does not suppress ERROR-level records."""
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from whisper_proxy.logging_setup import configure_logging
    from whisper_proxy.middleware import CorrelationMiddleware

    @asynccontextmanager
    async def _lifespan(a: FastAPI) -> AsyncGenerator[None]:
        configure_logging("WARNING", "json")
        yield

    err_app = FastAPI(lifespan=_lifespan)
    err_app.add_middleware(CorrelationMiddleware)
    _err_log = logging.getLogger("test.errors")

    @err_app.get("/err")
    async def err_route() -> dict[str, str]:
        _err_log.error("something went wrong")
        return {"ok": "1"}

    with caplog.at_level(logging.DEBUG):
        with TestClient(err_app) as c:
            c.get("/err")

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) >= 1
