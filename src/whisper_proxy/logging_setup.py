import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from pythonjsonlogger import json as pythonjson

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
stage_timings_var: ContextVar[dict[str, float] | None] = ContextVar("stage_timings", default=None)


class _RFC3339JsonFormatter(pythonjson.JsonFormatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=UTC).isoformat()


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        rid = getattr(record, "request_id", None)
        if rid is not None:
            return f"{msg} request_id={rid}"
        return msg


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get()
        if rid is not None:
            record.request_id = rid
        return True


_installed_handler: logging.Handler | None = None


def configure_logging(level: str, fmt: str) -> None:
    global _installed_handler
    root = logging.getLogger()
    if _installed_handler is not None and _installed_handler in root.handlers:
        root.removeHandler(_installed_handler)

    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())

    formatter: logging.Formatter
    if fmt == "json":
        formatter = _RFC3339JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
    else:
        formatter = _TextFormatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)

    root.setLevel(level.upper())
    root.addHandler(handler)
    _installed_handler = handler


@asynccontextmanager
async def record_stage(name: str) -> AsyncGenerator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings = stage_timings_var.get(None)
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + elapsed_ms
