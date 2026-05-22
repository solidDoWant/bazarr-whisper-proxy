import logging
import time
import uuid
from typing import cast

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from whisper_proxy.logging_setup import request_id_var, stage_timings_var

_BODY_LOG_LIMIT = 2000


def _fmt_timing(v: object) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(v, 2)
    return v


_log = logging.getLogger(__name__)


def _validated_uuid(value: str) -> str:
    try:
        uuid.UUID(value)
    except ValueError, AttributeError:
        return str(uuid.uuid4())
    return value


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        raw = request.headers.get("x-request-id", "")
        req_id = _validated_uuid(raw) if raw else str(uuid.uuid4())

        id_tok = request_id_var.set(req_id)
        tim_tok = stage_timings_var.set({})
        start = time.perf_counter()
        status = 500
        error_body: bytes | None = None

        try:
            response = await call_next(request)
            status = response.status_code
            if status >= 400:
                raw_chunks = [
                    chunk async for chunk in cast(StreamingResponse, response).body_iterator
                ]
                error_body = b"".join(
                    chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                    for chunk in raw_chunks
                )
                response = Response(
                    content=error_body,
                    status_code=status,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
        finally:
            if not (request.url.path == "/healthz" and status == 200):
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                timings = stage_timings_var.get(None) or {}
                extra: dict = {
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "total_ms": elapsed,
                    **{k: _fmt_timing(v) for k, v in timings.items()},
                }
                if error_body is not None:
                    extra["response_body"] = error_body[:_BODY_LOG_LIMIT].decode(
                        "utf-8", errors="replace"
                    )
                _log.info(
                    "request completed",
                    extra=extra,
                )
            request_id_var.reset(id_tok)
            stage_timings_var.reset(tim_tok)

        response.headers["x-request-id"] = req_id
        return response
