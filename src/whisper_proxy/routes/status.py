from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.openarc import OpenArcUnavailable

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("whisper-proxy")
except Exception:
    # OCI image installs an empty stub (no dist-info); fall back gracefully.
    _VERSION = "0.1.0"

router = APIRouter()

_RETRY_AFTER = {"Retry-After": "30"}


@router.get("/status")
async def status(request: Request) -> Response:
    settings = get_settings(request)
    client = get_openarc_client(request)
    try:
        model_state = await client.model_state()
    except OpenArcUnavailable:
        model_state = "unknown"
    if model_state == "loaded":
        return JSONResponse(
            {
                "status": "ok",
                "version": _VERSION,
                "model": settings.OPENARC_MODEL,
                "model_state": "loaded",
            }
        )
    return JSONResponse(
        {"detail": "model loading", "model_state": model_state},
        status_code=503,
        headers=_RETRY_AFTER,
    )
