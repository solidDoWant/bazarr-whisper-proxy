from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.openarc import OpenArcUnavailable

router = APIRouter()


@router.get("/status")
async def status(request: Request) -> Response:
    settings = get_settings(request)
    client = get_openarc_client(request)
    try:
        model_state = await client.model_state()
    except OpenArcUnavailable:
        model_state = "unknown"
    if model_state not in ("loaded", "unknown"):
        return JSONResponse(
            {"status": "ok", "model": settings.OPENARC_MODEL, "model_state": model_state},
            status_code=503,
            headers={"Retry-After": "30"},
        )
    return JSONResponse(
        {"status": "ok", "model": settings.OPENARC_MODEL, "model_state": model_state}
    )
