import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from whisper_proxy.config import Settings
from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.logging_setup import configure_logging
from whisper_proxy.middleware import CorrelationMiddleware
from whisper_proxy.openarc import OpenArcClient, OpenArcUnavailable
from whisper_proxy.routes import asr, detect_language


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error — aborting startup:\n{exc}", file=sys.stderr)
        sys.exit(1)
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    app.state.settings = settings
    async with OpenArcClient(settings) as client:
        app.state.openarc_client = client
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="whisper-proxy", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CorrelationMiddleware)
    app.include_router(asr.router)
    app.include_router(detect_language.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
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

    return app


app = create_app()
