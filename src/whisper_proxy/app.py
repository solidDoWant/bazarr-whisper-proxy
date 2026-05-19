import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from whisper_proxy.config import Settings
from whisper_proxy.logging_setup import configure_logging
from whisper_proxy.middleware import CorrelationMiddleware
from whisper_proxy.openarc import OpenArcClient
from whisper_proxy.routes import asr, detect, healthz, status


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
    app.include_router(detect.router)
    app.include_router(healthz.router)
    app.include_router(status.router)

    return app


app = create_app()
