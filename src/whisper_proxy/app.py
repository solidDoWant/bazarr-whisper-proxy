import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from pydantic import ValidationError

from whisper_proxy.config import Settings
from whisper_proxy.logging_setup import configure_logging
from whisper_proxy.middleware import CorrelationMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error — aborting startup:\n{exc}", file=sys.stderr)
        sys.exit(1)
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    app.state.settings = settings
    yield


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def create_app() -> FastAPI:
    app = FastAPI(title="whisper-proxy", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CorrelationMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
