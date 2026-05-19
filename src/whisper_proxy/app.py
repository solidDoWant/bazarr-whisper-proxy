import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import ValidationError

from whisper_proxy.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    try:
        app.state.settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error — aborting startup:\n{exc}", file=sys.stderr)
        sys.exit(1)
    yield


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def create_app() -> FastAPI:
    app = FastAPI(title="whisper-proxy", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
