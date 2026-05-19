import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import pysrt
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from whisper_proxy.aligner import AlignmentFailed, align
from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, pcm_to_float32, pcm_to_wav
from whisper_proxy.config import Settings
from whisper_proxy.logging_setup import configure_logging, record_stage
from whisper_proxy.middleware import CorrelationMiddleware
from whisper_proxy.openarc import (
    OpenArcBadRequest,
    OpenArcClient,
    OpenArcInferenceError,
    OpenArcUnavailable,
)
from whisper_proxy.srt.segment import Cue, SegmentPolicy, words_to_cues
from whisper_proxy.srt.writer import cues_to_srt

_log = logging.getLogger(__name__)

_SOURCE_HEADER = "Transcribed using Bazarr to OpenAI Whisper Bridge!"


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


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_openarc_client(request: Request) -> OpenArcClient:
    return cast(OpenArcClient, request.app.state.openarc_client)


def create_app() -> FastAPI:
    app = FastAPI(title="whisper-proxy", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CorrelationMiddleware)

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

    @app.post("/asr")
    async def asr(
        request: Request,
        task: str = Query(...),
        language: str | None = Query(None),
        output: str = Query("srt"),
        encode: str = Query("false"),
        video_file: str | None = Query(None),
        audio_file: UploadFile = File(...),
    ) -> Response:
        settings = get_settings(request)
        client = get_openarc_client(request)

        if task == "translate":
            return JSONResponse(
                {"detail": "translate not implemented", "code": "translate_unsupported"},
                status_code=422,
            )

        async with record_stage("ingest_ms"):
            pcm = await audio_file.read()

        try:
            assert_within_size_limit(pcm, settings.MAX_AUDIO_BYTES)
        except AudioTooLarge:
            return JSONResponse({"detail": "audio too large"}, status_code=413)

        if video_file:
            _log.info("request summary video_file=%s", video_file)

        audio_wav = pcm_to_wav(pcm)
        audio_float32 = pcm_to_float32(pcm)

        try:
            async with record_stage("openarc_ms"):
                transcription = await client.transcribe(audio_wav, language)
        except (OpenArcUnavailable, OpenArcBadRequest, OpenArcInferenceError) as exc:
            _log.error("openarc error: %s", exc)
            return JSONResponse({"detail": str(exc)}, status_code=502)

        del audio_wav

        align_language = language or "en"

        try:
            async with record_stage("align_ms"):
                words = await align(
                    audio_float32,
                    transcription.text,
                    align_language,
                    _settings=settings,
                )
        except AlignmentFailed as exc:
            _log.warning("alignment failed, using fallback SRT: %s", exc)
            duration = float(
                transcription.metrics.get("audio_duration_sec") or len(pcm) / (16000 * 2)
            )
            fallback = _fallback_srt(transcription.text, duration)
            return PlainTextResponse(
                fallback,
                headers={"Source": _SOURCE_HEADER},
            )

        del audio_float32

        async with record_stage("format_ms"):
            policy = SegmentPolicy(
                max_chars=settings.CUE_MAX_CHARS,
                max_sec=settings.CUE_MAX_SEC,
                min_sec=settings.CUE_MIN_SEC,
                silence_ms=settings.CUE_SILENCE_MS,
            )
            cues = words_to_cues(words, policy)
            srt_text = cues_to_srt(cues)

        try:
            pysrt.from_string(srt_text, error_handling=pysrt.ERROR_RAISE)
        except Exception:
            _log.error("BUG: produced invalid SRT internally")
            return JSONResponse({"detail": "internal SRT error"}, status_code=500)

        return PlainTextResponse(
            srt_text,
            headers={"Source": _SOURCE_HEADER},
        )

    return app


def _fallback_srt(text: str, duration_sec: float) -> str:
    cue = Cue(start_sec=0.0, end_sec=max(duration_sec, 0.001), lines=(text,))
    return cues_to_srt([cue])


app = create_app()
