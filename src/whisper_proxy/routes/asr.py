import logging

import pysrt
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from whisper_proxy.aligner import AlignmentFailed, align
from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, pcm_to_float32, pcm_to_wav
from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.logging_setup import record_stage
from whisper_proxy.openarc import OpenArcBadRequest, OpenArcInferenceError, OpenArcUnavailable
from whisper_proxy.srt.segment import SegmentPolicy, words_to_cues
from whisper_proxy.srt.writer import cues_to_srt, fallback_srt

_log = logging.getLogger(__name__)

_SOURCE_HEADER = "Transcribed using Bazarr to OpenAI Whisper Bridge!"

router = APIRouter()


@router.post("/asr")
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
        duration = float(transcription.metrics.get("audio_duration_sec") or len(pcm) / (16000 * 2))
        return PlainTextResponse(
            fallback_srt(transcription.text, duration),
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
