import logging

import pysrt
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.datastructures import UploadFile

from whisper_proxy._types import TranscriptionSegment
from whisper_proxy.aligner import AlignmentFailed, align
from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, pcm_to_float32, pcm_to_wav
from whisper_proxy.deps import get_lingarr_client, get_openarc_client, get_settings
from whisper_proxy.lingarr import (
    LingarrClient,
    LingarrCountMismatch,
    LingarrError,
    LingarrInvalidResponse,
    LingarrPositionMismatch,
    arr_media_id_for,
    title_for,
)
from whisper_proxy.logging_setup import log_extra, record_stage
from whisper_proxy.openarc import OpenArcBadRequest, OpenArcInferenceError, OpenArcUnavailable
from whisper_proxy.srt.segment import Cue, SegmentPolicy, words_to_cues
from whisper_proxy.srt.writer import cues_to_srt, fallback_srt

_log = logging.getLogger(__name__)

_SOURCE_HEADER = "Transcribed using Bazarr to OpenAI Whisper Bridge!"
_NBSP = "\u00a0"

router = APIRouter()


async def _do_translate(
    lingarr: LingarrClient,
    cues: list[Cue],
    source_language: str,
    target_language: str,
    media_type: str,
    video_file: str | None,
) -> Response:
    lines = [(i + 1, " ".join(cue.lines)) for i, cue in enumerate(cues)]

    try:
        async with record_stage("translate_ms"):
            translated = await lingarr.translate(
                lines=lines,
                source_language=source_language,
                target_language=target_language,
                media_type=media_type,
                title=title_for(video_file),
                arr_media_id=arr_media_id_for(video_file),
            )
    except LingarrCountMismatch as exc:
        _log.error(
            "translation count mismatch expected=%d received=%d",
            exc.expected,
            exc.received,
        )
        return JSONResponse({"detail": "translation count mismatch"}, status_code=502)
    except LingarrPositionMismatch as exc:
        _log.error("translation position mismatch position=%d", exc.position)
        return JSONResponse({"detail": "translation position mismatch"}, status_code=502)
    except LingarrInvalidResponse:
        _log.error("lingarr invalid response")
        return JSONResponse({"detail": "lingarr invalid response"}, status_code=502)
    except LingarrError as exc:
        _log.error("lingarr error: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=502)

    log_extra("translate_cues", float(len(cues)))

    translated_cues: list[Cue] = []
    for i, cue in enumerate(cues):
        pos = i + 1
        text = translated[pos]
        if not text.strip():
            _log.warning("empty translation for position %d", pos)
            text = _NBSP
        translated_cues.append(Cue(start_sec=cue.start_sec, end_sec=cue.end_sec, lines=(text,)))

    srt_text = cues_to_srt(translated_cues)

    try:
        pysrt.from_string(srt_text, error_handling=pysrt.ERROR_RAISE)
    except Exception:
        _log.error("BUG: produced invalid SRT internally after translation")
        return JSONResponse({"detail": "internal SRT error"}, status_code=500)

    return PlainTextResponse(srt_text, headers={"Source": _SOURCE_HEADER})


@router.post("/asr")
async def asr(
    request: Request,
    task: str = Query(...),
    language: str | None = Query(None),
    output: str = Query("srt"),
    encode: str = Query("false"),
    video_file: str | None = Query(None),
) -> Response:
    settings = get_settings(request)
    client = get_openarc_client(request)

    lingarr: LingarrClient | None = None
    if task == "translate":
        lingarr = get_lingarr_client(request)
        if lingarr is None:
            return JSONResponse(
                {"detail": "translate not implemented", "code": "translate_unsupported"},
                status_code=422,
            )

    # Parse form with project-level limit to bypass Starlette's default 1 MB cap.
    form = await request.form(max_part_size=settings.MAX_AUDIO_BYTES)
    try:
        audio_field = form.get("audio_file")
        if not isinstance(audio_field, UploadFile):
            return JSONResponse({"detail": "audio_file required"}, status_code=422)
        async with record_stage("ingest_ms"):
            pcm = await audio_field.read()
    finally:
        await form.close()

    try:
        assert_within_size_limit(pcm, settings.MAX_AUDIO_BYTES)
    except AudioTooLarge as exc:
        return JSONResponse(
            {
                "detail": "audio too large",
                "actual_size": exc.actual_human,
                "max_size": exc.max_human,
            },
            status_code=413,
        )

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

    # Per-segment alignment: Qwen3-ASR already provides utterance-boundary
    # segments (start, end, text); aligning each independently keeps the
    # ctc-forced-aligner DP table small and bounds peak memory regardless
    # of total audio length. If the upstream response lacked segments, fall
    # back to a single segment covering the full audio.
    align_segments = transcription.segments
    if not align_segments:
        audio_duration = float(
            transcription.metrics.get("audio_duration_sec") or len(pcm) / (16000 * 2)
        )
        align_segments = [
            TranscriptionSegment(
                start_sec=0.0,
                end_sec=max(audio_duration, 0.001),
                text=transcription.text,
            )
        ]

    try:
        async with record_stage("align_ms"):
            words = await align(
                audio_float32,
                align_segments,
                align_language,
                _settings=settings,
            )
    except AlignmentFailed as exc:
        _log.warning("alignment failed, using fallback SRT: %s", exc)
        duration = float(transcription.metrics.get("audio_duration_sec") or len(pcm) / (16000 * 2))

        if task != "translate":
            return PlainTextResponse(
                fallback_srt(transcription.text, duration),
                headers={"Source": _SOURCE_HEADER},
            )

        # translate: send the single fallback cue to Lingarr
        assert lingarr is not None
        fallback_cues = [
            Cue(start_sec=0.0, end_sec=max(duration, 0.001), lines=(transcription.text,))
        ]
        return await _do_translate(
            lingarr,
            fallback_cues,
            language or "und",
            settings.LINGARR_TARGET_LANGUAGE,
            settings.LINGARR_DEFAULT_MEDIA_TYPE,
            video_file,
        )

    del audio_float32

    async with record_stage("format_ms"):
        policy = SegmentPolicy(
            max_chars=settings.CUE_MAX_CHARS,
            max_sec=settings.CUE_MAX_SEC,
            min_sec=settings.CUE_MIN_SEC,
            silence_ms=settings.CUE_SILENCE_MS,
            min_chars=settings.CUE_MIN_CHARS,
            max_merge_gap_sec=settings.CUE_MAX_MERGE_GAP_SEC,
        )
        cues = words_to_cues(words, policy)

    if task != "translate":
        srt_text = cues_to_srt(cues)
        try:
            pysrt.from_string(srt_text, error_handling=pysrt.ERROR_RAISE)
        except Exception:
            _log.error("BUG: produced invalid SRT internally")
            return JSONResponse({"detail": "internal SRT error"}, status_code=500)
        return PlainTextResponse(srt_text, headers={"Source": _SOURCE_HEADER})

    assert lingarr is not None
    return await _do_translate(
        lingarr,
        cues,
        language or "und",
        settings.LINGARR_TARGET_LANGUAGE,
        settings.LINGARR_DEFAULT_MEDIA_TYPE,
        video_file,
    )
