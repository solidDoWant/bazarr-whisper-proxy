import logging

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, head_clip, pcm_to_wav
from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.lang import name_to_alpha2, normalize_name
from whisper_proxy.logging_setup import record_stage
from whisper_proxy.openarc import OpenArcError

_log = logging.getLogger(__name__)

router = APIRouter()

_UND_RESPONSE = {"language_code": "und", "detected_language": "unknown"}


@router.post("/detect-language")
async def detect_language(
    request: Request,
    encode: str = Query("false"),
    video_file: str | None = Query(None),
    audio_file: UploadFile = File(...),
) -> Response:
    settings = get_settings(request)
    client = get_openarc_client(request)

    async with record_stage("ingest_ms"):
        pcm = await audio_file.read()

    try:
        assert_within_size_limit(pcm, settings.MAX_AUDIO_BYTES)
    except AudioTooLarge:
        return JSONResponse({"detail": "audio too large"}, status_code=413)

    if video_file:
        _log.info("request summary video_file=%s", video_file)

    clipped = head_clip(pcm, settings.LANG_DETECT_HEAD_SEC)
    audio_wav = pcm_to_wav(clipped)

    try:
        async with record_stage("openarc_ms"):
            raw_language = await client.detect_language(audio_wav)
    except OpenArcError as exc:
        _log.warning("openarc error during language detection: %s", exc)
        return JSONResponse(_UND_RESPONSE)

    if not raw_language or not raw_language.strip():
        return JSONResponse(_UND_RESPONSE)

    detected_language = normalize_name(raw_language)
    language_code = name_to_alpha2(raw_language)

    return JSONResponse({"language_code": language_code, "detected_language": detected_language})
