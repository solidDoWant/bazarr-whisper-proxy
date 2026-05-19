import logging

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, head_clip, pcm_to_wav
from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.lang.map import UNDETERMINED, name_to_alpha2, normalize_name
from whisper_proxy.logging_setup import record_stage

_log = logging.getLogger(__name__)

_FALLBACK = {"language_code": UNDETERMINED, "detected_language": "unknown"}

router = APIRouter()


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

    clipped = head_clip(pcm, float(settings.LANG_DETECT_HEAD_SEC))
    audio_wav = pcm_to_wav(clipped)

    try:
        async with record_stage("openarc_ms"):
            raw_lang = await client.detect_language(audio_wav)
    except Exception as exc:
        _log.warning("language detection failed, returning und: %s", exc)
        return JSONResponse(_FALLBACK)

    alpha2 = name_to_alpha2(raw_lang)
    detected = normalize_name(raw_lang) if raw_lang.strip() else "unknown"

    return JSONResponse({"language_code": alpha2, "detected_language": detected})
