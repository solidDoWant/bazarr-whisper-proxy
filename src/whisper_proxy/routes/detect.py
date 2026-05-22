import logging
import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from whisper_proxy.audio import AudioTooLarge, assert_within_size_limit, pcm_to_wav
from whisper_proxy.config import Settings
from whisper_proxy.deps import get_openarc_client, get_settings
from whisper_proxy.lang import name_to_alpha2, normalize_name
from whisper_proxy.logging_setup import record_stage, stage_timings_var
from whisper_proxy.openarc import OpenArcClient, OpenArcError

_log = logging.getLogger(__name__)

router = APIRouter()

_UND_RESPONSE = {"language_code": "und", "detected_language": "unknown"}

_SAMPLE_RATE = 16_000
_FRAME_SIZE = 2  # s16le mono


def _window_start_samples(
    duration_sec: float,
    window_sec: float,
    shift_sec: float,
    max_attempts: int,
    sample_rate: int = _SAMPLE_RATE,
) -> list[int]:
    """Center-first shifting window: return up to max_attempts distinct clamped start samples."""
    max_start_sec = max(0.0, duration_sec - window_sec)
    center_sec = duration_sec / 2
    low_sample = 0
    high_sample = round(max_start_sec * sample_rate)
    seen: set[int] = set()
    results: list[int] = []
    step = 0

    while len(results) < max_attempts:
        raws = (
            [center_sec]
            if step == 0
            else [center_sec - step * shift_sec, center_sec + step * shift_sec]
        )
        for raw_sec in raws:
            if len(results) >= max_attempts:
                break
            clamped = max(0.0, min(raw_sec, max_start_sec))
            s = round(clamped * sample_rate)
            if s not in seen:
                seen.add(s)
                results.append(s)
        step += 1
        # Once both boundary clamps are exhausted, all future offsets will duplicate.
        if low_sample in seen and high_sample in seen:
            break

    return results


def _strip_patterns(text: str, patterns: list[str]) -> str:
    for p in patterns:
        text = re.sub(re.escape(p), "", text, flags=re.IGNORECASE)
    return text.strip()


async def detect_language_window_search(
    pcm: bytes,
    client: OpenArcClient,
    settings: Settings,
) -> tuple[str | None, int, bool]:
    """Center-first shifting-window language detection. Returns (raw_language, attempts, hit)."""
    total_samples = len(pcm) // _FRAME_SIZE
    duration_sec = total_samples / _SAMPLE_RATE
    window_samples = round(settings.LANG_DETECT_WINDOW_SEC * _SAMPLE_RATE)
    patterns = [
        p.strip() for p in settings.LANG_DETECT_HALLUCINATION_PATTERNS.split(",") if p.strip()
    ]

    starts = _window_start_samples(
        duration_sec,
        float(settings.LANG_DETECT_WINDOW_SEC),
        float(settings.LANG_DETECT_SHIFT_SEC),
        settings.LANG_DETECT_MAX_ATTEMPTS,
    )

    for attempt_idx, start_sample in enumerate(starts):
        clip_pcm = pcm[start_sample * _FRAME_SIZE : (start_sample + window_samples) * _FRAME_SIZE]
        wav = pcm_to_wav(clip_pcm)
        offset_sec = start_sample / _SAMPLE_RATE

        try:
            async with record_stage("openarc_ms"):
                tr = await client.transcribe(wav, language=None)
        except OpenArcError as exc:
            _log.warning("lang_detect attempt=%d error: %s", attempt_idx + 1, exc)
            return None, attempt_idx + 1, False

        text = _strip_patterns(tr.text.strip(), patterns)
        passed = len(text) >= settings.LANG_DETECT_MIN_TEXT_CHARS

        _log.debug(
            "lang_detect offset=%.2fs text_len=%d %s",
            offset_sec,
            len(text),
            "accepted" if passed else "rejected",
        )

        if passed:
            raw_lang = str(tr.metrics.get("language", ""))
            return raw_lang if raw_lang.strip() else None, attempt_idx + 1, True

    return None, len(starts), False


@router.post("/detect-language")
async def detect_language(
    request: Request,
    encode: str = Query("false"),
    video_file: str | None = Query(None),
) -> Response:
    settings = get_settings(request)
    client = get_openarc_client(request)

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
            {"detail": "audio too large", "actual_size": exc.actual_human, "max_size": exc.max_human},
            status_code=413,
        )

    if video_file:
        _log.info("request summary video_file=%s", video_file)

    raw_language, attempts, hit = await detect_language_window_search(pcm, client, settings)

    timings = stage_timings_var.get(None)
    if timings is not None:
        timings["lang_detect_attempts"] = attempts
        timings["lang_detect_hit"] = hit

    if not hit or not raw_language:
        return JSONResponse(_UND_RESPONSE)

    detected_language = normalize_name(raw_language)
    language_code = name_to_alpha2(raw_language)

    return JSONResponse({"language_code": language_code, "detected_language": detected_language})
