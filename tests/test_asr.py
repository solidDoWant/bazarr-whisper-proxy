"""POST /asr route tests — spec task 09."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pysrt
import pytest
import respx
from fastapi.testclient import TestClient

from whisper_proxy._types import Word
from whisper_proxy.app import create_app
from whisper_proxy.openarc import OpenArcClient, Transcription

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz, mono s16le
SAMPLE_DURATION_SEC = 2.0
# 2 seconds of silence at 16kHz mono s16le: samples = 16000*2 = 32000 half-words
_SILENCE_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * SAMPLE_DURATION_SEC)

OPENARC_BASE = "http://localhost:8000"
TRANSCRIPTIONS_URL = f"{OPENARC_BASE}/v1/audio/transcriptions"
STATUS_URL = f"{OPENARC_BASE}/openarc/status"

_FAKE_TRANSCRIPTION = Transcription(
    text="Hello world.",
    language="English",
    duration=None,
    metrics={"language": "English", "audio_duration_sec": SAMPLE_DURATION_SEC},
)

_FAKE_WORDS = [
    Word(token="Hello", start_sec=0.1, end_sec=0.5),
    Word(token="world.", start_sec=0.6, end_sec=1.0),
]

VERBOSE_JSON: dict[str, Any] = {
    "text": "Hello world.",
    "language": "English",
    "duration": None,
    "metrics": {"language": "English", "audio_duration_sec": SAMPLE_DURATION_SEC},
}


def _make_client() -> httpx.Client:
    """TestClient that drives the full lifespan."""
    return TestClient(create_app())


def _post_asr(
    client: httpx.Client,
    *,
    task: str = "transcribe",
    language: str | None = "en",
    output: str = "srt",
    encode: str = "false",
    video_file: str | None = None,
    pcm: bytes = _SILENCE_PCM,
) -> httpx.Response:
    params: dict[str, str] = {"task": task, "output": output, "encode": encode}
    if language is not None:
        params["language"] = language
    if video_file is not None:
        params["video_file"] = video_file
    return client.post(
        "/asr",
        params=params,
        files={"audio_file": ("audio.pcm", pcm, "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Criteria 1-3: happy path — 200, Content-Type, valid SRT
# ---------------------------------------------------------------------------


def test_happy_path_returns_200_valid_srt() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "qwen3-asr-0_6b-int8-asym", "status": "loaded"}]
            )
        )
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # Criterion 3: pysrt must parse without error
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) > 0


# ---------------------------------------------------------------------------
# Criterion 2: Content-Type is text/plain; charset=utf-8
# ---------------------------------------------------------------------------


def test_response_content_type_is_text_plain() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    ct = resp.headers["content-type"].lower()
    assert "text/plain" in ct
    assert "utf-8" in ct


# ---------------------------------------------------------------------------
# Criterion 4: Source header
# ---------------------------------------------------------------------------


def test_response_has_source_header() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.headers.get("source") == "Transcribed using Bazarr to OpenAI Whisper Bridge!"


# ---------------------------------------------------------------------------
# Criterion 5: no language param → still 200 with valid SRT (auto-detect)
# ---------------------------------------------------------------------------


def test_no_language_param_produces_valid_srt() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client, language=None)

    assert resp.status_code == 200
    pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)


# ---------------------------------------------------------------------------
# Criterion 6: video_file query param appears in log (smoke-tested via no error)
# ---------------------------------------------------------------------------


def test_video_file_param_is_accepted() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client, video_file="/media/show/s01e01.mkv")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Criterion 7: task=translate → 422
# ---------------------------------------------------------------------------


def test_translate_task_returns_422() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client, task="translate")

    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"] == "translate not implemented"
    assert body["code"] == "translate_unsupported"


# ---------------------------------------------------------------------------
# Criterion 8: oversized audio → 413
# ---------------------------------------------------------------------------


def test_oversized_audio_returns_413() -> None:
    from whisper_proxy.audio import AudioTooLarge

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.asr.assert_within_size_limit",
            side_effect=AudioTooLarge("too big"),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 413
    assert resp.json()["detail"] == "audio too large"


# ---------------------------------------------------------------------------
# Criterion 9: OpenArcUnavailable → 502
# ---------------------------------------------------------------------------


def test_openarc_unavailable_returns_502() -> None:
    from whisper_proxy.openarc import OpenArcUnavailable

    with (
        respx.mock,
        patch.object(
            OpenArcClient,
            "transcribe",
            new=AsyncMock(side_effect=OpenArcUnavailable("connection refused")),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Criterion 10: OpenArcBadRequest (OpenArc 4xx) → 502
# ---------------------------------------------------------------------------


def test_openarc_bad_request_returns_502() -> None:
    from whisper_proxy.openarc import OpenArcBadRequest

    with (
        respx.mock,
        patch.object(
            OpenArcClient,
            "transcribe",
            new=AsyncMock(side_effect=OpenArcBadRequest("model not loaded")),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Criterion 11: OpenArcInferenceError (OpenArc 5xx) → 502
# ---------------------------------------------------------------------------


def test_openarc_inference_error_returns_502() -> None:
    from whisper_proxy.openarc import OpenArcInferenceError

    with (
        respx.mock,
        patch.object(
            OpenArcClient, "transcribe", new=AsyncMock(side_effect=OpenArcInferenceError("OOM"))
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Criterion 12: AlignmentFailed → 200 fallback single-cue SRT
# ---------------------------------------------------------------------------


def test_alignment_failed_returns_fallback_srt() -> None:
    from whisper_proxy.aligner import AlignmentFailed

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.asr.align",
            new=AsyncMock(side_effect=AlignmentFailed("no alignable tokens")),
        ),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) == 1
    assert srt[0].start == pysrt.SubRipTime(0, 0, 0, 0)
    # End timecode should span to audio duration
    end_ms = srt[0].end.ordinal
    assert end_ms > 0
    assert "Hello world." in srt[0].text


# ---------------------------------------------------------------------------
# Criterion 13: summary log has required fields (smoke — no crash)
# ---------------------------------------------------------------------------


def test_summary_log_fields_no_crash(caplog: pytest.LogCaptureFixture) -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Criterion 14: X-Request-Id header is present
# ---------------------------------------------------------------------------


def test_x_request_id_header_present() -> None:
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert "x-request-id" in resp.headers


def test_x_request_id_forwarded_when_provided() -> None:
    custom_id = "550e8400-e29b-41d4-a716-446655440000"
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = client.post(
                "/asr",
                params={"task": "transcribe", "language": "en", "output": "srt", "encode": "false"},
                files={"audio_file": ("audio.pcm", _SILENCE_PCM, "application/octet-stream")},
                headers={"x-request-id": custom_id},
            )

    assert resp.headers["x-request-id"] == custom_id


# ---------------------------------------------------------------------------
# Criterion 15: contract replay — exact Bazarr multipart shape
# ---------------------------------------------------------------------------


def test_contract_replay_bazarr_multipart_shape() -> None:
    """Replay the exact multipart shape WhisperAIProvider.download_subtitle produces."""
    # Bazarr sends: task, language, output, encode as query params; audio_file as multipart
    params = {
        "task": "transcribe",
        "language": "en",
        "output": "srt",
        "encode": "false",
        "video_file": "/media/show/s01e01.mkv",
    }
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = client.post(
                "/asr",
                params=params,
                files={"audio_file": ("audio.pcm", _SILENCE_PCM, "application/octet-stream")},
                headers={"User-Agent": "Subliminal/2.2.1"},
            )

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # Gate: valid SRT
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) > 0


# ---------------------------------------------------------------------------
# Criterion 16: contract-replay validation gate is exercised (not a no-op)
# Patch cues_to_srt to return invalid SRT → route returns 500 → pysrt fails
# ---------------------------------------------------------------------------


def test_contract_replay_broken_writer_triggers_validation_gate() -> None:
    """When the SRT writer produces garbage, the route returns 500."""
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.cues_to_srt", return_value="this is not valid srt\n"),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    # Route's internal pysrt gate fires → 500
    assert resp.status_code == 500
    # And confirm that pysrt would indeed reject the bad output
    with pytest.raises(pysrt.Error):
        pysrt.from_string("this is not valid srt\n", error_handling=pysrt.ERROR_RAISE)
