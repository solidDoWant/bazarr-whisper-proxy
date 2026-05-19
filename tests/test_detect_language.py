"""POST /detect-language route tests — spec task 10."""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, patch

import httpx
import respx
from fastapi.testclient import TestClient

from whisper_proxy.app import create_app
from whisper_proxy.openarc import OpenArcClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz, mono s16le
_2S_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * 2)  # 2 s silence
_40S_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * 40)  # 40 s silence (> default 30 s head)

OPENARC_BASE = "http://localhost:8000"
TRANSCRIPTIONS_URL = f"{OPENARC_BASE}/v1/audio/transcriptions"
STATUS_URL = f"{OPENARC_BASE}/openarc/status"

_ENGLISH_VERBOSE_JSON = {
    "text": "Hello.",
    "language": "English",
    "duration": None,
    "metrics": {"language": "English", "audio_duration_sec": 2.0},
}


def _make_client() -> httpx.Client:
    return TestClient(create_app())


def _post_detect(
    client: httpx.Client,
    *,
    encode: str = "false",
    video_file: str | None = None,
    pcm: bytes = _2S_PCM,
) -> httpx.Response:
    params: dict[str, str] = {"encode": encode}
    if video_file is not None:
        params["video_file"] = video_file
    return client.post(
        "/detect-language",
        params=params,
        files={"audio_file": ("audio.pcm", pcm, "application/octet-stream")},
    )


def _wav_duration_sec(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# Criterion 1: happy path returns 200
# ---------------------------------------------------------------------------


def test_happy_path_returns_200() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Criterion 2: Content-Type is application/json
# ---------------------------------------------------------------------------


def test_response_content_type_is_json() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Criterion 3: response body has exactly language_code and detected_language
# ---------------------------------------------------------------------------


def test_response_body_shape() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    body = resp.json()
    assert set(body.keys()) == {"language_code", "detected_language"}
    assert len(body["language_code"]) in (2, 3)  # alpha-2 or "und"
    assert body["detected_language"] == body["detected_language"].lower()


# ---------------------------------------------------------------------------
# Criterion 4: English audio returns en / english
# ---------------------------------------------------------------------------


def test_english_audio_returns_en() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.json() == {"language_code": "en", "detected_language": "english"}


# ---------------------------------------------------------------------------
# Criterion 5: audio longer than LANG_DETECT_HEAD_SEC is clipped
# ---------------------------------------------------------------------------


def test_long_audio_is_clipped_to_head() -> None:
    captured: list[bytes] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        body = request.content
        captured.append(body)
        return httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=_capture)
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_40S_PCM)

    assert resp.status_code == 200
    assert len(captured) == 1

    # The multipart body contains a WAV file; parse its duration from the WAV header.
    # Find the WAV bytes embedded in the multipart by looking for the RIFF header.
    raw_body = captured[0]
    riff_offset = raw_body.find(b"RIFF")
    assert riff_offset >= 0, "WAV RIFF header not found in captured request"
    wav_bytes = raw_body[riff_offset:]
    duration = _wav_duration_sec(wav_bytes)
    assert duration <= 30.0, f"Expected clipped audio ≤ 30 s, got {duration:.2f} s"


# ---------------------------------------------------------------------------
# Criterion 6: OpenArc 5xx → 200 with und/unknown
# ---------------------------------------------------------------------------


def test_openarc_5xx_returns_und() -> None:
    with (
        respx.mock,
        patch.object(
            OpenArcClient,
            "detect_language",
            new=AsyncMock(
                side_effect=__import__(
                    "whisper_proxy.openarc", fromlist=["OpenArcInferenceError"]
                ).OpenArcInferenceError("OOM")
            ),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


# ---------------------------------------------------------------------------
# Criterion 7: OpenArc unreachable → 200 with und/unknown
# ---------------------------------------------------------------------------


def test_openarc_unreachable_returns_und() -> None:
    with (
        respx.mock,
        patch.object(
            OpenArcClient,
            "detect_language",
            new=AsyncMock(
                side_effect=__import__(
                    "whisper_proxy.openarc", fromlist=["OpenArcUnavailable"]
                ).OpenArcUnavailable("connection refused")
            ),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


# ---------------------------------------------------------------------------
# Criterion 8: unknown language name → und + lowercased name
# ---------------------------------------------------------------------------


def test_unknown_language_name_returns_und_with_lowercased_name() -> None:
    unknown_json = {
        "text": "",
        "language": "Klingon",
        "duration": None,
        "metrics": {"language": "Klingon", "audio_duration_sec": 2.0},
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=unknown_json))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    body = resp.json()
    assert body["language_code"] == "und"
    assert body["detected_language"] == "klingon"


# ---------------------------------------------------------------------------
# Criterion 9: audio > MAX_AUDIO_BYTES → 413
# ---------------------------------------------------------------------------


def test_oversized_audio_returns_413() -> None:
    from whisper_proxy.audio import AudioTooLarge

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.detect.assert_within_size_limit",
            side_effect=AudioTooLarge("too big"),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 413
    assert resp.json()["detail"] == "audio too large"


# ---------------------------------------------------------------------------
# Criterion 10: summary log fields — smoke (no crash + middleware fires)
# ---------------------------------------------------------------------------


def test_summary_log_fields_no_crash() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    # middleware always fires; if it crashes we'd get 500
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# Criterion 11: contract replay — exact Bazarr multipart shape
# ---------------------------------------------------------------------------


def test_contract_replay_bazarr_detect_language_shape() -> None:
    """Replay the exact multipart body WhisperAIProvider.detect_language produces."""
    params = {
        "encode": "false",
        "video_file": "/media/show/s01e01.mkv",
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = client.post(
                "/detect-language",
                params=params,
                files={"audio_file": ("audio.pcm", _2S_PCM, "application/octet-stream")},
                headers={"User-Agent": "Subliminal/2.2.1"},
            )

    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    body = resp.json()
    assert set(body.keys()) == {"language_code", "detected_language"}
    assert body["language_code"] == "en"
    assert body["detected_language"] == "english"
