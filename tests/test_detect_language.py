"""POST /detect-language route tests — spec task 15 (center-first shifting-window)."""

from __future__ import annotations

import io
import logging
import wave
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from whisper_proxy.app import create_app
from whisper_proxy.openarc import OpenArcClient, OpenArcInferenceError, OpenArcUnavailable
from whisper_proxy.routes.detect import _window_start_samples

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz, mono s16le

_2S_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * 2)
_30S_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * 30)
_120S_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * 120)

OPENARC_BASE = "http://localhost:8000"
TRANSCRIPTIONS_URL = f"{OPENARC_BASE}/v1/audio/transcriptions"
STATUS_URL = f"{OPENARC_BASE}/openarc/status"

# Text ≥ 20 chars so the first attempt passes the MIN_TEXT_CHARS threshold.
_ENGLISH_VERBOSE_JSON = {
    "text": "Hello, welcome to the show. Today we discuss several interesting topics.",
    "language": "English",
    "duration": None,
    "metrics": {"language": "English", "audio_duration_sec": 2.0},
}

# Text < 20 chars — always sub-threshold.
_THIN_JSON = {
    "text": "hi",
    "language": "English",
    "duration": None,
    "metrics": {"language": "English", "audio_duration_sec": 2.0},
}

# Empty text — always sub-threshold.
_EMPTY_JSON = {
    "text": "",
    "language": "English",
    "duration": None,
    "metrics": {"language": "English", "audio_duration_sec": 2.0},
}


def _make_client() -> TestClient:
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


def _extract_wav_from_multipart(body: bytes) -> bytes:
    riff_offset = body.find(b"RIFF")
    assert riff_offset >= 0, "WAV RIFF header not found in captured request"
    return body[riff_offset:]


# ---------------------------------------------------------------------------
# Unit tests: _window_start_samples — criteria 1, 5, 6, 7
# ---------------------------------------------------------------------------


def test_offsets_center_first_order_long_audio() -> None:
    """Criterion 1: for long audio, offsets follow C, C-S, C+S, C-2S, C+2S, ..."""
    sr = SAMPLE_RATE
    # 120 s audio: C=60, S=15, window=10, max=6
    starts = _window_start_samples(120.0, 10.0, 15.0, 6, sr)
    secs = [s / sr for s in starts]
    assert secs == pytest.approx([60, 45, 75, 30, 90, 15])


def test_boundary_clamping_stays_in_range() -> None:
    """Criterion 5: clamped offsets always lie in [0, duration - window]."""
    sr = SAMPLE_RATE
    starts = _window_start_samples(60.0, 10.0, 20.0, 6, sr)
    max_start = round((60.0 - 10.0) * sr)
    for s in starts:
        assert 0 <= s <= max_start


def test_short_audio_produces_single_attempt() -> None:
    """Criterion 6: audio shorter than window_sec → single attempt at start=0."""
    sr = SAMPLE_RATE
    starts = _window_start_samples(5.0, 10.0, 15.0, 6, sr)
    assert starts == [0]


def test_duplicate_offsets_are_skipped() -> None:
    """Criterion 7: after clamping, duplicate start positions are not repeated."""
    sr = SAMPLE_RATE
    # 30 s audio, window=10, shift=20, max=6
    # center=15, max_start=20
    # step0: [15] → ok; step1: [-5→0, 35→20] → ok; step2: [-25→0 dup, 55→20 dup] → skip
    starts = _window_start_samples(30.0, 10.0, 20.0, 6, sr)
    assert len(starts) == 3
    secs = sorted(s / sr for s in starts)
    assert secs == pytest.approx([0.0, 15.0, 20.0])


def test_max_attempts_caps_candidates() -> None:
    """Criterion 1 / 4: at most max_attempts distinct starts are returned."""
    sr = SAMPLE_RATE
    # Very long audio: many possible positions, but capped at 4
    starts = _window_start_samples(3600.0, 10.0, 15.0, 4, sr)
    assert len(starts) == 4


# ---------------------------------------------------------------------------
# HTTP contract tests — criteria 10, 11, 14
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


def test_response_content_type_is_json() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)
    assert "application/json" in resp.headers["content-type"]


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
    assert len(body["language_code"]) in (2, 3)
    assert body["detected_language"] == body["detected_language"].lower()


def test_english_audio_returns_en() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)
    assert resp.json() == {"language_code": "en", "detected_language": "english"}


def test_oversized_audio_returns_413() -> None:
    from whisper_proxy.audio import AudioTooLarge

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.detect.assert_within_size_limit",
            side_effect=AudioTooLarge(actual_bytes=10_000_000, max_bytes=1_000_000),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)
    assert resp.status_code == 413
    assert resp.json()["detail"] == "audio too large"


# ---------------------------------------------------------------------------
# Window search algorithm — criteria 2, 3, 4, 8, 9
# ---------------------------------------------------------------------------


def test_clip_duration_is_window_sec() -> None:
    """Criterion 2: each clip sent to OpenArc is LANG_DETECT_WINDOW_SEC seconds long."""
    captured: list[bytes] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=_capture)
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_30S_PCM)

    assert resp.status_code == 200
    assert len(captured) >= 1
    wav_bytes = _extract_wav_from_multipart(captured[0])
    duration = _wav_duration_sec(wav_bytes)
    # Default LANG_DETECT_WINDOW_SEC=10; allow ±1 sample tolerance
    assert abs(duration - 10.0) < 1.0 / SAMPLE_RATE + 0.001


def test_first_passing_attempt_stops_search() -> None:
    """Criterion 3: once a window passes the threshold, no further calls are made."""
    responses = iter(
        [
            httpx.Response(200, json=_THIN_JSON),
            httpx.Response(200, json=_ENGLISH_VERBOSE_JSON),
        ]
    )
    call_count = 0

    def _multi(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return next(responses)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=_multi)
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_120S_PCM)

    assert resp.status_code == 200
    assert resp.json()["language_code"] == "en"
    assert call_count == 2


def test_exhausted_attempts_returns_und() -> None:
    """Criterion 4: all attempts sub-threshold → und/unknown."""
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_THIN_JSON))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_120S_PCM)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


def test_hallucination_pattern_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 8: hallucination patterns strip matched substrings before the length check."""
    monkeypatch.setenv("LANG_DETECT_HALLUCINATION_PATTERNS", "thank you,subtitles by")
    thank_you_json = {
        "text": "Thank you.",
        "language": "English",
        "duration": None,
        "metrics": {"language": "English", "audio_duration_sec": 2.0},
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=thank_you_json))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with TestClient(create_app()) as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


def test_whitespace_only_text_is_sub_threshold() -> None:
    """Criterion 9: whitespace-only transcription is always sub-threshold."""
    ws_json = {
        "text": "     ",
        "language": "English",
        "duration": None,
        "metrics": {"language": "English", "audio_duration_sec": 2.0},
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=ws_json))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_120S_PCM)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


# ---------------------------------------------------------------------------
# OpenArc error handling — criteria 12, 13
# ---------------------------------------------------------------------------


def test_openarc_unreachable_first_attempt_returns_und() -> None:
    """Criterion 12: OpenArc unreachable on first attempt → HTTP 200 with und/unknown."""
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
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


def test_openarc_error_mid_search_returns_und() -> None:
    """Criterion 13: error mid-search → remaining attempts abandoned, und/unknown returned."""
    call_count = 0

    async def _fail_on_second(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            from whisper_proxy.openarc import Transcription

            return Transcription(text="hi", language="English", duration=None, metrics={})
        raise OpenArcUnavailable("gone away")

    with (
        respx.mock,
        patch.object(OpenArcClient, "transcribe", new=AsyncMock(side_effect=_fail_on_second)),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client, pcm=_120S_PCM)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}
    assert call_count == 2


def test_openarc_5xx_returns_und() -> None:
    """OpenArc inference error is treated as an OpenArc error → und/unknown."""
    with (
        respx.mock,
        patch.object(
            OpenArcClient,
            "transcribe",
            new=AsyncMock(side_effect=OpenArcInferenceError("OOM")),
        ),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert resp.json() == {"language_code": "und", "detected_language": "unknown"}


def test_unknown_language_name_returns_und_with_lowercased_name() -> None:
    """Unknown language in metrics → und code, lowercased name in response."""
    klingon_json = {
        "text": "Heghlu meH QaQ jajvam. tlhIngan maH. nuqneH. yIntagh. batlh bIHeghjaj.",
        "language": "Klingon",
        "duration": None,
        "metrics": {"language": "Klingon", "audio_duration_sec": 2.0},
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=klingon_json))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    body = resp.json()
    assert body["language_code"] == "und"
    assert body["detected_language"] == "klingon"


# ---------------------------------------------------------------------------
# Observability — criteria 15, 17
# ---------------------------------------------------------------------------


def test_summary_log_has_lang_detect_fields(caplog: pytest.LogCaptureFixture) -> None:
    """Criterion 15: summary log includes lang_detect_attempts (int) and lang_detect_hit (bool)."""
    with caplog.at_level(logging.INFO):
        with respx.mock:
            respx.post(TRANSCRIPTIONS_URL).mock(
                return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
            )
            respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
            with _make_client() as client:
                resp = _post_detect(client)

    assert resp.status_code == 200
    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert len(summaries) == 1
    rec = summaries[0]
    assert hasattr(rec, "lang_detect_attempts")
    assert hasattr(rec, "lang_detect_hit")
    assert isinstance(rec.lang_detect_attempts, int)  # type: ignore[attr-defined]
    assert isinstance(rec.lang_detect_hit, bool)  # type: ignore[attr-defined]
    assert rec.lang_detect_attempts >= 1  # type: ignore[attr-defined]
    assert rec.lang_detect_hit is True  # type: ignore[attr-defined]


def test_summary_log_hit_false_when_exhausted(caplog: pytest.LogCaptureFixture) -> None:
    """Criterion 15: lang_detect_hit is False when all attempts are exhausted."""
    with caplog.at_level(logging.INFO):
        with respx.mock:
            respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_THIN_JSON))
            respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
            with _make_client() as client:
                _post_detect(client, pcm=_120S_PCM)

    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    rec = summaries[0]
    assert rec.lang_detect_hit is False  # type: ignore[attr-defined]


def test_openarc_ms_accumulates_across_attempts(caplog: pytest.LogCaptureFixture) -> None:
    """Criterion 17: openarc_ms in summary log is the sum over all per-attempt timings."""
    # Two thin + one passing = three OpenArc calls; openarc_ms must reflect all three.
    responses = iter(
        [
            httpx.Response(200, json=_THIN_JSON),
            httpx.Response(200, json=_THIN_JSON),
            httpx.Response(200, json=_ENGLISH_VERBOSE_JSON),
        ]
    )

    def _multi(request: httpx.Request) -> httpx.Response:
        return next(responses)

    with caplog.at_level(logging.INFO):
        with respx.mock:
            respx.post(TRANSCRIPTIONS_URL).mock(side_effect=_multi)
            respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
            with _make_client() as client:
                _post_detect(client, pcm=_120S_PCM)

    summaries = [r for r in caplog.records if r.getMessage() == "request completed"]
    rec = summaries[0]
    assert hasattr(rec, "openarc_ms")
    assert rec.openarc_ms > 0  # type: ignore[attr-defined]
    assert rec.lang_detect_attempts == 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Bazarr contract replay
# ---------------------------------------------------------------------------


def test_summary_log_fields_no_crash() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(200, json=_ENGLISH_VERBOSE_JSON)
        )
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_detect(client)

    assert resp.status_code == 200
    assert "x-request-id" in resp.headers


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
