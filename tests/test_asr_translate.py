"""POST /asr task=translate tests — spec task 14."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pysrt
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import AnyHttpUrl

from whisper_proxy._types import Word
from whisper_proxy.app import create_app
from whisper_proxy.lingarr import LingarrClient, arr_media_id_for, title_for
from whisper_proxy.openarc import Transcription

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
SAMPLE_DURATION_SEC = 2.0
_SILENCE_PCM: bytes = b"\x00\x00" * int(SAMPLE_RATE * SAMPLE_DURATION_SEC)

OPENARC_BASE = "http://localhost:8000"
TRANSCRIPTIONS_URL = f"{OPENARC_BASE}/v1/audio/transcriptions"
STATUS_URL = f"{OPENARC_BASE}/openarc/status"

LINGARR_BASE = "http://lingarr:9876"
LINGARR_TRANSLATE_URL = f"{LINGARR_BASE}/api/Translate/content"

_FAKE_WORDS = [
    Word(token="Hola", start_sec=0.1, end_sec=0.4),
    Word(token="mundo.", start_sec=0.5, end_sec=0.9),
]

_VERBOSE_JSON_ES: dict[str, Any] = {
    "text": "Hola mundo.",
    "language": "Spanish",
    "duration": None,
    "metrics": {"language": "Spanish", "audio_duration_sec": SAMPLE_DURATION_SEC},
}

_FAKE_TRANSCRIPTION_ES = Transcription(
    text="Hola mundo.",
    language="Spanish",
    duration=None,
    metrics={"language": "Spanish", "audio_duration_sec": SAMPLE_DURATION_SEC},
)


def _make_client() -> httpx.Client:
    return TestClient(create_app())


def _post_asr(
    client: httpx.Client,
    *,
    task: str = "translate",
    language: str | None = "es",
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


def _mock_lingarr(translated: dict[int, str]) -> MagicMock:
    m = MagicMock(spec=LingarrClient)
    m.translate = AsyncMock(return_value=translated)
    return m


# ---------------------------------------------------------------------------
# Criterion 1: feature disabled → 422
# ---------------------------------------------------------------------------


def test_translate_feature_disabled_returns_422() -> None:
    """LINGARR_BASE_URL unset → 422 with translate_unsupported code."""
    with respx.mock:
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"] == "translate not implemented"
    assert body["code"] == "translate_unsupported"


# ---------------------------------------------------------------------------
# Criterion 2: startup aborts when base URL set without API key
# ---------------------------------------------------------------------------


def test_startup_aborts_when_lingarr_api_key_missing() -> None:
    """LINGARR_BASE_URL set but LINGARR_API_KEY empty → ValidationError from Settings."""
    from pydantic import ValidationError

    from whisper_proxy.config import Settings

    with pytest.raises(ValidationError, match="LINGARR_API_KEY must be set"):
        Settings(
            LINGARR_BASE_URL=AnyHttpUrl("http://lingarr:9876"),
            LINGARR_API_KEY="",
        )


# ---------------------------------------------------------------------------
# Criteria 3-4: happy path returns 200 with pysrt-valid SRT
# ---------------------------------------------------------------------------


def test_translate_happy_path_200_valid_srt() -> None:
    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) > 0


# ---------------------------------------------------------------------------
# Criterion 5: cue text is the English translation, not Spanish
# ---------------------------------------------------------------------------


def test_translate_cue_text_is_english_output() -> None:
    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert srt[0].text == "Hello world."


# ---------------------------------------------------------------------------
# Criterion 6: cue timing is bit-exact to transcribe with same audio
# ---------------------------------------------------------------------------


def test_translate_timing_matches_transcribe() -> None:
    # Get transcribe timings
    verbose_en: dict[str, Any] = {
        "text": "Hello world.",
        "language": "English",
        "duration": None,
        "metrics": {"language": "English", "audio_duration_sec": SAMPLE_DURATION_SEC},
    }
    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=verbose_en))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            transcribe_resp = _post_asr(client, task="transcribe", language="en")

    transcribe_srt = pysrt.from_string(transcribe_resp.text, error_handling=pysrt.ERROR_RAISE)
    n = len(transcribe_srt)

    # Get translate timings with same words
    mock_lingarr = _mock_lingarr({i + 1: f"Translated {i}" for i in range(n)})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            translate_resp = _post_asr(client, task="translate", language="es")

    translate_srt = pysrt.from_string(translate_resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(translate_srt) == n
    for src, tr in zip(transcribe_srt, translate_srt, strict=True):
        assert src.start == tr.start
        assert src.end == tr.end


# ---------------------------------------------------------------------------
# Criterion 7: cue indices are sequential starting at 1
# ---------------------------------------------------------------------------


def test_translate_cue_indices_sequential() -> None:
    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    for i, item in enumerate(srt, 1):
        assert item.index == i


# ---------------------------------------------------------------------------
# Criterion 8: translate cue count equals source-language cue count
# ---------------------------------------------------------------------------


def test_translate_cue_count_matches_source() -> None:
    # Multi-word list produces multiple cues. The gap between the cues must
    # exceed CUE_MAX_MERGE_GAP_SEC (1.5s) after min-duration extension, or
    # _merge_short_cues will combine these two short cues into one.
    words = [
        Word(token="One.", start_sec=0.0, end_sec=0.5),
        Word(token="Two.", start_sec=3.0, end_sec=3.5),  # large gap → separate cue
    ]
    mock_lingarr = _mock_lingarr({1: "One.", 2: "Two."})

    verbose: dict[str, Any] = {
        "text": "One. Two.",
        "language": "Spanish",
        "duration": None,
        "metrics": {"audio_duration_sec": 4.0},
    }

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=words)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=verbose))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) == 2


# ---------------------------------------------------------------------------
# Criteria 9-10: Lingarr request shape — unit-test the client directly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_lingarr_client_request_shape() -> None:
    """Client sends correct URL, headers, and body shape."""
    from whisper_proxy.config import Settings

    with respx.mock:
        route = respx.post(LINGARR_TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=[{"position": 1, "line": "Hello world."}])
        )

        settings = Settings(
            LINGARR_BASE_URL=AnyHttpUrl(LINGARR_BASE),
            LINGARR_API_KEY="secret-key",
        )
        async with LingarrClient(settings) as lingarr:
            await lingarr.translate(
                lines=[(1, "Hola mundo.")],
                source_language="es",
                target_language="en",
                media_type="Episode",
                title="s01e01.mkv",
                arr_media_id=42,
            )

    req = route.calls[0].request
    # Criterion 9: X-Api-Key header set; no Authorization header
    assert req.headers["x-api-key"] == "secret-key"
    assert "authorization" not in {k.lower() for k in req.headers}
    assert req.headers["content-type"].startswith("application/json")

    # Criterion 10: body fields
    body = json.loads(req.content)
    assert body["sourceLanguage"] == "es"
    assert body["targetLanguage"] == "en"
    assert body["mediaType"] == "Episode"
    assert body["title"] == "s01e01.mkv"
    assert body["arrMediaId"] == 42
    assert body["lines"] == [{"position": 1, "line": "Hola mundo."}]


# ---------------------------------------------------------------------------
# Criterion 11: empty/whitespace cue line → NBSP sent to Lingarr
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_lingarr_empty_cue_line_sent_as_nbsp() -> None:
    from whisper_proxy.config import Settings

    with respx.mock:
        route = respx.post(LINGARR_TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=[{"position": 1, "line": "\u00a0"}])
        )

        settings = Settings(
            LINGARR_BASE_URL=AnyHttpUrl(LINGARR_BASE),
            LINGARR_API_KEY="key",
        )
        async with LingarrClient(settings) as lingarr:
            await lingarr.translate(
                lines=[(1, "   ")],  # whitespace-only
                source_language="es",
                target_language="en",
                media_type="Episode",
                title="test",
                arr_media_id=0,
            )

    body = json.loads(route.calls[0].request.content)
    assert body["lines"][0]["line"] == "\u00a0"


# ---------------------------------------------------------------------------
# Criterion 12: arr_media_id computation
# ---------------------------------------------------------------------------


def test_arr_media_id_deterministic() -> None:
    path = "/media/show/s01e01.mkv"
    assert arr_media_id_for(path) == arr_media_id_for(path)


def test_arr_media_id_within_int32_range() -> None:
    assert 0 <= arr_media_id_for("/media/show/s01e01.mkv") <= 0x7FFF_FFFF


def test_arr_media_id_zero_when_absent() -> None:
    assert arr_media_id_for(None) == 0


def test_title_for_uses_basename() -> None:
    assert title_for("/media/show/s01e01.mkv") == "s01e01.mkv"


def test_title_for_fallback_when_absent() -> None:
    assert title_for(None) == "bazarr-whisper-proxy"


# ---------------------------------------------------------------------------
# Criterion 13: reconciliation by position, not array index
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_lingarr_out_of_order_response_reconciled_correctly() -> None:
    from whisper_proxy.config import Settings

    # Lingarr returns positions in reverse order
    response_data = [{"position": 2, "line": "Second"}, {"position": 1, "line": "First"}]

    with respx.mock:
        respx.post(LINGARR_TRANSLATE_URL).mock(return_value=httpx.Response(200, json=response_data))

        settings = Settings(
            LINGARR_BASE_URL=AnyHttpUrl(LINGARR_BASE),
            LINGARR_API_KEY="key",
        )
        async with LingarrClient(settings) as lingarr:
            result = await lingarr.translate(
                lines=[(1, "Primero"), (2, "Segundo")],
                source_language="es",
                target_language="en",
                media_type="Episode",
                title="test",
                arr_media_id=0,
            )

    assert result[1] == "First"
    assert result[2] == "Second"


# ---------------------------------------------------------------------------
# Criterion 14: count mismatch → 502
# ---------------------------------------------------------------------------


def test_lingarr_count_mismatch_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrCountMismatch

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(side_effect=LingarrCountMismatch(expected=3, received=2))

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502
    assert resp.json()["detail"] == "translation count mismatch"


# ---------------------------------------------------------------------------
# Criterion 15: position mismatch → 502
# ---------------------------------------------------------------------------


def test_lingarr_position_mismatch_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrPositionMismatch

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(side_effect=LingarrPositionMismatch(position=99))

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502
    assert resp.json()["detail"] == "translation position mismatch"


# ---------------------------------------------------------------------------
# Criterion 16: empty response line → NBSP in output SRT
# ---------------------------------------------------------------------------


def test_empty_translated_line_replaced_with_nbsp() -> None:
    # Lingarr returns empty string for a cue
    mock_lingarr = _mock_lingarr({1: ""})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    # SRT must parse without error (pysrt-valid despite the NBSP line)
    pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    # NBSP (U+00A0) must appear literally in the raw response body
    assert "\u00a0" in resp.text


# ---------------------------------------------------------------------------
# Criteria 17-18: Lingarr 4xx/5xx → 502
# ---------------------------------------------------------------------------


def test_lingarr_4xx_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrBadRequest

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(side_effect=LingarrBadRequest("HTTP 400"))

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


def test_lingarr_5xx_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrServerError

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(side_effect=LingarrServerError("HTTP 503"))

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Criterion 19: Lingarr unreachable → 502
# ---------------------------------------------------------------------------


def test_lingarr_unreachable_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrUnavailable

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(side_effect=LingarrUnavailable("connection refused"))

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Criterion 20: non-JSON Lingarr response → 502 "lingarr invalid response"
# ---------------------------------------------------------------------------


def test_lingarr_invalid_response_returns_502() -> None:
    from whisper_proxy.lingarr import LingarrInvalidResponse

    mock_lingarr = MagicMock(spec=LingarrClient)
    mock_lingarr.translate = AsyncMock(
        side_effect=LingarrInvalidResponse("non-JSON response from Lingarr")
    )

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502
    assert resp.json()["detail"] == "lingarr invalid response"


@pytest.mark.anyio
async def test_lingarr_client_non_json_body_raises_invalid_response() -> None:
    from whisper_proxy.config import Settings
    from whisper_proxy.lingarr import LingarrInvalidResponse

    with respx.mock:
        respx.post(LINGARR_TRANSLATE_URL).mock(
            return_value=httpx.Response(
                200, text="not json", headers={"content-type": "text/plain"}
            )
        )

        settings = Settings(
            LINGARR_BASE_URL=AnyHttpUrl(LINGARR_BASE),
            LINGARR_API_KEY="key",
        )
        async with LingarrClient(settings) as lingarr:
            with pytest.raises(LingarrInvalidResponse):
                await lingarr.translate(
                    lines=[(1, "text")],
                    source_language="es",
                    target_language="en",
                    media_type="Episode",
                    title="test",
                    arr_media_id=0,
                )


# ---------------------------------------------------------------------------
# Criterion 21: OpenArc 5xx during translate → 502
# ---------------------------------------------------------------------------


def test_openarc_5xx_during_translate_returns_502() -> None:
    from whisper_proxy.openarc import OpenArcInferenceError

    mock_lingarr = MagicMock(spec=LingarrClient)

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.asr.get_openarc_client",
            return_value=MagicMock(
                transcribe=AsyncMock(side_effect=OpenArcInferenceError("GPU OOM"))
            ),
        ),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 502
    # Lingarr should never be called
    mock_lingarr.translate.assert_not_called()


# ---------------------------------------------------------------------------
# Criterion 22: alignment failure during translate → single translated cue
# ---------------------------------------------------------------------------


def test_alignment_failed_during_translate_sends_fallback_to_lingarr() -> None:
    from whisper_proxy.aligner import AlignmentFailed

    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch(
            "whisper_proxy.routes.asr.align",
            new=AsyncMock(side_effect=AlignmentFailed("no tokens")),
        ),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with _make_client() as client:
            resp = _post_asr(client)

    assert resp.status_code == 200
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) == 1
    assert srt[0].text == "Hello world."

    # Verify Lingarr was called with the fallback cue
    mock_lingarr.translate.assert_called_once()
    call_lines = mock_lingarr.translate.call_args.kwargs["lines"]
    assert len(call_lines) == 1
    assert call_lines[0][0] == 1  # position 1
    assert call_lines[0][1] == "Hola mundo."  # source text


# ---------------------------------------------------------------------------
# Criteria 23-24: translate_ms and translate_cues appear in summary log
# ---------------------------------------------------------------------------


def test_translate_summary_includes_translate_ms_and_cues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with caplog.at_level("INFO"):
            with _make_client() as client:
                resp = _post_asr(client)

    assert resp.status_code == 200

    summary_records = [r for r in caplog.records if "request completed" in r.getMessage()]
    assert summary_records, "no 'request completed' log record found"
    record = summary_records[-1]
    assert hasattr(record, "translate_ms"), "translate_ms missing from summary"
    assert hasattr(record, "translate_cues"), "translate_cues missing from summary"


# ---------------------------------------------------------------------------
# Criterion 25: transcribe task does NOT include translate_ms / translate_cues
# ---------------------------------------------------------------------------


def test_transcribe_summary_excludes_translate_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    verbose_en: dict[str, Any] = {
        "text": "Hello.",
        "language": "English",
        "duration": None,
        "metrics": {"audio_duration_sec": SAMPLE_DURATION_SEC},
    }
    words = [Word(token="Hello.", start_sec=0.1, end_sec=0.5)]

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=words)),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=verbose_en))
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        with caplog.at_level("INFO"):
            with _make_client() as client:
                client.post(
                    "/asr",
                    params={
                        "task": "transcribe",
                        "language": "en",
                        "output": "srt",
                        "encode": "false",
                    },
                    files={"audio_file": ("audio.pcm", _SILENCE_PCM, "application/octet-stream")},
                )

    summary_records = [r for r in caplog.records if "request completed" in r.getMessage()]
    assert summary_records
    record = summary_records[-1]
    assert not hasattr(record, "translate_ms"), "translate_ms should not appear for transcribe"
    assert not hasattr(record, "translate_cues"), "translate_cues should not appear for transcribe"


# ---------------------------------------------------------------------------
# Criterion 26: X-Api-Key never appears in log records
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_api_key_not_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from whisper_proxy.config import Settings
    from whisper_proxy.logging_setup import configure_logging

    configure_logging("DEBUG", "text")
    api_key = "super-secret-api-key-12345"

    with respx.mock, caplog.at_level("DEBUG"):
        respx.post(LINGARR_TRANSLATE_URL).mock(
            return_value=httpx.Response(200, json=[{"position": 1, "line": "Hello"}])
        )

        settings = Settings(
            LINGARR_BASE_URL=AnyHttpUrl(LINGARR_BASE),
            LINGARR_API_KEY=api_key,
        )
        async with LingarrClient(settings) as lingarr:
            await lingarr.translate(
                lines=[(1, "Hola")],
                source_language="es",
                target_language="en",
                media_type="Episode",
                title="test",
                arr_media_id=0,
            )

    for record in caplog.records:
        assert api_key not in record.getMessage(), f"API key leaked in log: {record.getMessage()}"


# ---------------------------------------------------------------------------
# Criterion 27: contract replay — exact Bazarr task=translate request shape
# ---------------------------------------------------------------------------


def test_contract_replay_translate_shape() -> None:
    """Replay Bazarr's exact task=translate multipart shape."""
    params = {
        "task": "translate",
        "language": "es",
        "output": "srt",
        "encode": "false",
        "video_file": "/media/show/s01e01.mkv",
    }
    mock_lingarr = _mock_lingarr({1: "Hello world."})

    with (
        respx.mock,
        patch("whisper_proxy.routes.asr.align", new=AsyncMock(return_value=_FAKE_WORDS)),
        patch("whisper_proxy.routes.asr.get_lingarr_client", return_value=mock_lingarr),
    ):
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=_VERBOSE_JSON_ES))
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
    srt = pysrt.from_string(resp.text, error_handling=pysrt.ERROR_RAISE)
    assert len(srt) > 0
    # title_for is applied using video_file basename
    call_kwargs = mock_lingarr.translate.call_args.kwargs
    assert call_kwargs["title"] == "s01e01.mkv"
    assert call_kwargs["source_language"] == "es"
