import httpx
import pytest
import respx

from whisper_proxy.config import Settings
from whisper_proxy.openarc import (
    OpenArcBadRequest,
    OpenArcClient,
    OpenArcInferenceError,
    OpenArcUnavailable,
    Transcription,
    _clean_text,
)

BASE = "http://openarc-test:8000"
TRANSCRIPTIONS_URL = f"{BASE}/v1/audio/transcriptions"
STATUS_URL = f"{BASE}/openarc/status"

FAKE_AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 28

VERBOSE_JSON = {
    "text": "Hello world.",
    "language": "English",
    "duration": None,
    "metrics": {
        "language": "English",
        "audio_duration_sec": 10.5,
    },
}


def make_settings(**kwargs: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,  # pyright: ignore[reportCallIssue]
        OPENARC_BASE_URL=BASE,
        OPENARC_MODEL="test-model",
        OPENARC_CONNECT_TIMEOUT=5,
        OPENARC_READ_TIMEOUT=30,
        **kwargs,
    )


# --- Qwen3-ASR tag stripping ---


def test_clean_text_strips_language_english_tag() -> None:
    assert _clean_text("language English<asr_text>Hello world.") == "Hello world."


def test_clean_text_strips_language_none_tag() -> None:
    assert _clean_text("language None<asr_text>Hey Clay, check me out.") == "Hey Clay, check me out."


def test_clean_text_strips_multiple_tags() -> None:
    raw = (
        "language English<asr_text>This is a sentence. "
        "language None<asr_text>Another sentence. "
        "language English<asr_text>Final words."
    )
    assert _clean_text(raw) == "This is a sentence. Another sentence. Final words."


def test_clean_text_no_space_at_segment_boundary() -> None:
    # Real Qwen3-ASR output has no space between segments; words must not merge.
    raw = "language English<asr_text>All right.language None<asr_text>Hey Clay."
    assert _clean_text(raw) == "All right. Hey Clay."


def test_clean_text_passthrough_when_no_tags() -> None:
    assert _clean_text("Plain transcript with no tags.") == "Plain transcript with no tags."


async def test_transcribe_strips_qwen_tags_from_text() -> None:
    tagged = {**VERBOSE_JSON, "text": "language English<asr_text>Hello world."}
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=tagged))
        async with OpenArcClient(make_settings()) as c:
            result = await c.transcribe(FAKE_AUDIO, language="en")
    assert result.text == "Hello world."


# --- Criterion 1: transcribe with language sends all required fields ---


async def test_transcribe_with_language_sends_language_param() -> None:
    captured: dict[str, bytes] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        captured["ct"] = request.headers["content-type"].encode()
        return httpx.Response(200, json=VERBOSE_JSON)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=capture)
        async with OpenArcClient(make_settings()) as c:
            await c.transcribe(FAKE_AUDIO, language="en")

    assert b"multipart/form-data" in captured["ct"]
    body = captured["body"]
    assert b"verbose_json" in body
    assert b"test-model" in body
    assert b'name="language"' in body
    assert b"en" in body
    assert b"openarc_asr" not in body
    assert b"audio.wav" in body
    assert FAKE_AUDIO in body


async def test_transcribe_passes_unknown_language_codes_through() -> None:
    """Any non-empty code is forwarded as-is; OpenArc handles unknown values."""
    captured: dict[str, bytes] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json=VERBOSE_JSON)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=capture)
        async with OpenArcClient(make_settings()) as c:
            await c.transcribe(FAKE_AUDIO, language="xx")

    assert b'name="language"' in captured["body"]
    assert b"xx" in captured["body"]


# --- Criterion 2: transcribe without language omits openarc_asr ---


async def test_transcribe_without_language_omits_openarc_asr() -> None:
    captured: dict[str, bytes] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json=VERBOSE_JSON)

    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=capture)
        async with OpenArcClient(make_settings()) as c:
            await c.transcribe(FAKE_AUDIO, language=None)

    assert b"openarc_asr" not in captured["body"]


# --- Criterion 3: Transcription exposes required fields ---


async def test_transcribe_returns_transcription_with_all_fields() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        async with OpenArcClient(make_settings()) as c:
            result = await c.transcribe(FAKE_AUDIO, language="en")

    assert isinstance(result, Transcription)
    assert result.text == "Hello world."
    assert result.language == "English"
    assert result.duration is None
    assert result.metrics["language"] == "English"
    assert result.metrics["audio_duration_sec"] == pytest.approx(10.5)
    assert result.segments == []


# --- Segment parsing: OpenArc returns segments at the top level (OpenAI shape) ---


async def test_transcribe_parses_segments_from_top_level() -> None:
    body = {
        **VERBOSE_JSON,
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "language English<asr_text>Hello world.",
            },
            {
                "start": 1.5,
                "end": 3.25,
                "text": "language None<asr_text>Second segment.",
            },
        ],
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=body))
        async with OpenArcClient(make_settings()) as c:
            result = await c.transcribe(FAKE_AUDIO, language="en")

    assert len(result.segments) == 2
    assert result.segments[0].start_sec == pytest.approx(0.0)
    assert result.segments[0].end_sec == pytest.approx(1.5)
    assert result.segments[0].text == "Hello world."
    assert result.segments[1].start_sec == pytest.approx(1.5)
    assert result.segments[1].end_sec == pytest.approx(3.25)
    assert result.segments[1].text == "Second segment."


async def test_transcribe_ignores_segments_under_metrics() -> None:
    # Older OpenArc builds nested segments under `metrics`; the proxy now only
    # reads the OpenAI-shaped top-level field.
    body = {
        **VERBOSE_JSON,
        "metrics": {
            **VERBOSE_JSON["metrics"],
            "segments": [{"start": 0.0, "end": 1.0, "text": "Stale."}],
        },
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=body))
        async with OpenArcClient(make_settings()) as c:
            result = await c.transcribe(FAKE_AUDIO, language="en")

    assert result.segments == []


async def test_transcribe_skips_segments_with_missing_fields() -> None:
    body = {
        **VERBOSE_JSON,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Good."},
            {"start": 1.0, "text": "No end."},
            {"end": 2.0, "text": "No start."},
            {"start": 2.0, "end": 3.0},  # no text
            {"start": 3.0, "end": 4.0, "text": "language English<asr_text>"},  # cleans to ""
        ],
    }
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=body))
        async with OpenArcClient(make_settings()) as c:
            result = await c.transcribe(FAKE_AUDIO, language="en")

    assert len(result.segments) == 1
    assert result.segments[0].text == "Good."


# --- Criterion 4: model_state returns status or "unknown" ---


async def test_model_state_loaded() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "test-model", "status": "loaded"}]
            )
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "loaded"


async def test_model_state_loading() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "test-model", "status": "loading"}]
            )
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "loading"


async def test_model_state_unloaded() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "test-model", "status": "unloaded"}]
            )
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "unloaded"


async def test_model_state_model_not_in_array_returns_unknown() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200, json=[{"model_name": "other-model", "status": "loaded"}]
            )
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "unknown"


async def test_model_state_empty_array_returns_unknown() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(return_value=httpx.Response(200, json=[]))
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "unknown"


async def test_model_state_unrecognised_status_returns_unknown() -> None:
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(200, json=[{"model_name": "test-model", "status": "error"}])
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "unknown"


async def test_model_state_dict_wrapped_response() -> None:
    """OpenArc 1.x wraps entries in {"models": [...]} — accept both shapes."""
    with respx.mock:
        respx.get(STATUS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "total_loaded_models": 1,
                    "models": [{"model_name": "test-model", "status": "loaded"}],
                },
            )
        )
        async with OpenArcClient(make_settings()) as c:
            assert await c.model_state() == "loaded"


# --- Criterion 5: 4xx → OpenArcBadRequest with detail ---


async def test_4xx_raises_bad_request_with_detail() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(400, json={"detail": "Model not loaded"})
        )
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcBadRequest) as exc_info:
                await c.transcribe(FAKE_AUDIO, language="en")

    assert exc_info.value.detail == "Model not loaded"


async def test_404_raises_bad_request() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(404, json={"detail": "Not found"})
        )
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcBadRequest):
                await c.transcribe(FAKE_AUDIO, language=None)


# --- Criterion 6: 5xx → OpenArcInferenceError with detail ---


async def test_5xx_raises_inference_error_with_detail() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(500, json={"detail": "Transcription failed: OOM"})
        )
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcInferenceError) as exc_info:
                await c.transcribe(FAKE_AUDIO, language="en")

    assert exc_info.value.detail == "Transcription failed: OOM"


async def test_503_raises_inference_error() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            return_value=httpx.Response(503, json={"detail": "Service unavailable"})
        )
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcInferenceError):
                await c.transcribe(FAKE_AUDIO, language=None)


# --- Criterion 7: connection refused → OpenArcUnavailable ---


async def test_connect_error_raises_unavailable() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcUnavailable):
                await c.transcribe(FAKE_AUDIO, language="en")


async def test_dns_failure_raises_unavailable() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(
            side_effect=httpx.ConnectError("Name or service not known")
        )
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcUnavailable):
                await c.transcribe(FAKE_AUDIO, language=None)


# --- Criterion 8: connect timeout → OpenArcUnavailable ---


async def test_connect_timeout_raises_unavailable() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(side_effect=httpx.ConnectTimeout("Connect timed out"))
        async with OpenArcClient(make_settings()) as c:
            with pytest.raises(OpenArcUnavailable):
                await c.transcribe(FAKE_AUDIO, language="en")


# --- Criterion 9: no socket leaks ---
# All tests use `async with OpenArcClient(...) as c:` which calls aclose().
# pytest-asyncio with asyncio_mode=auto will surface unclosed-resource warnings
# as test failures if aclose() is not called.


async def test_client_is_reusable_across_requests() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        async with OpenArcClient(make_settings()) as c:
            r1 = await c.transcribe(FAKE_AUDIO, language="en")
            r2 = await c.transcribe(FAKE_AUDIO, language="fr")

    assert r1.text == r2.text == "Hello world."


# --- detect_language delegates to transcribe and returns metrics.language ---


async def test_detect_language_returns_metrics_language() -> None:
    with respx.mock:
        respx.post(TRANSCRIPTIONS_URL).mock(return_value=httpx.Response(200, json=VERBOSE_JSON))
        async with OpenArcClient(make_settings()) as c:
            lang = await c.detect_language(FAKE_AUDIO)

    assert lang == "English"
