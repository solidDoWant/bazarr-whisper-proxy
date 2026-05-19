import json

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


# --- Criterion 1: transcribe with language sends all required fields ---


async def test_transcribe_with_language_sends_openarc_asr() -> None:
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
    assert json.dumps({"qwen3_asr": {"language": "en"}}).encode() in body
    assert b"audio.wav" in body
    assert FAKE_AUDIO in body


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
