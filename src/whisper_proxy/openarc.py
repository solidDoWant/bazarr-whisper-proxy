import json
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, cast

import httpx

from whisper_proxy.config import Settings


class OpenArcError(Exception):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "OpenArc error")
        self.detail = detail


class OpenArcUnavailable(OpenArcError):
    pass


class OpenArcBadRequest(OpenArcError):
    pass


class OpenArcInferenceError(OpenArcError):
    pass


@dataclass
class Transcription:
    text: str
    language: str | None
    duration: float | None
    metrics: dict[str, Any] = field(default_factory=dict)


class OpenArcClient:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.OPENARC_MODEL
        self._connect_timeout = float(settings.OPENARC_CONNECT_TIMEOUT)
        self._client = httpx.AsyncClient(
            base_url=str(settings.OPENARC_BASE_URL),
            timeout=httpx.Timeout(
                None,
                connect=self._connect_timeout,
                read=float(settings.OPENARC_READ_TIMEOUT),
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenArcClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def transcribe(self, audio_wav: bytes, language: str | None) -> Transcription:
        data: dict[str, str] = {
            "model": self._model,
            "response_format": "verbose_json",
        }
        if language is not None:
            data["openarc_asr"] = json.dumps({"qwen3_asr": {"language": language}})

        try:
            resp = await self._client.post(
                "/v1/audio/transcriptions",
                data=data,
                files={"file": ("audio.wav", audio_wav, "audio/wav")},
            )
        except httpx.TransportError as exc:
            raise OpenArcUnavailable(str(exc)) from exc

        self._check_status(resp)
        body = resp.json()

        return Transcription(
            text=body["text"],
            language=body.get("language"),
            duration=body.get("duration"),
            metrics=body.get("metrics") or {},
        )

    async def detect_language(self, audio_wav: bytes) -> str:
        tr = await self.transcribe(audio_wav, language=None)
        return str(tr.metrics.get("language", ""))

    async def model_state(self) -> Literal["loaded", "loading", "unloaded", "unknown"]:
        try:
            resp = await self._client.get(
                "/openarc/status",
                timeout=httpx.Timeout(None, connect=self._connect_timeout, read=5.0),
            )
        except httpx.TransportError as exc:
            raise OpenArcUnavailable(str(exc)) from exc

        self._check_status(resp)
        for entry in resp.json():
            if entry.get("model_name") != self._model:
                continue

            status = entry.get("status")
            if status in ("loaded", "loading", "unloaded"):
                return cast(Literal["loaded", "loading", "unloaded"], status)
            return "unknown"

        return "unknown"

    def _check_status(self, resp: httpx.Response) -> None:
        if resp.is_success:
            return

        detail: str | None = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail")
        except Exception:
            pass

        if resp.is_client_error:
            raise OpenArcBadRequest(detail)
        raise OpenArcInferenceError(detail)
