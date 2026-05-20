"""Lingarr translation client (Phase 2 — task=translate)."""

import os
import zlib
from types import TracebackType

import httpx

from whisper_proxy.config import Settings

_NBSP = "\u00a0"


class LingarrError(Exception):
    pass


class LingarrUnavailable(LingarrError):
    pass


class LingarrBadRequest(LingarrError):
    pass


class LingarrServerError(LingarrError):
    pass


class LingarrInvalidResponse(LingarrError):
    pass


class LingarrCountMismatch(LingarrError):
    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"expected {expected}, received {received}")
        self.expected = expected
        self.received = received


class LingarrPositionMismatch(LingarrError):
    def __init__(self, position: int) -> None:
        super().__init__(f"unexpected position {position}")
        self.position = position


def arr_media_id_for(video_file: str | None) -> int:
    """Deterministic 31-bit ID from video_file path; 0 when absent.

    CRC-32 masked to 31 bits stays within Lingarr's int32 constraint.
    """
    if video_file is None:
        return 0
    return zlib.crc32(video_file.encode()) & 0x7FFF_FFFF


def title_for(video_file: str | None) -> str:
    if video_file is None:
        return "bazarr-whisper-proxy"
    return os.path.basename(video_file)


class LingarrClient:
    def __init__(self, settings: Settings) -> None:
        assert settings.LINGARR_BASE_URL is not None
        # X-Api-Key is set on the client and never logged.
        self._client = httpx.AsyncClient(
            base_url=str(settings.LINGARR_BASE_URL),
            headers={"X-Api-Key": settings.LINGARR_API_KEY},
            timeout=httpx.Timeout(float(settings.LINGARR_TIMEOUT)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LingarrClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def translate(
        self,
        lines: list[tuple[int, str]],
        source_language: str,
        target_language: str,
        media_type: str,
        title: str,
        arr_media_id: int,
    ) -> dict[int, str]:
        """POST to /api/Translate/content; return position → translated text.

        Raises LingarrCountMismatch or LingarrPositionMismatch on reconciliation
        failures, LingarrInvalidResponse on malformed replies, and
        LingarrUnavailable / LingarrBadRequest / LingarrServerError for transport
        and HTTP-level errors.
        """
        sent_positions = {pos for pos, _ in lines}

        payload_lines = [
            {"position": pos, "line": text if text.strip() else _NBSP} for pos, text in lines
        ]

        body = {
            "arrMediaId": arr_media_id,
            "title": title,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
            "mediaType": media_type,
            "lines": payload_lines,
        }

        try:
            resp = await self._client.post("/api/Translate/content", json=body)
        except httpx.TransportError as exc:
            raise LingarrUnavailable(str(exc)) from exc

        if not resp.is_success:
            if resp.is_client_error:
                raise LingarrBadRequest(f"Lingarr HTTP {resp.status_code}")
            raise LingarrServerError(f"Lingarr HTTP {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:
            raise LingarrInvalidResponse("non-JSON response from Lingarr") from exc

        if not isinstance(data, list):
            raise LingarrInvalidResponse("expected array response from Lingarr")

        result: dict[int, str] = {}
        for item in data:
            pos = item["position"]
            if pos not in sent_positions:
                raise LingarrPositionMismatch(pos)
            result[pos] = item["line"]

        if len(result) < len(sent_positions):
            raise LingarrCountMismatch(expected=len(sent_positions), received=len(result))

        return result
