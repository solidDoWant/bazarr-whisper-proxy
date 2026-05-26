"""Live integration test for LingarrClient against a real Lingarr instance.

Skipped unless LINGARR_URL is set in the environment.  Does not require the
full compose stack (Bazarr / Radarr) — exercises LingarrClient directly.

Run:
    LINGARR_URL=http://lingarr.media.svc.cluster.local:8080 \
      pytest tests/e2e/test_lingarr_live.py -v
"""

from __future__ import annotations

import os

import pytest

from whisper_proxy.config import Settings
from whisper_proxy.lingarr import LingarrClient, arr_media_id_for

_LINGARR_URL = os.environ.get("LINGARR_URL")

pytestmark = pytest.mark.skipif(
    not _LINGARR_URL,
    reason="LINGARR_URL not set — pass a reachable Lingarr base URL to run live tests",
)

# Spanish pangram from the e2e fixture manifest.
_ES_LINES: list[tuple[int, str]] = [
    (0, "El veloz murciélago hindú comía feliz cardillo y kiwi."),
    (1, "La cigüeña tocaba el saxofón detrás del palenque de paja."),
]

# Words that should appear in a correct es→en translation.
_EXPECTED_EN_WORDS = ["bat", "kiwi", "stork"]


@pytest.fixture(scope="module")
def lingarr_settings() -> Settings:
    assert _LINGARR_URL is not None
    return Settings(
        LINGARR_BASE_URL=_LINGARR_URL,  # type: ignore[arg-type]
        LINGARR_API_KEY="e2e-live-placeholder",
    )


@pytest.mark.asyncio
async def test_translate_es_to_en(lingarr_settings: Settings) -> None:
    """LingarrClient can translate the fixture Spanish pangram to English."""
    async with LingarrClient(lingarr_settings) as client:
        result = await client.translate(
            lines=_ES_LINES,
            source_language="es",
            target_language="en",
            media_type="Movie",
            title="e2e-live-test",
            arr_media_id=arr_media_id_for("/media/Night.of.the.Living.Dead.1968.mkv"),
        )

    assert len(result) == len(_ES_LINES), f"expected {len(_ES_LINES)} cues, got {len(result)}"

    full_text = " ".join(result[pos] for pos, _ in _ES_LINES).lower()
    hits = [w for w in _EXPECTED_EN_WORDS if w in full_text]
    assert hits, (
        f"translation {full_text!r} contains none of expected {_EXPECTED_EN_WORDS}; "
        "service may not be translating"
    )


@pytest.mark.asyncio
async def test_translate_preserves_position_mapping(lingarr_settings: Settings) -> None:
    """Result positions map 1:1 with sent positions."""
    async with LingarrClient(lingarr_settings) as client:
        result = await client.translate(
            lines=_ES_LINES,
            source_language="es",
            target_language="en",
            media_type="Movie",
            title="e2e-live-test",
            arr_media_id=0,
        )

    for pos, _ in _ES_LINES:
        assert pos in result, f"position {pos} missing from result"
        assert result[pos].strip(), f"position {pos} is blank"


@pytest.mark.asyncio
async def test_translate_empty_line_roundtrip(lingarr_settings: Settings) -> None:
    """Empty / whitespace-only lines are passed through as empty strings, not sent to Lingarr.

    Lingarr backends (localai, libretranslate) reject whitespace-only lines with 400.
    LingarrClient skips them and fills the result with "" for those positions.
    """
    lines: list[tuple[int, str]] = [(0, "Hola mundo."), (1, "")]
    async with LingarrClient(lingarr_settings) as client:
        result = await client.translate(
            lines=lines,
            source_language="es",
            target_language="en",
            media_type="Movie",
            title="e2e-live-test",
            arr_media_id=0,
        )

    assert 0 in result, "position 0 missing"
    assert 1 in result, "position 1 (empty passthrough) missing"
    assert result[0].strip(), "position 0 should have translated text"
    assert result[1] == "", "position 1 (empty input) should pass through as empty string"
