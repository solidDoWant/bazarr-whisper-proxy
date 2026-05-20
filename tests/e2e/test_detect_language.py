"""Spec 16 criteria 10-11: Bazarr -> bridge /detect-language path.

We exercise the bridge's /detect-language route directly with the Spanish
fixture audio. We additionally assert the response shape per criterion 11.
The full Bazarr-side detection flow varies across Bazarr versions (the
'force language detection' option migrated between providers); the direct
exercise is what's load-bearing for the contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import E2eConfig

_log = logging.getLogger(__name__)


def _extract_pcm(media_path: Path) -> bytes:
    """Convert MKV → s16le mono 16kHz raw PCM via ffmpeg (matches the format
    Bazarr sends to the bridge)."""
    import subprocess

    out = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return out.stdout


@pytest.fixture(scope="module")
def spanish_clip(clips_by_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return clips_by_language["es"]


def test_10_and_11_detect_language_returns_es(
    spanish_clip: dict[str, Any], e2e_config: E2eConfig
) -> None:
    """Criteria 10 + 11: /detect-language returns a JSON shape with the right alpha-2."""
    media_path = (
        e2e_config.media_host_root / spanish_clip["radarr_folder"] / spanish_clip["filename"]
    )
    pcm = _extract_pcm(media_path)

    files = {"audio_file": ("audio.pcm", pcm, "application/octet-stream")}
    with httpx.Client(base_url=e2e_config.bridge_url, timeout=180.0) as client:
        resp = client.post("/detect-language", files=files)

    assert resp.status_code == 200, f"bridge returned {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    # Criterion 11: contract-mandated keys
    assert "language_code" in body, body
    assert "detected_language" in body, body
    # Criterion 10: correct language for the fixture
    assert body["language_code"] == "es", (
        f"expected language_code=es, got {body['language_code']} (response={body!r})"
    )
