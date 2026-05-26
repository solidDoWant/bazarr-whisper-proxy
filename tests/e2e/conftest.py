"""Shared fixtures for the e2e suite.

These tests assume `scripts/e2e.sh` (or equivalent) has already brought the
compose stack up and run the provisioning step. The fixtures here just
wrap connection details and provide convenience accessors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.fixtures.build import MANIFEST_PATH
from tests.e2e.provision.bazarr import BazarrProvisioner
from tests.e2e.provision.radarr import RadarrProvisioner

_log = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        pytest.skip(
            f"e2e env var {name} not set — run via scripts/e2e.sh", allow_module_level=False
        )
        raise AssertionError("unreachable")
    return val


@dataclass(frozen=True)
class E2eConfig:
    radarr_url: str
    bazarr_url: str
    bridge_url: str
    lingarr_url: str | None
    media_host_root: Path
    clips: list[dict[str, Any]]


@pytest.fixture(scope="session")
def e2e_config() -> E2eConfig:
    radarr_url = _env("RADARR_URL", "http://127.0.0.1:7878")
    bazarr_url = _env("BAZARR_URL", "http://127.0.0.1:6767")
    bridge_url = _env("BRIDGE_URL", "http://127.0.0.1:9000")
    lingarr_url = os.environ.get("LINGARR_URL")
    media_host_root = Path(_env("MEDIA_HOST_ROOT"))

    clips = list(json.loads(MANIFEST_PATH.read_text())["clips"])
    return E2eConfig(
        radarr_url=radarr_url,
        bazarr_url=bazarr_url,
        bridge_url=bridge_url,
        lingarr_url=lingarr_url,
        media_host_root=media_host_root,
        clips=clips,
    )


@pytest.fixture(scope="session")
def radarr(e2e_config: E2eConfig) -> RadarrProvisioner:
    r = RadarrProvisioner(e2e_config.radarr_url)
    r.bootstrap()
    return r


@pytest.fixture(scope="session")
def bazarr(e2e_config: E2eConfig) -> BazarrProvisioner:
    b = BazarrProvisioner(e2e_config.bazarr_url)
    b.bootstrap()
    return b


@pytest.fixture(scope="session")
def clips_by_language(e2e_config: E2eConfig) -> dict[str, dict[str, Any]]:
    return {c["language"]: c for c in e2e_config.clips}


@pytest.fixture(scope="session")
def radarr_movie_by_tmdb(radarr: RadarrProvisioner) -> dict[int, dict[str, Any]]:
    return {m["tmdbId"]: m for m in radarr.list_movies()}


# -------- helpers used by tests ------------------------------------


def wait_for_srt(
    movie_folder: Path,
    base_filename: str,
    *,
    language_alpha2: str | None = None,
    timeout_sec: float = 180.0,
) -> Path:
    """Poll the movie folder for an .srt that matches the fixture's base name."""
    deadline = time.monotonic() + timeout_sec
    stem = Path(base_filename).stem
    while time.monotonic() < deadline:
        for candidate in movie_folder.glob(f"{stem}*.srt"):
            if language_alpha2 is None or f".{language_alpha2}" in candidate.name:
                return candidate
        time.sleep(2)
    raise TimeoutError(
        f"no .srt in {movie_folder} within {timeout_sec}s (looking for {stem}*.srt, lang={language_alpha2})"
    )


def bazarr_logs(since: str | None = None) -> str:
    """Return Bazarr's docker logs.

    ``since`` is an RFC 3339 timestamp (e.g. ``"2026-05-20T15:30:00Z"``);
    when given only logs from that point forward are returned.
    """
    import shutil
    import subprocess

    docker = shutil.which("docker") or "docker"
    cmd = [docker, "compose", "-f", "compose.e2e.yml", "logs", "--no-color", "bazarr"]
    if since:
        cmd.extend(["--since", since])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout + out.stderr


def bridge_logs() -> str:
    import shutil
    import subprocess

    docker = shutil.which("docker") or "docker"
    out = subprocess.run(
        [docker, "compose", "-f", "compose.e2e.yml", "logs", "--no-color", "whisper-proxy"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout + out.stderr


def http_client(base_url: str, *, timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)
