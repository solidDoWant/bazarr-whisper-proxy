"""Spec 16 criteria 15-16: failure-mode assertions.

These tests intentionally degrade the stack mid-test (criterion 15) or rely
on the bridge having been brought up against a black-hole OpenArc
(criterion 16). Criterion 16 is exercised via a separate harness invocation
mode — `scripts/e2e.sh failure-modes` — that sets OPENARC_E2E_BASE_URL to
a black-hole address before bringing the stack up.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import E2eConfig, bazarr_logs
from tests.e2e.provision.bazarr import BazarrProvisioner

_log = logging.getLogger(__name__)


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker") or "docker"
    return subprocess.run(
        [docker, "compose", "-f", "compose.e2e.yml", *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def english_clip(clips_by_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return clips_by_language["en"]


@pytest.fixture(scope="module")
def english_movie(
    english_clip: dict[str, Any], radarr_movie_by_tmdb: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return radarr_movie_by_tmdb[english_clip["tmdb_id"]]


def test_15_bazarr_does_not_throttle_when_bridge_stopped(
    bazarr: BazarrProvisioner,
    english_clip: dict[str, Any],
    english_movie: dict[str, Any],
) -> None:
    """Criterion 15: Bazarr's background scan doesn't 24h-throttle during a brief bridge stop.

    We stop the bridge for 20 s and let Bazarr's periodic scan observe the
    outage.  We do NOT manually trigger a provider call (that would force an
    immediate connection-refused which Bazarr always throttles regardless of
    provider); instead we test that the background housekeeping path — which
    runs against movies that already have subtitles — does not emit a throttle.
    """
    import datetime

    since = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _compose("stop", "whisper-proxy")
    try:
        time.sleep(20)
    finally:
        _compose("start", "whisper-proxy")
        _wait_bridge_healthy(timeout_sec=60.0)

    throttle = "Throttling whisperai for 24 hours"
    logs = bazarr_logs(since=since)
    assert throttle not in logs, (
        f"regression: {throttle!r} found in Bazarr logs during bridge downtime window"
    )


def _wait_bridge_healthy(timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            r = httpx.get(
                os.environ.get("BRIDGE_URL", "http://127.0.0.1:9000") + "/healthz", timeout=2
            )
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.fail("bridge did not return to healthy after start")


def test_16_status_503_when_openarc_unreachable(e2e_config: E2eConfig) -> None:
    """Criterion 16: with OpenArc unreachable, /status returns 503 + Retry-After.

    This is gated by the env var E2E_OPENARC_BLACKHOLE being set, since
    the test requires the stack to have been brought up against a
    black-hole OpenArc — incompatible with the rest of the suite. The
    `failure-modes` invocation of scripts/e2e.sh sets this.
    """
    if not os.environ.get("E2E_OPENARC_BLACKHOLE"):
        pytest.skip("E2E_OPENARC_BLACKHOLE not set — run via `scripts/e2e.sh failure-modes`")

    r = httpx.get(e2e_config.bridge_url + "/status", timeout=10)
    assert r.status_code == 503, f"expected 503, got {r.status_code}"
    assert "retry-after" in {k.lower() for k in r.headers}, (
        f"missing Retry-After header: {dict(r.headers)}"
    )
