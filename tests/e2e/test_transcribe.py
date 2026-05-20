"""Spec 16 criteria 6-9: Bazarr -> bridge task=transcribe path."""

from __future__ import annotations

import logging
import time
from typing import Any

import pysrt
import pytest

from tests.e2e.conftest import E2eConfig, bazarr_logs, wait_for_srt
from tests.e2e.provision.bazarr import BazarrProvisioner

_log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def english_clip(clips_by_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return clips_by_language["en"]


@pytest.fixture(scope="module")
def english_movie(
    english_clip: dict[str, Any], radarr_movie_by_tmdb: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return radarr_movie_by_tmdb[english_clip["tmdb_id"]]


def test_6_subtitle_file_appears_within_timeout(
    bazarr: BazarrProvisioner,
    english_clip: dict[str, Any],
    english_movie: dict[str, Any],
    e2e_config: E2eConfig,
) -> None:
    """Criterion 6: triggering a manual search produces an .srt within 3 minutes."""
    folder = e2e_config.media_host_root / english_clip["radarr_folder"]
    base = english_clip["filename"]

    # Capture wall-clock for criterion 17.
    start = time.monotonic()
    bazarr.trigger_subtitle_download(english_movie["id"], "en")
    srt_path = wait_for_srt(folder, base, language_alpha2="en", timeout_sec=180)
    elapsed = time.monotonic() - start
    _log.info("transcribe wall-clock: %.1fs (path=%s)", elapsed, srt_path)

    # Criterion 17 sanity: WARN if slow, don't fail.
    if elapsed > 90.0:
        _log.warning("transcribe took %.1fs (> 90s sanity bound)", elapsed)


def test_7_srt_parses_cleanly(english_clip: dict[str, Any], e2e_config: E2eConfig) -> None:
    """Criterion 7: pysrt.from_string(..., ERROR_RAISE) accepts the file."""
    folder = e2e_config.media_host_root / english_clip["radarr_folder"]
    base = english_clip["filename"]
    srt_path = wait_for_srt(folder, base, language_alpha2="en", timeout_sec=10)

    # Force strict parsing — this is the regression gate for the
    # "Downloaded subtitles isn't valid" failure mode.
    subs = pysrt.from_string(srt_path.read_text(encoding="utf-8"), error_handling=pysrt.ERROR_RAISE)
    assert len(subs) >= 1, "expected at least one cue"


def test_8_first_cue_contains_expected_word(
    english_clip: dict[str, Any], e2e_config: E2eConfig
) -> None:
    """Criterion 8: first cue's text contains an expected word (loose check)."""
    folder = e2e_config.media_host_root / english_clip["radarr_folder"]
    base = english_clip["filename"]
    srt_path = wait_for_srt(folder, base, language_alpha2="en", timeout_sec=10)
    subs = pysrt.open(str(srt_path), encoding="utf-8")

    first_cue_text = " ".join(subs[0].text.lower().split())
    expected: list[str] = [w.lower() for w in english_clip["expected_words"]]
    hits = [w for w in expected if w in first_cue_text]
    assert hits, (
        f"first cue {first_cue_text!r} contains none of expected {expected}; "
        "OpenArc output likely wrong"
    )


def test_9_bazarr_does_not_log_invalid_subtitles(
    english_clip: dict[str, Any], e2e_config: E2eConfig
) -> None:
    """Criterion 9: Bazarr's logs contain no 'Downloaded subtitles isn't valid for this file'."""
    # Ensure the subtitle has been generated before we read logs.
    folder = e2e_config.media_host_root / english_clip["radarr_folder"]
    wait_for_srt(folder, english_clip["filename"], language_alpha2="en", timeout_sec=10)

    logs = bazarr_logs()
    bad = "Downloaded subtitles isn't valid for this file"
    assert bad not in logs, f"regression gate hit: {bad!r} in Bazarr logs"
