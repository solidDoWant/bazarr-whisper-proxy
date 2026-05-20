"""Spec 16 criteria 12-14: Bazarr -> bridge task=translate (Lingarr path)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pysrt
import pytest

from tests.e2e.conftest import E2eConfig, bazarr_logs, wait_for_srt
from tests.e2e.provision.bazarr import BazarrProvisioner

_log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def spanish_clip(clips_by_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return clips_by_language["es"]


@pytest.fixture(scope="module")
def spanish_movie(
    spanish_clip: dict[str, Any], radarr_movie_by_tmdb: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return radarr_movie_by_tmdb[spanish_clip["tmdb_id"]]


@pytest.fixture(scope="module")
def spanish_movie_with_es_en_profile(
    bazarr: BazarrProvisioner,
    spanish_movie: dict[str, Any],
) -> dict[str, Any]:
    """Reassign the Spanish movie to a profile that wants both Spanish *and*
    English subtitles.

    Without this, Bazarr's missing_subtitles list only contains Spanish (the
    profile's only language), so ``GET /api/providers/movies`` returns no
    English candidates and the translate path can't be exercised.
    """
    profile_id = 99  # well above the per-language profile ids the provisioner uses
    profile = {
        "profileId": profile_id,
        "name": "E2E es+en (translate)",
        "items": [
            {
                "id": 1,
                "language": "es",
                "forced": "False",
                "hi": "False",
                "audio_exclude": "False",
                "audio_only_include": "False",
            },
            {
                "id": 2,
                "language": "en",
                "forced": "False",
                "hi": "False",
                "audio_exclude": "False",
                "audio_only_include": "False",
            },
        ],
        "cutoff": None,
        "mustContain": [],
        "mustNotContain": [],
        "originalFormat": False,
        "tag": "",
    }
    existing = bazarr._api().get_json("/api/system/languages/profiles")
    all_profiles = [p for p in existing if p.get("profileId") != profile_id] + [profile]
    bazarr._post_settings_form([("languages-profiles", json.dumps(all_profiles))])
    bazarr.assign_movie_profile(spanish_movie["id"], profile_id)
    return spanish_movie


def test_12_translated_subtitle_is_english(
    bazarr: BazarrProvisioner,
    spanish_clip: dict[str, Any],
    spanish_movie_with_es_en_profile: dict[str, Any],
    e2e_config: E2eConfig,
) -> None:
    """Criterion 12: Spanish audio + Bazarr asking for English subs → English .srt."""
    folder = e2e_config.media_host_root / spanish_clip["radarr_folder"]
    base = spanish_clip["filename"]

    # Ask Bazarr to fetch English subtitles for the Spanish fixture. Bazarr
    # decides task=translate because the requested language differs from the
    # fixture's detected/profile language.
    bazarr.trigger_subtitle_download(spanish_movie_with_es_en_profile["id"], "en")
    srt_path = wait_for_srt(folder, base, language_alpha2="en", timeout_sec=240)

    subs = pysrt.open(str(srt_path), encoding="utf-8")
    assert len(subs) >= 1, "expected at least one translated cue"

    text = " ".join(c.text.lower() for c in subs)
    expected = [w.lower() for w in spanish_clip["expected_translated_words"]]
    hits = [w for w in expected if w in text]
    assert hits, (
        f"translated text {text[:200]!r} contains none of expected {expected}; "
        "translation likely failed"
    )


def test_13_translated_srt_preserves_timing(
    bazarr: BazarrProvisioner,
    spanish_clip: dict[str, Any],
    spanish_movie_with_es_en_profile: dict[str, Any],
    e2e_config: E2eConfig,
) -> None:
    """Criterion 13: translated .srt has same cue count + identical timing as a transcribe pass.

    Run a transcribe pass (es subs for the Spanish fixture) to get a
    reference, then compare against the already-produced translated .srt.
    """
    folder = e2e_config.media_host_root / spanish_clip["radarr_folder"]
    base = spanish_clip["filename"]

    # Trigger Spanish (transcribe) — same source language as the audio.
    bazarr.trigger_subtitle_download(spanish_movie_with_es_en_profile["id"], "es")
    es_srt = wait_for_srt(folder, base, language_alpha2="es", timeout_sec=180)
    en_srt = wait_for_srt(folder, base, language_alpha2="en", timeout_sec=10)

    es_cues = pysrt.open(str(es_srt), encoding="utf-8")
    en_cues = pysrt.open(str(en_srt), encoding="utf-8")

    assert len(es_cues) == len(en_cues), (
        f"cue count differs: transcribe={len(es_cues)} translate={len(en_cues)}"
    )
    for i, (e, t) in enumerate(zip(es_cues, en_cues, strict=True)):
        assert e.start.ordinal == t.start.ordinal, (
            f"cue {i} start differs: transcribe={e.start} translate={t.start}"
        )
        assert e.end.ordinal == t.end.ordinal, (
            f"cue {i} end differs: transcribe={e.end} translate={t.end}"
        )


def test_14_bazarr_does_not_log_invalid_for_translated(
    spanish_clip: dict[str, Any], e2e_config: E2eConfig
) -> None:
    """Criterion 14: 'Downloaded subtitles isn't valid' absent for translated subs too."""
    folder = e2e_config.media_host_root / spanish_clip["radarr_folder"]
    wait_for_srt(folder, spanish_clip["filename"], language_alpha2="en", timeout_sec=10)

    bad = "Downloaded subtitles isn't valid for this file"
    assert bad not in bazarr_logs(), f"regression gate hit (translate): {bad!r}"
