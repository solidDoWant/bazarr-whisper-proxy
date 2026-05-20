"""CLI entry for the e2e provisioning step.

Usage (from scripts/e2e.sh, after `docker compose up -d`):
    python -m tests.e2e.provision

All connection details come from environment variables. The script is
idempotent — running it twice against the same containers leaves the
same configuration state (spec 16 criterion 21).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from tests.e2e.fixtures.build import build_all
from tests.e2e.provision.bazarr import BazarrProvisioner
from tests.e2e.provision.lingarr import LingarrProvisioner
from tests.e2e.provision.radarr import RadarrProvisioner

_log = logging.getLogger("e2e.provision")


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise SystemExit(f"required env var {name} is unset")
    return val


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    radarr_url = _env("RADARR_URL", "http://127.0.0.1:7878")
    bazarr_url = _env("BAZARR_URL", "http://127.0.0.1:6767")
    lingarr_url = _env("LINGARR_URL", "http://127.0.0.1:9876")
    libretranslate_in_compose = _env("LIBRETRANSLATE_IN_COMPOSE_URL", "http://libretranslate:5000")
    bridge_in_compose = _env("BRIDGE_IN_COMPOSE_URL", "http://whisper-proxy:9000")
    radarr_in_compose = _env("RADARR_IN_COMPOSE_URL", "http://radarr:7878")
    media_host_root = _env("MEDIA_HOST_ROOT")
    media_container_root = _env("MEDIA_CONTAINER_ROOT", "/media")

    _log.info("Building fixture media (into the repo; will be staged into the volume per-movie)")
    fixture_paths = build_all()
    _log.info("Built %d fixture file(s): %s", len(fixture_paths), [p.name for p in fixture_paths])

    Path(media_host_root).mkdir(parents=True, exist_ok=True)

    radarr = RadarrProvisioner(radarr_url)
    bazarr = BazarrProvisioner(bazarr_url)
    lingarr = LingarrProvisioner(lingarr_url)

    # 1. Radarr setup ------------------------------------------------------
    _log.info("Waiting for Radarr at %s", radarr_url)
    radarr.wait_ready()
    radarr_api_key = radarr.bootstrap()
    _log.info("Radarr ready (api_key=%s…)", radarr_api_key[:6])

    radarr.ensure_root_folder(media_container_root)

    clips = radarr.load_manifest()
    movie_records = []
    for clip in clips:
        m = radarr.ensure_movie_imported(clip, media_container_root)
        movie_records.append((clip, m))
    _log.info("Radarr movies provisioned: %s", [m["id"] for _, m in movie_records])

    # 2. Bazarr setup ------------------------------------------------------
    _log.info("Waiting for Bazarr at %s", bazarr_url)
    bazarr.wait_ready()
    bazarr_api_key = bazarr.bootstrap()
    _log.info("Bazarr ready (api_key=%s…)", bazarr_api_key[:6])

    bazarr.ensure_radarr_connection(radarr_in_compose, radarr_api_key)
    bazarr.ensure_whisper_provider(bridge_in_compose)

    # Create one language profile per fixture clip (id derived from index).
    # Bazarr requires per-movie profile assignment to know which language to
    # request from the provider.
    profile_for_lang: dict[str, int] = {}
    for idx, (clip, _movie) in enumerate(movie_records, start=1):
        lang = clip["language"]
        if lang in profile_for_lang:
            continue
        profile_id = idx
        bazarr.ensure_language_profile(
            profile_id=profile_id, name=f"E2E {lang}", language_alpha2=lang
        )
        profile_for_lang[lang] = profile_id

    bazarr.trigger_radarr_sync()

    # Wait for Bazarr's view of the Radarr inventory to match.
    _wait_bazarr_inventory(bazarr, expected_count=len(movie_records))

    # Assign per-movie language profiles. We re-resolve Bazarr's view of the
    # movie ids since Bazarr stores its own ids that mirror Radarr's.
    bazarr_movies = {m["radarrId"]: m for m in bazarr.list_movies() if "radarrId" in m}
    for clip, movie in movie_records:
        b_movie = bazarr_movies.get(movie["id"])
        if not b_movie:
            _log.warning(
                "Bazarr has no entry yet for radarrid=%d; skipping profile assign", movie["id"]
            )
            continue
        profile_id = profile_for_lang[clip["language"]]
        bazarr.assign_movie_profile(movie["id"], profile_id)

    # 3. Lingarr setup -----------------------------------------------------
    _log.info("Waiting for Lingarr at %s", lingarr_url)
    lingarr.wait_ready()
    lingarr.ensure_libretranslate_backend(libretranslate_in_compose)

    _log.info("Provisioning complete.")
    return 0


def _wait_bazarr_inventory(
    bazarr: BazarrProvisioner, expected_count: int, timeout_sec: float = 120.0
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        movies = bazarr.list_movies()
        if len(movies) >= expected_count:
            _log.info("Bazarr sees %d movie(s) (expected %d)", len(movies), expected_count)
            return
        _log.debug("Bazarr inventory: %d/%d", len(movies), expected_count)
        time.sleep(3)
    raise TimeoutError(f"Bazarr did not sync {expected_count} movies within {timeout_sec}s")


if __name__ == "__main__":
    sys.exit(main())
