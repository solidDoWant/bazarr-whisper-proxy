"""Spec 16 criteria 18-22: provisioning assertions."""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.e2e.conftest import E2eConfig
from tests.e2e.provision.bazarr import BazarrProvisioner
from tests.e2e.provision.radarr import RadarrProvisioner


def test_18_radarr_has_one_movie_per_fixture(
    radarr: RadarrProvisioner, e2e_config: E2eConfig
) -> None:
    """Criterion 18: Radarr's /api/v3/movie returns one entry per fixture file."""
    movies = radarr.list_movies()
    fixture_tmdb_ids = {c["tmdb_id"] for c in e2e_config.clips}
    radarr_tmdb_ids = {m["tmdbId"] for m in movies}
    assert fixture_tmdb_ids.issubset(radarr_tmdb_ids), (
        f"missing in Radarr: {fixture_tmdb_ids - radarr_tmdb_ids}"
    )

    media_root = "/media"
    for clip in e2e_config.clips:
        matching = [m for m in movies if m["tmdbId"] == clip["tmdb_id"]]
        assert matching, f"no Radarr movie for clip {clip['slug']}"
        assert matching[0]["path"].startswith(media_root), (
            f"movie path {matching[0]['path']!r} not under root {media_root}"
        )


def test_19_bazarr_inventory_matches_radarr(
    bazarr: BazarrProvisioner, radarr: RadarrProvisioner
) -> None:
    """Criterion 19: Bazarr's /api/movies has the same set as Radarr's."""
    radarr_ids = {m["id"] for m in radarr.list_movies()}
    bazarr_ids = {m["radarrId"] for m in bazarr.list_movies() if "radarrId" in m}
    assert radarr_ids.issubset(bazarr_ids), f"Bazarr missing radarrIds {radarr_ids - bazarr_ids}"


def test_20_bazarr_whisperai_provider_enabled(
    bazarr: BazarrProvisioner, e2e_config: E2eConfig
) -> None:
    """Criterion 20: Bazarr settings show whisperai enabled + pointed at the bridge."""
    settings = bazarr.fetch_settings()
    enabled = settings["general"]["enabled_providers"]
    assert "whisperai" in enabled, f"whisperai not in enabled providers: {enabled}"

    endpoint = settings["whisperai"]["endpoint"].rstrip("/")
    # The compose-side URL we configured (http://whisper-proxy:9000) — confirms
    # the provisioner wrote the in-stack address, not the host-side one.
    assert endpoint.endswith(":9000"), f"unexpected whisperai endpoint: {endpoint}"


def test_21_provisioning_is_idempotent(
    bazarr: BazarrProvisioner, radarr: RadarrProvisioner, e2e_config: E2eConfig
) -> None:
    """Criterion 21: a second provision run leaves state unchanged.

    We invoke the same provisioner methods the CLI does, then compare the
    Radarr/Bazarr inventories before and after.
    """
    # Before
    r_count = len(radarr.list_movies())
    b_count = len(bazarr.list_movies())
    b_providers_before = list(bazarr.fetch_settings()["general"]["enabled_providers"])
    r_roots_before = [r["path"] for r in radarr._api().get_json("/api/v3/rootfolder")]

    # Re-run the idempotent operations
    radarr.ensure_root_folder("/media")
    for clip in e2e_config.clips:
        radarr.ensure_movie_imported(clip, "/media")
    bazarr.ensure_radarr_connection("http://radarr:7878", radarr.api_key)
    bazarr.ensure_whisper_provider("http://whisper-proxy:9000")

    # After
    assert len(radarr.list_movies()) == r_count, "Radarr movie count grew on re-provision"
    assert len(bazarr.list_movies()) == b_count, "Bazarr movie count grew on re-provision"
    assert bazarr.fetch_settings()["general"]["enabled_providers"] == b_providers_before, (
        "Bazarr enabled_providers changed on re-provision"
    )
    assert [r["path"] for r in radarr._api().get_json("/api/v3/rootfolder")] == r_roots_before, (
        "Radarr rootfolder list changed on re-provision"
    )


def test_22_image_tags_pinned_in_compose() -> None:
    """Criterion 22: every service in compose.e2e.yml is pinned (no :latest)."""
    compose = yaml.safe_load(Path("compose.e2e.yml").read_text())
    for svc_name, svc in compose["services"].items():
        image = svc.get("image", "")
        # Allow our own image (built locally, tag set by the flake), but
        # require everything else to be pinned.
        if svc_name == "whisper-proxy":
            continue
        assert ":" in image, f"{svc_name} has no tag: {image!r}"
        tag = image.rsplit(":", 1)[1]
        assert tag != "latest", f"{svc_name} pinned to :latest in compose.e2e.yml"
        assert tag != "", f"{svc_name} has empty tag"
